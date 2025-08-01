#!/usr/bin/env python3
"""
AetherNet Model Release Script

Converts trained AetherNet models to optimized deployment formats using the modular architecture.
"""

import argparse
import os
import sys
import time
import logging
import warnings
import platform
import traceback
from pathlib import Path
from copy import deepcopy
from typing import Callable, Any, List, Optional  # Fixed: Added Optional import

import torch
import numpy as np
from PIL import Image
import onnxruntime as ort

# Import from modular implementation
from aethernet_arch import AetherNet
from aethernet_utils import (
    export_onnx, 
    parse_version, 
    stabilize_for_fp16,
    save_optimized,
    load_optimized
)

# ------------------- Environment Configuration ------------------- 
# Suppress benign warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Converting a tensor to a Python boolean.*")
warnings.filterwarnings("ignore", module="torch.ao.quantization")
ort.set_default_logger_severity(3)  # ONNX Runtime: Only show errors

# Detect execution environment
IS_WINDOWS = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"
HAS_CUDA = torch.cuda.is_available()
HAS_DML = "DmlExecutionProvider" in ort.get_available_providers()

# ------------------- Constants & Configuration ------------------- 
MAX_RETRIES = 3
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
SUPPORTED_PRECISIONS = ['fp32', 'fp16', 'int8']
SUPPORTED_ONNX_PROVIDERS = ['CPUExecutionProvider']
if HAS_CUDA:
    SUPPORTED_ONNX_PROVIDERS.insert(0, 'CUDAExecutionProvider')
if HAS_DML:
    SUPPORTED_ONNX_PROVIDERS.insert(0, 'DmlExecutionProvider')

# ------------------- Setup Logging ------------------- 
def setup_logger(output_dir: Path) -> logging.Logger:
    """Configure logger with console and file handlers."""
    logger = logging.getLogger("AetherNetRelease")
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(ch)
    log_file = output_dir / "aether_release.log"
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(fh)
    return logger

# Global logger
logger = logging.getLogger("AetherNetRelease")

# ------------------- Helper Functions ------------------- 
def load_image(image_path: Path) -> torch.Tensor:
    """
    Load an image and convert to normalized tensor (CxHxW, [0,1]).
    
    Args:
        image_path: Path to image file
        
    Returns:
        Normalized torch.Tensor of shape (1, 3, H, W)
        
    Raises:
        IOError: If image loading fails
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    except Exception as e:
        raise IOError(f"Failed to load image {image_path}: {str(e)}")


def validate_pytorch_model(
    model: torch.nn.Module,
    validation_data: torch.Tensor,
    device: torch.device
) -> bool:
    """
    Validate model produces finite outputs with correct shape.
    
    Args:
        model: Model to validate
        validation_data: Input tensor for validation
        device: Device to run validation on
        
    Returns:
        True if validation succeeds, False otherwise
    """
    try:
        model.eval()
        model.to(device)
        
        with torch.no_grad():
            input_tensor = validation_data.to(device)
            # Match input dtype to model dtype
            if next(model.parameters()).dtype == torch.float16:
                input_tensor = input_tensor.half()

            output = model(input_tensor)
            
            if not torch.isfinite(output).all():
                raise RuntimeError("Model produced non-finite output values (NaN or Inf)")
            return True
    except Exception as e:
        logger.error(f"PyTorch validation FAILED: {str(e)}")
        return False


def validate_onnx_model(
    onnx_path: Path, validation_data: torch.Tensor, scale: int
) -> bool:
    """
    Validate ONNX model produces finite outputs with correct shape.
    
    Args:
        onnx_path: Path to ONNX model
        validation_data: Input tensor for validation
        scale: Super-resolution scale factor
        
    Returns:
        True if validation succeeds, False otherwise
    """
    try:
        session = ort.InferenceSession(str(onnx_path), providers=SUPPORTED_ONNX_PROVIDERS)
        input_name = session.get_inputs()[0].name
        input_type = session.get_inputs()[0].type
        
        input_data = validation_data.cpu().numpy()
        if 'float16' in input_type:
            input_data = input_data.astype(np.float16)
        
        output = session.run(None, {input_name: input_data})[0]
        
        _, c, h, w = validation_data.shape
        expected_shape = (1, c, h * scale, w * scale)
        if output.shape != expected_shape:
            logger.error(f"ONNX shape mismatch. Expected {expected_shape}, got {output.shape}")
            return False
        if not np.isfinite(output).all():
            logger.error("ONNX produced non-finite output values")
            return False
        return True
    except Exception as e:
        logger.error(f"ONNX validation FAILED: {str(e)}")
        return False


def attempt_operation(
    operation: Callable, 
    validation: Callable[[Any], bool], 
    success_msg: str, 
    error_msg: str, 
    max_retries: int = MAX_RETRIES
) -> Any:
    """
    Execute operation with retries and validation checks.
    
    Args:
        operation: Function to execute
        validation: Function to validate result
        success_msg: Message for successful operation
        error_msg: Message for failed operation
        max_retries: Maximum number of retries
        
    Returns:
        Result of operation if successful, None otherwise
    """
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Attempt {attempt}/{max_retries}")
            result = operation()
            if validation(result):
                logger.info(f"✅ {success_msg}")
                return result
            else:
                logger.warning(f"Validation failed on attempt {attempt}")
        except Exception as e:
            logger.error(f"Operation failed on attempt {attempt}: {str(e)}")
            logger.debug(traceback.format_exc())
            time.sleep(1)
    logger.error(f"❌ {error_msg}")
    return None


def memory_cleanup():
    """Clear GPU memory cache if available."""
    if HAS_CUDA:
        torch.cuda.empty_cache()


# ------------------- Model Conversion Functions ------------------- 
def create_fp32_model(
    base_model: AetherNet,
    validation_sample: torch.Tensor,
    device: torch.device
) -> Optional[AetherNet]:
    """
    Create clean fused FP32 model from base model.
    
    Args:
        base_model: Original model instance
        validation_sample: Sample for validation
        device: Device for validation
        
    Returns:
        Fused FP32 model if successful, None otherwise
    """
    def _operation():
        # Create fresh model instance to ensure clean state
        model = AetherNet(**base_model.get_config())
        # Load state dict ignoring quantization parameters
        model.load_state_dict(base_model.state_dict(), strict=False)
        model.fuse_model()
        model.eval()
        return model
        
    return attempt_operation(
        operation=_operation,
        validation=lambda m: validate_pytorch_model(m, validation_sample, device),
        success_msg="Created optimized FP32 model",
        error_msg="Failed to create FP32 model"
    )


def create_fp16_model(
    fp32_model: AetherNet,
    validation_sample: torch.Tensor,
    device: torch.device
) -> Optional[AetherNet]:
    """
    Convert FP32 model to stable FP16 format.
    
    Args:
        fp32_model: Fused FP32 model
        validation_sample: Sample for validation
        device: Device for validation
        
    Returns:
        FP16 model if successful, None otherwise
    """
    def _operation():
        model = deepcopy(fp32_model)
        # Apply FP16 stabilization before conversion
        stabilize_for_fp16(model)
        model = model.half()
        model.eval()
        return model
        
    return attempt_operation(
        operation=_operation,
        validation=lambda m: validate_pytorch_model(m, validation_sample, device),
        success_msg="Created stable FP16 model",
        error_msg="Failed to create FP16 model"
    )


def create_int8_model(
    base_qat_model: AetherNet,
    validation_sample: torch.Tensor,
    device: torch.device
) -> Optional[AetherNet]:
    """
    Convert QAT model to final INT8 quantized model.
    
    Args:
        base_qat_model: QAT-prepared model
        validation_sample: Sample for validation
        device: Device for validation
        
    Returns:
        INT8 model if successful, None otherwise
    """
    def _operation():
        # Convert on CPU for stability
        model_to_convert = deepcopy(base_qat_model).cpu().eval()
        int8_model = model_to_convert.convert_to_quantized()
            
        if not int8_model.verify_quantization():
            raise RuntimeError("Quantization verification failed")
        return int8_model
            
    # Validate on CPU where conversion happens
    cpu_device = torch.device('cpu')
    return attempt_operation(
        operation=_operation,
        validation=lambda m: validate_pytorch_model(m, validation_sample, cpu_device),
        success_msg="Created INT8 model",
        error_msg="Failed to create INT8 model"
    )


def export_onnx_wrapper(
    model: AetherNet, 
    scale: int, 
    precision: str, 
    output_path: Path, 
    validation_sample: torch.Tensor
) -> Optional[Path]:
    def _operation():
            
        export_onnx(deepcopy(model), scale, precision, str(output_path))
        return output_path

    return attempt_operation(
        operation=_operation,
        validation=lambda p: validate_onnx_model(p, validation_sample, scale),
        success_msg=f"Exported and validated {precision.upper()} ONNX model",
        error_msg=f"Failed to export {precision.upper()} ONNX model"
    )


def load_checkpoint(model_path: Path) -> tuple:
    """
    Load model checkpoint and extract architecture configuration.
    
    Args:
        model_path: Path to model checkpoint
        
    Returns:
        (state_dict, arch_config, scale, is_qat_checkpoint)
        
    Raises:
        ValueError: If checkpoint is invalid
    """
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        # Extract state_dict from various possible locations
        state_dict = checkpoint.get('params_ema', 
                                  checkpoint.get('params', 
                                  checkpoint.get('state_dict', {})))
        
        if not state_dict:
            raise ValueError("Could not find a valid state_dict in the checkpoint.")

        arch_config = checkpoint.get('arch_config', None)
        
        # Detect QAT checkpoint by specific keys
        is_qat_checkpoint = any(
            'fake_quant' in k or 'activation_post_process' in k 
            for k in state_dict.keys()
        )

        if not arch_config:
            # Fallback for neosr-style checkpoints
            train_args = checkpoint.get('args', {})
            if train_args:
                arch_config = {
                    'in_chans': 3, 
                    'embed_dim': train_args.get('embed_dim', 64),
                    'depths': train_args.get('depths', (3, 3, 3)),
                    'scale': train_args.get('scale', 2), 
                    'mlp_ratio': 1.5,
                }
                logger.info("Reconstructed arch_config from training args.")
            else:
                raise ValueError("Checkpoint is missing 'arch_config' and 'args'")

        if isinstance(arch_config.get('depths'), list):
            arch_config['depths'] = tuple(arch_config['depths'])
        
        scale = arch_config['scale']
        return state_dict, arch_config, scale, is_qat_checkpoint
    except Exception as e:
        raise ValueError(f"Checkpoint loading failed: {str(e)}")


# ------------------- Main Execution ------------------- 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert AetherNet models to deployment formats",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model (.pth)')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for converted models')
    parser.add_argument('--validation-dir', type=str, required=True, help='Directory with validation images')
    parser.add_argument('--skip-int8', action='store_true', help='Skip INT8 quantization')
    parser.add_argument('--skip-fp16', action='store_true', help='Skip FP16 conversion')
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    device = torch.device('cuda' if HAS_CUDA else 'cpu')
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(output_dir)
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info("=== AetherNet Model Release Script ===")
    logger.info(f"PyTorch: {torch.__version__}, Device: {device}, Model: {model_path.name}")

    # --- Phase 1: Load and Validate Checkpoint ---
    logger.info("\n[1/4] Loading and analyzing checkpoint")
    try:
        state_dict, arch_config, scale, is_qat_checkpoint = load_checkpoint(model_path)
        validation_images = list(Path(args.validation_dir).glob('*.[jp][pn]g'))
        if not validation_images: 
            raise FileNotFoundError("No validation images found in specified directory")
        validation_sample = load_image(validation_images[0])
        logger.info(f"Scale: {scale}x, QAT: {is_qat_checkpoint}, Validation sample: {validation_images[0].name}")
    except Exception as e:
        logger.error(f"Checkpoint loading failed: {e}")
        sys.exit(1)

    # --- Phase 2: Initialize Base Model ---
    logger.info("\n[2/4] Initializing base model")
    try:
        base_model = AetherNet(**arch_config)
        arch_name = base_model._get_architecture_name()
        logger.info(f"Detected Architecture: {arch_name}")

        if is_qat_checkpoint:
            logger.info("Preparing model for QAT state dict...")
            base_model.prepare_qat()
        elif any('fused_conv' in k for k in state_dict.keys()):
            logger.info("Fusing model to match non-QAT fused checkpoint.")
            base_model.fuse_model()

        base_model.load_state_dict(state_dict, strict=False)
        base_model.eval()
        logger.info("Base model loaded successfully.")

        logger.info("Validating base model...")
        if not validate_pytorch_model(base_model, validation_sample, device):
            raise RuntimeError("Base model failed validation.")
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        sys.exit(1)

    # --- Phase 3: Convert Models ---
    logger.info("\n[3/4] Converting models to different precisions")
    model_stem = f"{model_path.stem}_{arch_name}"
    
    # FP32 conversion
    logger.info("\n>> Creating FP32 model")
    fp32_model = create_fp32_model(base_model, validation_sample, device)
    if fp32_model:
        fp32_path = output_dir / f"{model_stem}_fp32.pth"
        save_optimized(fp32_model, str(fp32_path), 'fp32')
        logger.info(f"Saved FP32 model to: {fp32_path}")
        export_onnx_wrapper(fp32_model, scale, 'fp32', output_dir / f"{model_stem}_fp32.onnx", validation_sample)
    
    # FP16 conversion
    if not args.skip_fp16 and fp32_model:
        logger.info("\n>> Creating FP16 model")
        fp16_model = create_fp16_model(fp32_model, validation_sample, device)
        if fp16_model:
            fp16_path = output_dir / f"{model_stem}_fp16.pth"
            save_optimized(fp16_model, str(fp16_path), 'fp16')
            logger.info(f"Saved FP16 model to: {fp16_path}")
            export_onnx_wrapper(fp16_model, scale, 'fp16', output_dir / f"{model_stem}_fp16.onnx", validation_sample)
    
    # INT8 conversion
    if not args.skip_int8 and is_qat_checkpoint:
        logger.info("\n>> Creating INT8 model")
        int8_model = create_int8_model(base_model, validation_sample, device)
        if int8_model:
            int8_path = output_dir / f"{model_stem}_int8.pth"
            save_optimized(int8_model, str(int8_path), 'int8')
            logger.info(f"Saved INT8 model to: {int8_path}")
            export_onnx_wrapper(int8_model, scale, 'int8', output_dir / f"{model_stem}_int8.onnx", validation_sample)
    elif not is_qat_checkpoint and not args.skip_int8:
        logger.warning("\nSkipping INT8 conversion: The provided checkpoint is not from QAT.")

    # --- Phase 4: Final Report ---
    logger.info("\n[4/4] Conversion complete.")
    logger.info(f"Results saved to: {output_dir}")
    memory_cleanup()
    sys.exit(0)