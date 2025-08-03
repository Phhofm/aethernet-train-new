# -*- coding: utf-8 -*-
"""
https://github.com/Phhofm/aethernet

AetherNet: Ultra-Fast Super-Resolution Network with QAT Support

Core architecture definition featuring:
- Structural reparameterization for inference efficiency
- Quantization-aware training (INT8, FP16) support
- Deployment-friendly normalization
- Multi-scale feature fusion
- Adaptive upsampling

Designed for easy integration into super-resolution frameworks like Spandrel, 
neosr, and traiNNer-redux. Minimal dependencies - only requires PyTorch.
"""

import math
import warnings
from copy import deepcopy
from typing import Tuple, List, Dict, Any, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_
import torch.ao.quantization as tq


class DropPath(nn.Module):
    """
    Stochastic Depth implementation compatible with ONNX export.
    
    During training, randomly drops entire sample paths with given probability.
    During inference, acts as identity function.
    
    Args:
        drop_prob: Probability of dropping a path (0.0 = no drop)
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        if not 0.0 <= drop_prob <= 1.0:
            raise ValueError("drop_prob must be between 0.0 and 1.0")
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
            
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize to 0 or 1
        return x.div(keep_prob) * random_tensor


class ReparamLargeKernelConv(nn.Module):
    """
    Efficient large kernel convolution using structural reparameterization.
    
    Combines large and small kernel convolutions during training, fuses them
    into a single convolution for inference efficiency.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Main kernel size (must be odd)
        stride: Convolution stride
        groups: Number of groups
        small_kernel: Parallel small kernel size (must be odd)
        fused_init: Initialize in fused mode
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, groups: int, small_kernel: int, fused_init: bool = False):
        super().__init__()
        if kernel_size % 2 == 0 or small_kernel % 2 == 0:
            raise ValueError("Kernel sizes must be odd for symmetrical padding")
        if groups <= 0:
            raise ValueError("Number of groups must be positive")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = kernel_size // 2
        self.small_kernel = small_kernel
        self.fused = fused_init
        self.is_quantized = False

        if self.fused:
            # Directly initialize fused convolution
            self.fused_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride,
                padding=self.padding, groups=groups, bias=True
            )
        else:
            # Training branch: large kernel convolution without bias
            self.lk_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride,
                self.padding, groups=groups, bias=False
            )
            # Training branch: small kernel convolution without bias
            self.sk_conv = nn.Conv2d(
                in_channels, out_channels, small_kernel, stride,
                small_kernel // 2, groups=groups, bias=False
            )
            # Bias parameters for each convolution
            self.lk_bias = nn.Parameter(torch.zeros(out_channels))
            self.sk_bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused:
            return self.fused_conv(x)

        # Training forward: apply both convolutions and add their outputs
        lk_out = self.lk_conv(x)
        sk_out = self.sk_conv(x)
        return (lk_out + self.lk_bias.view(1, -1, 1, 1) +
                sk_out + self.sk_bias.view(1, -1, 1, 1))

    def fuse(self):
        """Fuse the two convolutions and their biases into a single convolution for inference."""
        if self.fused:
            return

        # Pad the small kernel to the size of the large kernel
        pad = (self.kernel_size - self.small_kernel) // 2
        sk_kernel_padded = F.pad(self.sk_conv.weight, [pad] * 4)
        fused_kernel = self.lk_conv.weight + sk_kernel_padded
        fused_bias = self.lk_bias + self.sk_bias

        # Create the fused convolution
        self.fused_conv = nn.Conv2d(
            self.in_channels, self.out_channels, self.kernel_size,
            self.stride, padding=self.padding, groups=self.groups, bias=True
        )
        self.fused_conv.weight.data = fused_kernel
        self.fused_conv.bias.data = fused_bias

        # Preserve quantization configuration if present
        if self.is_quantized and hasattr(self.lk_conv, 'qconfig'):
            self.fused_conv.qconfig = self.lk_conv.qconfig

        # Cleanup the training-specific parameters
        del self.lk_conv, self.sk_conv, self.lk_bias, self.sk_bias
        self.fused = True


class GatedConvFFN(nn.Module):
    """
    Gated Feed-Forward Network for enhanced feature transformation.
    
    Uses SiLU activation with temperature scaling and quantization support.
    
    Args:
        in_channels: Input channels
        mlp_ratio: Hidden dimension multiplier (default: 1.5)
        drop: Dropout probability (default: 0.0)
    """
    def __init__(self, in_channels: int, mlp_ratio: float = 1.5, drop: float = 0.):
        super().__init__()
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if not 0.0 <= drop <= 1.0:
            raise ValueError("drop probability must be between 0 and 1")
            
        hidden_channels = int(in_channels * mlp_ratio)

        self.conv_gate = nn.Conv2d(in_channels, hidden_channels, 1)
        self.conv_main = nn.Conv2d(in_channels, hidden_channels, 1)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(drop)
        self.conv_out = nn.Conv2d(hidden_channels, in_channels, 1)
        self.drop2 = nn.Dropout(drop)
        self.temperature = nn.Parameter(torch.ones(1))
        self.is_quantized = False
        self.quant_mul = torch.nn.quantized.FloatFunctional()

        # Quantization stubs for activation and temperature scaling
        self.act_dequant = tq.DeQuantStub()
        self.act_quant = tq.QuantStub()
        self.temp_dequant = tq.DeQuantStub()
        self.temp_quant = tq.QuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gate branch with temperature scaling
        gate_unscaled = self.conv_gate(x)
        if self.is_quantized:
            # In quantized mode, do temperature scaling in float and requantize
            gate_float = self.temp_dequant(gate_unscaled)
            gate_scaled = gate_float * self.temperature
            gate = self.temp_quant(gate_scaled)
        else:
            gate = gate_unscaled * self.temperature

        # Main branch
        main = self.conv_main(x)

        # Apply activation in float domain for quantization compatibility
        gate = self.act_dequant(gate)
        activated = self.act(gate)
        activated = self.act_quant(activated)

        # Element-wise multiplication
        if x.dtype == torch.float16:
            # Handle FP16 with FP32 precision to avoid overflow
            x = activated.float() * main.float()
            x = x.half()
        elif self.is_quantized:
            # Use quantized multiplication
            x = self.quant_mul.mul(activated, main)
        else:
            x = activated * main

        x = self.drop1(x)
        x = self.conv_out(x)
        return x


class DynamicChannelScaling(nn.Module):
    """
    Efficient Channel Attention (Squeeze-and-Excitation) mechanism.
    
    Args:
        dim: Input dimension (number of channels)
        reduction: Channel reduction ratio (default: 8)
    """
    def __init__(self, dim: int, reduction: int = 8):
        super().__init__()
        if dim <= reduction:
            raise ValueError(f"Reduction ratio {reduction} is too large for {dim} channels")

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, bias=False),
            nn.Sigmoid()
        )
        self.quant_mul = torch.nn.quantized.FloatFunctional()
        self.is_quantized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(self.avg_pool(x))
        if self.is_quantized:
            return self.quant_mul.mul(x, scale)
        return x * scale


class SpatialAttention(nn.Module):
    """
    Lightweight spatial attention module.
    
    Args:
        kernel_size: Convolution kernel size (default: 7)
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd for symmetrical padding")

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.quant_mul = torch.nn.quantized.FloatFunctional()
        self.is_quantized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Aggregate spatial information using max and average pooling
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        concat = torch.cat([max_pool, avg_pool], dim=1)
        attention_map = self.sigmoid(self.conv(concat))
        if self.is_quantized:
            return self.quant_mul.mul(x, attention_map)
        return x * attention_map


class DeploymentNorm(nn.Module):
    """
    Deployment-friendly normalization layer with fusion support.
    
    Maintains running statistics during training, converts to
    simple affine transform for inference.
    
    Args:
        channels: Number of channels
        eps: Numerical stability epsilon (default: 1e-4)
    """
    def __init__(self, channels: int, eps: float = 1e-4):
        super().__init__()
        if channels <= 0:
            raise ValueError("Number of channels must be positive")
        if eps <= 0:
            raise ValueError("Epsilon must be positive")
            
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps
        self.fused = False
        self.register_buffer('running_mean', torch.zeros(1, channels, 1, 1))
        self.register_buffer('running_var', torch.ones(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused:
            return x * self.weight + self.bias

        if self.training:
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
            with torch.no_grad():
                # Update running statistics with momentum
                self.running_mean.mul_(0.9).add_(mean, alpha=0.1)
                self.running_var.mul_(0.9).add_(var, alpha=0.1)
        else:
            mean = self.running_mean
            var = self.running_var

        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias

    def fuse(self):
        """Fuse the normalization into a single affine transform for inference."""
        if self.fused:
            return
        
        # Compute fused scale and shift
        scale = self.weight / torch.sqrt(self.running_var + self.eps)
        shift = self.bias - self.running_mean * scale
        
        self.weight.data = scale
        self.bias.data = shift
        
        # Cleanup buffers
        del self.running_mean
        del self.running_var
        self.fused = True


class LayerNorm2d(nn.LayerNorm):
    """2D-adapted LayerNorm for 4D tensors (N, C, H, W)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        return x.permute(0, 3, 1, 2)


class AetherBlock(nn.Module):
    """
    Core building block of AetherNet architecture.
    
    Combines large kernel convolution, gated FFN, and attention mechanisms
    with residual connection and quantization support.
    
    Args:
        dim: Feature dimension
        mlp_ratio: FFN expansion ratio (default: 1.5)
        drop: Dropout probability (default: 0.0)
        drop_path: Stochastic path probability (default: 0.0)
        lk_kernel: Large kernel size (default: 13)
        sk_kernel: Small kernel size (default: 5)
        fused_init: Initialize in fused mode (default: False)
        quantize_residual: Quantize residual connection (default: True)
        use_channel_attn: Enable channel attention (default: True)
        use_spatial_attn: Enable spatial attention (default: False)
        norm_layer: Normalization layer type (default: DeploymentNorm)
        res_scale: Residual scaling factor (default: 0.1)
    """
    def __init__(self, dim: int, mlp_ratio: float = 1.5, drop: float = 0.,
                 drop_path: float = 0., lk_kernel: int = 13, sk_kernel: int = 5,
                 fused_init: bool = False, quantize_residual: bool = True,
                 use_channel_attn: bool = True, use_spatial_attn: bool = False,
                 norm_layer: nn.Module = DeploymentNorm, res_scale: float = 0.1):
        super().__init__()
        if dim <= 0:
            raise ValueError("Feature dimension must be positive")
        if not 0.0 <= drop_path <= 1.0:
            raise ValueError("drop_path probability must be between 0 and 1")
            
        self.res_scale = res_scale
        self.conv = ReparamLargeKernelConv(
            in_channels=dim, out_channels=dim, kernel_size=lk_kernel,
            stride=1, groups=dim, small_kernel=sk_kernel, fused_init=fused_init
        )
        self.norm = norm_layer(dim)
        self.ffn = GatedConvFFN(in_channels=dim, mlp_ratio=mlp_ratio, drop=drop)
        self.channel_attn = DynamicChannelScaling(dim) if use_channel_attn else nn.Identity()
        self.spatial_attn = SpatialAttention() if use_spatial_attn else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.is_quantized = False
        self.quant_add = torch.nn.quantized.FloatFunctional()
        self.norm_dequant = tq.DeQuantStub()
        self.norm_quant = tq.QuantStub()
        self.res_dequant = tq.DeQuantStub()
        self.res_quant = tq.QuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv(x)

        # Normalization in float domain for quantization compatibility
        x = self.norm_dequant(x)
        x = self.norm(x)
        x = self.norm_quant(x)
        
        x = self.ffn(x)
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        
        residual_unscaled = self.drop_path(x)
        
        # Residual scaling in float domain
        if self.is_quantized:
            res_float = self.res_dequant(residual_unscaled)
            res_scaled = res_float * self.res_scale
            residual = self.res_quant(res_scaled)
        else:
            residual = residual_unscaled * self.res_scale

        # Add residual connection
        if self.is_quantized:
            return self.quant_add.add(shortcut, residual)
        else:
            return shortcut + residual


class QuantFusion(nn.Module):
    """
    Multi-scale feature fusion with quantization support and error compensation.
    
    Args:
        in_channels: Total input channels from all features
        out_channels: Output channels after fusion
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("Channel counts must be positive")
            
        self.fusion_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.error_comp = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.quant_mul = torch.nn.quantized.FloatFunctional()
        self.is_quantized = False

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        if not features:
            raise ValueError("QuantFusion requires at least one feature")

        # Align all features to the size of the first feature
        target_size = features[0].shape[-2:]
        aligned_features = []
        for feat in features:
            if feat.shape[-2:] != target_size:
                aligned = F.interpolate(feat, size=target_size, mode='nearest')
                aligned_features.append(aligned)
            else:
                aligned_features.append(feat)

        x = torch.cat(aligned_features, dim=1)
        fused = self.fusion_conv(x)
        
        # Skip error compensation in quantized mode
        if self.is_quantized:
            return fused
        return fused + self.error_comp


class AdaptiveUpsample(nn.Module):
    """
    Resolution-aware upsampling module supporting powers of 2 and scale 3.
    
    Args:
        scale: Upscaling factor
        in_channels: Input channels
    """
    def __init__(self, scale: int, in_channels: int):
        super().__init__()
        if scale < 1:
            raise ValueError("Scale must be at least 1")
        if in_channels <= 0:
            raise ValueError("Input channels must be positive")
            
        self.scale = scale
        self.in_channels = in_channels
        self.blocks = nn.ModuleList()
        # Ensure out_channels is multiple of 4 for PixelShuffle
        self.out_channels = max(32, (in_channels // max(1, scale // 2)) & -2)

        # Power of 2 scaling
        if (scale & (scale - 1)) == 0 and scale != 1:
            num_ups = int(math.log2(scale))
            current_channels = in_channels
            for i in range(num_ups):
                next_channels = self.out_channels if (i == num_ups - 1) else current_channels // 2
                self.blocks.append(nn.Conv2d(current_channels, 4 * next_channels, 3, 1, 1))
                self.blocks.append(nn.PixelShuffle(2))
                current_channels = next_channels
        # Scale 3
        elif scale == 3:
            self.blocks.append(nn.Conv2d(in_channels, 9 * self.out_channels, 3, 1, 1))
            self.blocks.append(nn.PixelShuffle(3))
        # Scale 1 (no upsampling)
        elif scale == 1:
            self.blocks.append(nn.Conv2d(in_channels, self.out_channels, 3, 1, 1))
        else:
            raise ValueError(f"Unsupported scale: {scale}. Only 1, 3 and powers of 2")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class AetherNet(nn.Module):
    """
    Production-Ready Super-Resolution Network with QAT support.
    
    Args:
        in_chans: Input channels (default: 3 for RGB)
        embed_dim: Base channel dimension
        depths: Number of blocks in each stage
        mlp_ratio: FFN expansion ratio (default: 1.5)
        drop: Dropout probability (default: 0.0)
        drop_path_rate: Stochastic path probability (default: 0.1)
        lk_kernel: Large kernel size (default: 13)
        sk_kernel: Small kernel size (default: 5)
        scale: Super-resolution scale factor
        img_range: Input image range (default: 1.0 for [0,1])
        fused_init: Initialize in fused mode (default: False)
        quantize_residual: Quantize residual connections (default: True)
        use_channel_attn: Enable channel attention (default: True)
        use_spatial_attn: Enable spatial attention (default: False)
        norm_type: Normalization type ('deployment' or 'layernorm')
        res_scale: Residual scaling factor (default: 0.1)
        **kwargs: Additional keyword arguments
    """
    MODEL_VERSION = "3.1.1"
    
    def _init_weights(self, m: nn.Module):
        """Initialize weights using truncated normal distribution."""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (DeploymentNorm, nn.LayerNorm, nn.GroupNorm)):
            if m.bias is not None: 
                nn.init.constant_(m.bias, 0)
            if m.weight is not None: 
                nn.init.constant_(m.weight, 1.0)

    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (4, 4, 4, 4),
        mlp_ratio: float = 1.5,
        drop: float = 0.0,
        drop_path_rate: float = 0.1,
        lk_kernel: int = 13,
        sk_kernel: int = 5,
        scale: int = 4,
        img_range: float = 1.0,
        fused_init: bool = False,
        quantize_residual: bool = True,
        use_channel_attn: bool = True,
        use_spatial_attn: bool = False,
        norm_type: str = 'deployment',
        res_scale: float = 0.1,
        arch_option: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        # Validate inputs
        if in_chans not in (1, 3):
            warnings.warn(f"Unusual input channels: {in_chans}. Expected 1 or 3.")
        if embed_dim < 16:
            raise ValueError("Embed dimension must be at least 16")
        if not depths:
            raise ValueError("Depths must be a non-empty tuple of integers")
        if scale < 1:
            raise ValueError("Scale must be at least 1")
        if img_range <= 0:
            raise ValueError("Image range must be positive")
        if not 0.0 <= drop_path_rate <= 1.0:
            raise ValueError("drop_path_rate must be between 0 and 1")
        
        self.arch_option = arch_option

        # Capture ALL constructor parameters
        self.arch_config = {
            'in_chans': in_chans, 'embed_dim': embed_dim, 'depths': depths,
            'mlp_ratio': mlp_ratio, 'drop': drop, 'drop_path_rate': drop_path_rate,
            'lk_kernel': lk_kernel, 'sk_kernel': sk_kernel, 'scale': scale,
            'img_range': img_range, 'fused_init': fused_init,
            'quantize_residual': quantize_residual, 'use_channel_attn': use_channel_attn,
            'use_spatial_attn': use_spatial_attn, 'norm_type': norm_type,
            'arch_option': arch_option,
            'res_scale': res_scale, **kwargs
        }
        
        # Convert tuple parameters to lists for JSON serialization
        if isinstance(self.arch_config['depths'], tuple):
            self.arch_config['depths'] = list(self.arch_config['depths'])
        
        self.img_range = img_range
        self.register_buffer('scale_tensor', torch.tensor(scale, dtype=torch.int64))
        self.fused_init = fused_init
        self.embed_dim = embed_dim
        self.quantize_residual = quantize_residual
        self.num_stages = len(depths)
        self.is_quantized = False
        
        # Input normalization
        self.register_buffer('mean', torch.full((1, in_chans, 1, 1), 0.5))

        # Initial convolution
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # Normalization layer selection
        if norm_type.lower() == 'deployment':
            norm_layer = DeploymentNorm
        elif norm_type.lower() == 'layernorm':
            norm_layer = LayerNorm2d
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

        # Stage construction
        self.stages = nn.ModuleList()
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        self.fusion_convs = nn.ModuleList()

        # Channel distribution for multi-scale fusion
        base_ch = embed_dim // self.num_stages
        remainder = embed_dim % self.num_stages
        fusion_out_channels = [base_ch + 1 if i < remainder else base_ch for i in range(self.num_stages)]
        
        if sum(fusion_out_channels) != embed_dim:
            raise RuntimeError("Channel distribution error in feature fusion")

        # Build stages and fusion convolutions
        block_idx = 0
        for i, depth in enumerate(depths):
            stage_blocks = []
            for j in range(depth):
                stage_blocks.append(AetherBlock(
                    dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop,
                    drop_path=dpr[block_idx + j], lk_kernel=lk_kernel, sk_kernel=sk_kernel,
                    fused_init=fused_init, quantize_residual=quantize_residual,
                    use_channel_attn=use_channel_attn, use_spatial_attn=use_spatial_attn,
                    norm_layer=norm_layer, res_scale=res_scale))
            self.stages.append(nn.Sequential(*stage_blocks))
            block_idx += depth
            self.fusion_convs.append(nn.Conv2d(embed_dim, fusion_out_channels[i], 1))

        # Feature fusion and reconstruction
        self.quant_fusion_layer = QuantFusion(embed_dim, embed_dim)
        self.norm = norm_layer(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # Upsampling path
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1), 
            nn.LeakyReLU(inplace=True))
        self.upsample = AdaptiveUpsample(scale, embed_dim)
        self.conv_last = nn.Conv2d(self.upsample.out_channels, in_chans, 3, 1, 1)

        # Initialize weights if not fused
        if not self.fused_init:
            self.apply(self._init_weights)

        # Quantization stubs
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()
        self.body_norm_dequant = tq.DeQuantStub()
        self.body_norm_quant = tq.QuantStub()
        self.quant_add = torch.nn.quantized.FloatFunctional()
        self.upsample_dequant = tq.DeQuantStub()
        self.upsample_quant = tq.QuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input normalization
        x_in = x / self.img_range if self.img_range != 1.0 else x
        x_in = x_in - self.mean
        
        # Quantization entry point
        x = self.quant(x_in)
        x_first = self.conv_first(x)

        # Feature extraction stages
        features = []
        out = x_first
        for stage in self.stages:
            out = stage(out)
            features.append(self.fusion_convs[len(features)](out))

        # Multi-scale feature fusion
        fused_features = self.quant_fusion_layer(features)
        
        # Normalization in float domain
        body_out = self.body_norm_dequant(fused_features)
        body_out = self.norm(body_out)
        body_out = self.body_norm_quant(body_out)
        
        # Residual connections
        if self.is_quantized:
            body_out = self.quant_add.add(body_out, x_first)
            body_out = self.quant_add.add(self.conv_after_body(body_out), body_out)
        else:
            body_out = body_out + x_first
            body_out = self.conv_after_body(body_out) + body_out
        
        # Reconstruction
        recon = self.conv_before_upsample(body_out)
        
        # Upsampling in float domain for quantization compatibility
        if self.is_quantized:
            recon = self.upsample_dequant(recon)
            recon = self.upsample(recon)
            recon = self.upsample_quant(recon)
        else:
            recon = self.upsample(recon)
            
        recon = self.conv_last(recon)
        output = self.dequant(recon)

        # Output denormalization
        output = output + self.mean
        output = output * self.img_range if self.img_range != 1.0 else output
        return output

    def fuse_model(self):
        """Fuse reparameterizable components for inference."""
        if self.fused_init:
            return
        for module in self.modules():
            if hasattr(module, 'fuse') and callable(module.fuse):
                module.fuse()
        self.fused_init = True

    def prepare_qat(self, per_channel: bool = False):
        """
        Prepare model for Quantization-Aware Training.
        
        Args:
            per_channel: Use per-channel quantization for weights (default: False)
        """
        # Check per-channel compatibility
        if per_channel and not hasattr(tq, 'MovingAveragePerChannelMinMaxObserver'):
            warnings.warn("Per-channel quantization requires PyTorch 1.10+. Disabling.")
            per_channel = False

        # Configure quantization observers
        activation_observer = tq.MovingAverageMinMaxObserver.with_args(
            qscheme=torch.per_tensor_affine,
            dtype=torch.quint8,
            reduce_range=False
        )
        
        weight_observer = (
            tq.MovingAveragePerChannelMinMaxObserver.with_args(
                qscheme=torch.per_channel_symmetric, dtype=torch.qint8
            ) if per_channel else
            tq.MovingAverageMinMaxObserver.with_args(
                qscheme=torch.per_tensor_symmetric, dtype=torch.qint8
            )
        )
        
        qconfig = tq.QConfig(activation=activation_observer, weight=weight_observer)
        self.qconfig = qconfig
        
        self.fuse_model()
        
        # Exclude upsampling module from quantization (pixel shuffle isn't quantizable)
        self.upsample.qconfig = None
        warnings.warn("Excluded upsampling module from quantization (pixel shuffle operations aren't quantizable)")

        tq.prepare_qat(self, inplace=True)
        self._set_quantization_flags(True)
        
    def _set_quantization_flags(self, status: bool):
        """Set quantization status flag on all relevant modules."""
        for module in self.modules():
            if hasattr(module, 'is_quantized'):
                module.is_quantized = status

    def convert_to_quantized(self) -> nn.Module:
        """Convert QAT model to fully quantized INT8 model."""
        if not self.is_quantized:
            raise RuntimeError("Model must be prepared with prepare_qat() first")
        
        self.eval()
        
        # Preserve quantization parameters
        quant_params = {}
        try:
            # Extract from quantization stubs if available
            if hasattr(self.quant, 'scale'):
                quant_params['input_scale'] = self.quant.scale
                quant_params['input_zero_point'] = self.quant.zero_point
            
            if hasattr(self.dequant, 'scale'):
                quant_params['output_scale'] = self.dequant.scale
                quant_params['output_zero_point'] = self.dequant.zero_point
            
            # Convert to simple Python types
            for key in quant_params:
                if torch.is_tensor(quant_params[key]):
                    quant_params[key] = quant_params[key].item()
        except Exception as e:
            warnings.warn(f"Couldn't extract quantization params: {str(e)}")
            quant_params = {
                'input_scale': 1/255.0,
                'input_zero_point': 0,
                'output_scale': 1/255.0,
                'output_zero_point': 0
            }
        
        # Perform conversion
        quantized_model = tq.convert(self, inplace=False)
        quantized_model._set_quantization_flags(True)
        
        # Attach quantization parameters to the converted model
        quantized_model.quant_params = quant_params
        
        return quantized_model
    
    def verify_quantization(self) -> bool:
        """Check quantization status of all layers. Returns True if fully quantized."""
        non_quantized = []
        for name, module in self.named_modules():
            # Skip intentionally non-quantized modules
            if name == "upsample" or name.startswith("upsample."):
                continue
                
            # Check for standard layers that should be quantized
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # In a converted model, it should be a child of QuantizedModule
                if 'quantized' not in type(module).__name__.lower():
                    # Check if it has a fake_quant attribute (if it's a QAT model)
                    if not hasattr(module, 'weight_fake_quant'):
                        non_quantized.append(name)
        
        if non_quantized:
            warnings.warn(f"Non-quantized layers found: {non_quantized}")
        return len(non_quantized) == 0
    
    def get_config(self) -> Dict[str, Any]:
        """Return the complete architecture configuration."""
        return deepcopy(self.arch_config)

    def _get_architecture_name(self) -> str:
        """Get human-readable architecture name."""
        # Prioritize the explicitly set architecture name
        if self.arch_name:
            return self.arch_name

        # Fallback for older models or custom configurations without a name
        warnings.warn("Architecture name not set explicitly, inferring from embed_dim.")
        if self.embed_dim <= 64: return "aether_tiny"
        if self.embed_dim <= 96: return "aether_small"
        if self.embed_dim <= 128: return "aether_medium"
        if self.embed_dim <= 180: return "aether_large"
        return "custom"
    
############# network options for neosr, extracted from utils file ########################
        
def aether_mobile(scale: int, **kwargs) -> AetherNet:
    """
    AetherNet 'Mobile' Variant: Optimized for maximum performance and efficiency.

    This preset is designed for real-time applications on resource-constrained
    devices such as mobile phones, web browsers, or edge hardware. It achieves
    its small footprint and high speed by using a combination of:
    - A narrow channel dimension (64) to reduce memory usage.
    - A very shallow block depth to minimize the number of computations.
    - Smaller reparameterized kernels (7x7 and 3x3) for faster convolutions.
    - Disabled attention mechanisms to remove all computational overhead.

    This configuration prioritizes speed and efficiency over restoration quality,
    making it the ideal choice when performance is the most critical requirement.

    Args:
        scale: The desired super-resolution scale factor (e.g., 2, 3, 4).
        **kwargs: Additional keyword arguments that will be passed to the AetherNet
                  constructor, allowing for customization of other parameters like
                  `norm_type` or `img_range`.

    Returns:
        An AetherNet model instance configured with the 'Mobile' preset.
    """
    return AetherNet(
        embed_dim=64,
        depths=(2, 2, 2, 2),
        lk_kernel=7,
        sk_kernel=3,
        scale=scale,
        use_channel_attn=False,
        use_spatial_attn=False,
        res_scale=0.2,
        arch_option='aether_mobile',
        **kwargs  # Pass through any other user-specified arguments
    )


def aether_tiny(scale: int, **kwargs) -> AetherNet:
    """
    Minimal version for real-time use (64 channels).
    
    Args:
        scale: Super-resolution scale factor
        **kwargs: Additional keyword arguments for AetherNet
    """
    return AetherNet(embed_dim=64, depths=(3, 3, 3), scale=scale,
                  use_spatial_attn=False, res_scale=0.2, arch_option='aether_tiny', **kwargs)


def aether_small(scale: int, **kwargs) -> AetherNet:
    """
    Small version (96 channels).
    
    Args:
        scale: Super-resolution scale factor
        **kwargs: Additional keyword arguments for AetherNet
    """
    return AetherNet(embed_dim=96, depths=(4, 4, 4, 4), scale=scale,
                  use_spatial_attn=False, res_scale=0.1, arch_option='aether_small', **kwargs)


def aether_medium(scale: int, **kwargs) -> AetherNet:
    """
    Balanced version (128 channels).
    
    Args:
        scale: Super-resolution scale factor
        **kwargs: Additional keyword arguments for AetherNet
    """
    return AetherNet(embed_dim=128, depths=(6, 6, 6, 6), scale=scale,
                  use_channel_attn=True, res_scale=0.1, arch_option='aether_medium', **kwargs)


def aether_large(scale: int, **kwargs) -> AetherNet:
    """
    High-quality version (180 channels).
    
    Args:
        scale: Super-resolution scale factor
        **kwargs: Additional keyword arguments for AetherNet
    """
    return AetherNet(embed_dim=180, depths=(8, 8, 8, 8, 8), scale=scale,
                  use_channel_attn=True, use_spatial_attn=True, res_scale=0.1, arch_option='aether_large', **kwargs)

def aether_pro(scale: int, **kwargs) -> AetherNet:
    """
    High-quality 'Pro' version with a focus on efficiency (192 channels).

    This preset is designed to deliver a quality improvement over the 'large' variant
    with a minimal impact on performance and resource usage. It is the recommended
    choice for high-end, practical super-resolution. It uses:
    - A modest increase in channel width (192) for more feature capacity.
    - A larger reparameterized kernel (15x15) to expand the receptive field
      at a very low computational cost.
    - A slightly larger FFN expansion ratio (1.75) for better feature transformation.
    - The same depth as 'aether_large' to maintain a good speed profile.

    Args:
        scale: Super-resolution scale factor
        **kwargs: Additional keyword arguments for AetherNet
    """
    return AetherNet(
        embed_dim=192,
        depths=(8, 8, 8, 8, 8),
        lk_kernel=15,
        mlp_ratio=1.75,
        scale=scale,
        use_channel_attn=True,
        use_spatial_attn=True,
        res_scale=0.1,
        arch_option='aether_pro',
        **kwargs
    )


def aether_extreme(scale: int, **kwargs) -> AetherNet:
    """
    Peak performance 'Extreme' version, designed to compete with SOTA models.

    WARNING: This preset is for research and enthusiast use. It has extremely
    high VRAM requirements and will be very slow to train. It is not recommended
    for hardware with 12GB of VRAM or less without using advanced techniques
    like very small patch sizes and gradient accumulation.

    This preset pushes the AetherNet architecture to its limits by:
    - A massive channel dimension (256) for unparalleled feature richness.
    - Extreme depth (12 blocks per stage) for maximum sequential refinement.
    - A huge reparameterized kernel (21x21) to create an enormous receptive field.

    Args:
        scale: Super-resolution scale factor
        **kwargs: Additional keyword arguments for AetherNet
    """
    return AetherNet(
        embed_dim=256,
        depths=(12, 12, 12, 12, 12),
        lk_kernel=21,
        mlp_ratio=2.0,
        scale=scale,
        use_channel_attn=True,
        use_spatial_attn=True,
        res_scale=0.1,
        arch_option='aether_extreme',
        **kwargs
    )