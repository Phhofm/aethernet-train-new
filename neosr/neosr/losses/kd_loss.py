# File: neosr/neosr/losses/kd_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from neosr.utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss for SISR.
    Combines feature-based and output-based distillation.
    """
    def __init__(self, loss_weight=1.0, feature_weight=0.5, output_weight=0.5, criterion='l1', student_channels=64, teacher_channels=180):
        super().__init__()
        self.loss_weight = loss_weight
        self.feature_weight = feature_weight
        self.output_weight = output_weight

        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'l2':
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError(f'Criterion {criterion} not implemented')

        # 1x1 conv to align the feature channels from teacher to student
        self.align_conv = nn.Conv2d(teacher_channels, student_channels, 1, 1, 0)

    def forward(self, student_output, teacher_output, student_features, teacher_features):
        # 1. Output-based distillation
        loss_output = self.criterion(student_output, teacher_output)

        # 2. Feature-based distillation
        # We'll compare the final feature map from the body of each network
        s_feat = student_features[0]
        t_feat = teacher_features[0]

        # Align teacher features to student feature dimensions
        t_feat_aligned = self.align_conv(t_feat)
        
        # Resize if necessary (though they should be the same size before upsampling)
        if t_feat_aligned.shape != s_feat.shape:
             t_feat_aligned = F.interpolate(t_feat_aligned, size=s_feat.shape[2:], mode='bilinear', align_corners=False)

        loss_feature = self.criterion(s_feat, t_feat_aligned)

        # Combine losses
        total_loss = (self.output_weight * loss_output) + (self.feature_weight * loss_feature)
        
        return total_loss * self.loss_weight