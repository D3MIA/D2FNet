#!/usr/bin/env python3
"""
D2FNet Architecture with Multi-Scale Attention
======================================================
Enhanced architecture with attention mechanisms and pyramid pooling
for improved force prediction (target R² > 0.8).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiScaleAttention(nn.Module):
    """Multi-scale attention module combining spatial and channel attention."""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        spatial_att = self.spatial_attention(x)
        channel_att = self.channel_attention(x)
        attended = x * spatial_att * channel_att
        return attended

class PyramidPooling(nn.Module):
    """Pyramid pooling module for multi-scale context aggregation."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Multi-scale pooling at different resolutions
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1),   # Global context
            nn.AdaptiveAvgPool2d(2),   # 2x2
            nn.AdaptiveAvgPool2d(4),   # 4x4
            nn.AdaptiveAvgPool2d(8),   # 8x8
        ])
        
        pool_channels = out_channels // 4
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, pool_channels, 1) for _ in self.pools
        ])
        
        self.final_conv = nn.Conv2d(
            in_channels + len(self.pools) * pool_channels, 
            out_channels, 3, padding=1
        )
    
    def forward(self, x):
        h, w = x.size(2), x.size(3)
        
        pool_features = []
        for pool, conv in zip(self.pools, self.convs):
            pooled = pool(x)
            conv_out = conv(pooled)
            upsampled = F.interpolate(conv_out, size=(h, w), mode='bilinear', align_corners=False)
            pool_features.append(upsampled)
        
        cat_features = torch.cat([x] + pool_features, dim=1)
        return self.final_conv(cat_features)

class D2FNet(nn.Module):
    """
    D2FNet with multi-scale attention and pyramid pooling.
    Architecture: 5→32→64→128→256→512 with attention at each level.
    """
    
    def __init__(self, in_channels=5):
        super().__init__()
        
        # Encoder path with attention
        self.enc1 = self._conv_block_with_attention(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = self._conv_block_with_attention(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = self._conv_block_with_attention(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = self._conv_block_with_attention(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck with pyramid pooling
        self.bottleneck = nn.Sequential(
            self._conv_block_with_attention(256, 512),
            PyramidPooling(512, 512)
        )
        
        # Decoder path with skip connections
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = self._conv_block_with_attention(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self._conv_block_with_attention(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self._conv_block_with_attention(128, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = self._conv_block_with_attention(64, 32)
        
        # Multi-scale output heads
        self.output_conv = nn.Conv2d(32, 16, 3, padding=1)
        
        # Specialized heads for different force ranges
        self.low_force_head = nn.Conv2d(16, 1, 1)     # 0-0.5N
        self.med_force_head = nn.Conv2d(16, 1, 1)     # 0-1.5N
        self.high_force_head = nn.Conv2d(16, 1, 1)    # 0-3.0N
        
        # Adaptive fusion with attention
        self.fusion_attention = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 3, 3, padding=1),
            nn.Softmax(dim=1)
        )
        
        self._initialize_weights()
    
    def _conv_block_with_attention(self, in_ch, out_ch):
        """Convolutional block with batch norm, ReLU, attention, and dropout."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            MultiScaleAttention(out_ch),
            nn.Dropout2d(0.1)
        )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        mask = x[:, 3:4, :, :]
        
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        b = self.bottleneck(self.pool4(e4))
        
        # Decoder path with skip connections
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        # Multi-scale output heads
        features = self.output_conv(d1)
        
        low_pred = torch.sigmoid(self.low_force_head(features)) * 0.5
        med_pred = torch.sigmoid(self.med_force_head(features)) * 1.5
        high_pred = torch.sigmoid(self.high_force_head(features)) * 3.0
        
        multi_preds = torch.cat([low_pred, med_pred, high_pred], dim=1)
        
        # Adaptive fusion
        fusion_weights = self.fusion_attention(multi_preds)
        final_output = (multi_preds * fusion_weights).sum(dim=1, keepdim=True)
        
        # Apply brain mask
        final_output = final_output * mask
        
        return final_output.squeeze(1)

class AdaptiveR2Loss(nn.Module):
    """Loss function that directly optimizes R² score."""
    
    def __init__(self, r2_weight=0.3):
        super().__init__()
        self.r2_weight = r2_weight
        self.mse_weight = 1.0 - r2_weight
    
    def forward(self, pred, target):
        valid_mask = target >= 0
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred.device)
        
        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]
        
        mse_loss = F.mse_loss(pred_valid, target_valid)
        
        # R² loss: minimize (1 - R²)
        if len(target_valid) > 1:
            target_var = torch.var(target_valid)
            residual_var = torch.var(pred_valid - target_valid)
            
            r2 = 1 - (residual_var / (target_var + 1e-8))
            r2_loss = 1 - torch.clamp(r2, min=0, max=1)
            
            total_loss = self.mse_weight * mse_loss + self.r2_weight * r2_loss
            return total_loss
        else:
            return mse_loss