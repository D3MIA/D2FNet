#!/usr/bin/env python3
"""
Anti-Ghost Loss Function
=========================
Loss function that penalizes over-predictions in low-force regions
to eliminate ghost artifacts in force predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AntiGhostLoss(nn.Module):
    """
    Loss function to combat ghost forces (spurious high predictions in low-force zones).
    Applies higher penalty to over-predictions when ground truth forces are low.
    """
    
    def __init__(self, 
                 base_weight=1.0,
                 ghost_penalty=5.0,
                 low_force_threshold=0.05):
        """
        Args:
            base_weight: Base MSE weight for normal errors
            ghost_penalty: Penalty multiplier for over-predictions (e.g., 5.0 = 5x penalty)
            low_force_threshold: Threshold defining "low force" frames (in Newtons)
        """
        super().__init__()
        self.base_weight = base_weight
        self.ghost_penalty = ghost_penalty
        self.low_force_threshold = low_force_threshold
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predictions, shape (B, 1, H, W) or (B, H, W)
            target: Ground truth, shape (B, H, W)
        """
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        
        valid_mask = target >= 0
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred.device)
        
        batch_size = pred.size(0)
        losses = []
        
        for b in range(batch_size):
            pred_b = pred[b][valid_mask[b]]
            target_b = target[b][valid_mask[b]]
            
            if len(target_b) == 0:
                continue
            
            target_mean = target_b.mean()
            is_low_force_frame = target_mean < self.low_force_threshold
            error = pred_b - target_b
            mse_loss = error ** 2
            
            if is_low_force_frame:
                # Low force frame: heavily penalize over-predictions
                over_prediction_mask = error > 0
                under_prediction_mask = error <= 0
                
                ghost_loss = torch.zeros_like(mse_loss)
                ghost_loss[over_prediction_mask] = mse_loss[over_prediction_mask] * self.ghost_penalty
                ghost_loss[under_prediction_mask] = mse_loss[under_prediction_mask] * self.base_weight
                batch_loss = ghost_loss.mean()
            else:
                # Normal frame: standard MSE with mild weighting
                weights = torch.ones_like(target_b)
                
                if len(target_b) > 4:
                    threshold_p75 = torch.quantile(target_b, 0.75)
                    high_force_mask = target_b > threshold_p75
                    weights[high_force_mask] = 2.0
                
                batch_loss = (mse_loss * weights).mean()
            
            losses.append(batch_loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=pred.device)
        
        return torch.stack(losses).mean()


class ImprovedAdaptiveR2Loss(nn.Module):
    """
    Hybrid loss combining R² optimization with anti-ghost penalty.
    Balances global R² score with local ghost artifact reduction.
    """
    
    def __init__(self, 
                 r2_weight=0.3,
                 ghost_penalty=3.0,
                 low_force_threshold=0.05):
        """
        Args:
            r2_weight: Weight for R² loss (e.g., 0.3 = 30% R², 70% AntiGhost)
            ghost_penalty: Penalty multiplier for ghost artifacts
            low_force_threshold: Threshold defining low force zones (in Newtons)
        """
        super().__init__()
        self.r2_weight = r2_weight
        self.mse_weight = 1.0 - r2_weight
        self.anti_ghost = AntiGhostLoss(
            base_weight=1.0,
            ghost_penalty=ghost_penalty,
            low_force_threshold=low_force_threshold
        )
    
    def forward(self, pred, target):
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        
        valid_mask = target >= 0
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred.device)
        
        ghost_loss = self.anti_ghost(pred, target)
        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]
        
        if len(target_valid) > 1:
            target_var = torch.var(target_valid)
            residual_var = torch.var(pred_valid - target_valid)
            
            r2 = 1 - (residual_var / (target_var + 1e-8))
            r2_loss = 1 - torch.clamp(r2, min=0, max=1)
            
            total_loss = self.mse_weight * ghost_loss + self.r2_weight * r2_loss
            return total_loss
        else:
            return ghost_loss


class TemporalConsistencyLoss(nn.Module):
    """
    Optional temporal consistency loss.
    Penalizes abrupt prediction changes between consecutive frames.
    """
    
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight
    
    def forward(self, pred_t, pred_t_minus_1, displacement_magnitude):
        """
        Args:
            pred_t: Prediction at time t
            pred_t_minus_1: Prediction at time t-1
            displacement_magnitude: Magnitude of displacement between frames
        """
        pred_diff = torch.abs(pred_t - pred_t_minus_1)
        expected_diff = displacement_magnitude * 0.5
        temporal_loss = F.relu(pred_diff - expected_diff).mean()
        return self.weight * temporal_loss


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ANTI-GHOST LOSS FUNCTION TESTS")
    print("=" * 80)
    
    batch_size = 4
    H, W = 256, 256
    
    target = torch.zeros(batch_size, H, W)
    pred = torch.zeros(batch_size, H, W)
    
    # Frame 0: LOW force GT (mean=0.03), model over-predicts (ghost!)
    target[0, 100:150, 100:150] = 0.03
    pred[0, 80:170, 80:170] = 0.15
    
    # Frame 1: NORMAL force GT (mean=0.5), good prediction
    target[1, 100:150, 100:150] = 0.8
    pred[1, 100:150, 100:150] = 0.7
    
    # Frame 2: HIGH force GT (mean=1.5), good prediction
    target[2, 100:150, 100:150] = 2.0
    pred[2, 95:155, 95:155] = 1.8
    
    # Frame 3: VERY LOW (mean=0.01), model predicts almost nothing
    target[3, 120:125, 120:125] = 0.02
    pred[3, 120:125, 120:125] = 0.01
    
    # Test AntiGhostLoss
    print("\nTest AntiGhostLoss:")
    anti_ghost = AntiGhostLoss(ghost_penalty=5.0, low_force_threshold=0.05)
    loss = anti_ghost(pred, target)
    print(f"   Total loss: {loss.item():.6f}")
    
    # Test per frame
    print("\n   Details per frame:")
    for b in range(batch_size):
        valid_pixels = target[b] >= 0
        target_mean = target[b][valid_pixels].mean().item()
        pred_mean = pred[b][valid_pixels].mean().item()
        is_low = target_mean < 0.05
        
        pred_b = pred[b:b+1]
        target_b = target[b:b+1]
        loss_b = anti_ghost(pred_b, target_b).item()
        
        status = "GHOST!" if is_low and pred_mean > target_mean * 2 else "OK"
        print(f"   Frame {b}: GT_mean={target_mean:.4f}, PRED_mean={pred_mean:.4f}, "
              f"Loss={loss_b:.6f} {status}")
    
    # Test ImprovedAdaptiveR2Loss
    print("\nTest ImprovedAdaptiveR2Loss:")
    improved_loss = ImprovedAdaptiveR2Loss(r2_weight=0.3, ghost_penalty=3.0)
    loss = improved_loss(pred, target)
    print(f"   Total loss: {loss.item():.6f}")
    
    # Comparison with simple MSE
    print("\nComparison with simple MSE:")
    mse_loss = F.mse_loss(pred[target >= 0], target[target >= 0])
    print(f"   Simple MSE: {mse_loss.item():.6f}")
    print(f"   AntiGhost/MSE ratio: {loss.item() / mse_loss.item():.2f}x")
    
    print("\n" + "=" * 80)
    print("TESTS COMPLETED")
    print("=" * 80)
    print("\nINTERPRETATION:")
    print("   - Frame 0 (ghost) should have HIGH loss (5x penalty)")
    print("   - Frames 1-3 (normal) should have moderate loss")
    print("   - AntiGhostLoss > simple MSE when ghosts are present")
