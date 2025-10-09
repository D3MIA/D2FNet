#!/usr/bin/env python3
"""
INFERENCE + OVERLAY FOLD 2 - ENHANCED VERSION
==============================================
Améliorations:
- Omission du timestep 0
- Carte d'erreur complète (pas de trous)
- Analyse des trous dans les forces
- Graphes détaillés de tous les metrics
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import pickle
from tqdm import tqdm
from scipy.ndimage import maximum_filter, gaussian_filter
from sklearn.metrics import r2_score, mean_absolute_error
import sys
import glob
import argparse
import seaborn as sns

sys.path.append('.')
from spatial_cnn_ultra_stable import StableSpatialDataset
from advanced_r2_model import AdvancedUNet

def map_grid_to_image_smooth(grid_forces, mapping_info, image_shape=(1080, 1920), smooth=True):
    """
    Map forces from grid 256x256 to original image WITH SMOOTH INTERPOLATION
    to fill holes and create continuous force field
    """
    grid_x = mapping_info['grid_x']
    grid_y = mapping_info['grid_y']
    projected_pixels = mapping_info['projected_pixels']
    
    # Extract node forces from grid
    node_forces = grid_forces[grid_y, grid_x]
    
    # Get pixel positions on original image
    img_x = projected_pixels[:, 0].astype(int)
    img_y = projected_pixels[:, 1].astype(int)
    img_x = np.clip(img_x, 0, image_shape[1] - 1)
    img_y = np.clip(img_y, 0, image_shape[0] - 1)
    
    # Map forces to image
    image_forces = np.zeros(image_shape, dtype=np.float32)
    image_forces[img_y, img_x] = node_forces
    
    if smooth:
        # Dilate puis smooth pour éliminer les trous
        image_forces = maximum_filter(image_forces, size=9)
        image_forces = gaussian_filter(image_forces, sigma=2.0)
    
    return image_forces

def compute_alpha_map(force_img):
    """Compute alpha channel with smooth progressive transparency"""
    alpha = np.zeros_like(force_img)
    
    # Forces < 0.05N: INVISIBLE
    mask_invisible = force_img < 0.05
    alpha[mask_invisible] = 0.0
    
    # Forces >= 0.05N: Progressive transparency
    mask_visible = force_img >= 0.05
    alpha[mask_visible] = 0.15 + 0.75 * np.tanh((force_img[mask_visible] - 0.05) / 1.0)
    
    return alpha

def create_overlay(frame_idx, brain_img, gt_grid, pred_grid, mapping_info, output_path, model_name="Fold 2"):
    """Create enhanced overlay visualization with complete error map"""
    
    img_h, img_w = brain_img.shape[:2]
    
    # Map to original image WITH SMOOTHING
    gt_img = map_grid_to_image_smooth(gt_grid, mapping_info, (img_h, img_w), smooth=True)
    pred_img = map_grid_to_image_smooth(pred_grid, mapping_info, (img_h, img_w), smooth=True)
    
    # Calculate metrics on ALL pixels where GT > 0.05N (force significative)
    valid_mask = gt_img >= 0.05
    
    if valid_mask.sum() > 0:
        gt_valid = gt_img[valid_mask]
        pred_valid = pred_img[valid_mask]
        
        r2 = r2_score(gt_valid, pred_valid)
        mae = mean_absolute_error(gt_valid, pred_valid)
        rmse = np.sqrt(np.mean((pred_valid - gt_valid) ** 2))
        
        # Accuracy metrics (within tolerance)
        acc_50mN = np.mean(np.abs(pred_valid - gt_valid) <= 0.05) * 100
        acc_100mN = np.mean(np.abs(pred_valid - gt_valid) <= 0.1) * 100
        acc_200mN = np.mean(np.abs(pred_valid - gt_valid) <= 0.2) * 100
        
        # Absolute error map (COMPLETE - no holes!)
        abs_error_img = np.zeros_like(gt_img)
        abs_error_img[valid_mask] = np.abs(pred_valid - gt_valid)
        
        # Relative error map (COMPLETE - no holes!)
        rel_error_img = np.zeros_like(gt_img)
        rel_error_img[valid_mask] = np.abs(pred_valid - gt_valid) / (gt_valid + 1e-10)
        
    else:
        r2, mae, rmse = 0.0, 0.0, 0.0
        acc_50mN, acc_100mN, acc_200mN = 0.0, 0.0, 0.0
        abs_error_img = np.zeros_like(gt_img)
        rel_error_img = np.zeros_like(gt_img)
        gt_valid = np.array([0])
        pred_valid = np.array([0])
    
    # Compute smooth alpha maps
    gt_alpha = compute_alpha_map(gt_img)
    pred_alpha = compute_alpha_map(pred_img)
    error_alpha = compute_alpha_map(gt_img)  # Same mask as GT
    
    # Create visualization (3x2 layout)
    fig, axes = plt.subplots(3, 2, figsize=(20, 18))
    
    # Panel 1: Ground Truth Forces
    ax = axes[0, 0]
    ax.imshow(brain_img, cmap='gray')
    gt_masked = np.ma.masked_where(gt_img < 0.05, gt_img)
    im1 = ax.imshow(gt_masked, cmap='hot', alpha=gt_alpha, vmin=0, vmax=2.0, interpolation='bilinear')
    ax.set_title(f'Ground Truth Forces - Frame {frame_idx}', fontweight='bold', fontsize=16)
    ax.axis('off')
    cbar1 = plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    cbar1.set_label('Force (N)', fontsize=12)
    
    # Panel 2: Predicted Forces
    ax = axes[0, 1]
    ax.imshow(brain_img, cmap='gray')
    pred_masked = np.ma.masked_where(pred_img < 0.05, pred_img)
    im2 = ax.imshow(pred_masked, cmap='hot', alpha=pred_alpha, vmin=0, vmax=2.0, interpolation='bilinear')
    ax.set_title(f'Predicted Forces - {model_name}\nR² = {r2:.4f} | MAE = {mae:.4f} N', 
                 fontweight='bold', fontsize=16)
    ax.axis('off')
    cbar2 = plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cbar2.set_label('Force (N)', fontsize=12)
    
    # Panel 3: Absolute Error (COMPLETE - no holes!)
    ax = axes[1, 0]
    ax.imshow(brain_img, cmap='gray')
    abs_error_masked = np.ma.masked_where(gt_img < 0.05, abs_error_img)
    im3 = ax.imshow(abs_error_masked, cmap='viridis', alpha=error_alpha, vmin=0, vmax=0.5, interpolation='bilinear')
    ax.set_title(f'Absolute Error: |Pred - GT|\nRMSE = {rmse:.4f} N', 
                 fontweight='bold', fontsize=16)
    ax.axis('off')
    cbar3 = plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)
    cbar3.set_label('Abs Error (N)', fontsize=12)
    
    # Panel 4: Relative Error (COMPLETE - no holes!)
    ax = axes[1, 1]
    ax.imshow(brain_img, cmap='gray')
    rel_error_masked = np.ma.masked_where(gt_img < 0.05, rel_error_img)
    im4 = ax.imshow(rel_error_masked, cmap='plasma', alpha=error_alpha, vmin=0, vmax=1, interpolation='bilinear')
    ax.set_title(f'Relative Error: |Pred - GT| / GT\nMedian = {np.median(rel_error_img[valid_mask]):.2%}', 
                 fontweight='bold', fontsize=16)
    ax.axis('off')
    cbar4 = plt.colorbar(im4, ax=ax, fraction=0.046, pad=0.04)
    cbar4.set_label('Relative Error', fontsize=12)
    
    # Panel 5: Scatter Plot GT vs Pred
    ax = axes[2, 0]
    if valid_mask.sum() > 0:
        # Downsample pour lisibilité si trop de points
        n_points = len(gt_valid)
        if n_points > 5000:
            idx = np.random.choice(n_points, 5000, replace=False)
            gt_sample = gt_valid[idx]
            pred_sample = pred_valid[idx]
        else:
            gt_sample = gt_valid
            pred_sample = pred_valid
        
        ax.scatter(gt_sample, pred_sample, alpha=0.3, s=10, c='blue', edgecolors='none')
        
        # Perfect prediction line
        max_val = max(gt_sample.max(), pred_sample.max())
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect prediction')
        
        ax.set_xlabel('Ground Truth (N)', fontsize=12)
        ax.set_ylabel('Prediction (N)', fontsize=12)
        ax.set_title(f'GT vs Prediction Scatter\n{len(gt_valid):,} points', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    # Panel 6: Metrics Summary
    ax = axes[2, 1]
    ax.axis('off')
    
    metrics_text = f"""
METRICS SUMMARY - Frame {frame_idx}
Model: {model_name}

📊 Performance:
  R² Score:           {r2:.4f}
  MAE:                {mae:.4f} N
  RMSE:               {rmse:.4f} N
  
🎯 Accuracy (within tolerance):
  @ ±50mN:            {acc_50mN:.1f}%
  @ ±100mN:           {acc_100mN:.1f}%
  @ ±200mN:           {acc_200mN:.1f}%
  
📈 Ground Truth:
  Min:                {gt_valid.min():.4f} N
  Max:                {gt_valid.max():.4f} N
  Mean:               {gt_valid.mean():.4f} N
  Median:             {np.median(gt_valid):.4f} N
  
🔮 Prediction:
  Min:                {pred_valid.min():.4f} N
  Max:                {pred_valid.max():.4f} N
  Mean:               {pred_valid.mean():.4f} N
  Median:             {np.median(pred_valid):.4f} N
  
📍 Coverage:
  Valid pixels:       {valid_mask.sum():,}
  Total nodes:        {mapping_info['n_nodes']:,}
  Coverage:           {valid_mask.sum() / (img_h * img_w) * 100:.2f}%
"""
    
    ax.text(0.05, 0.5, metrics_text, fontsize=11, family='monospace',
            verticalalignment='center', 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.suptitle(f'{model_name} - ENHANCED OVERLAY - Frame {frame_idx}', 
                 fontsize=20, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return {
        'r2': r2, 
        'mae': mae, 
        'rmse': rmse,
        'acc_50mN': acc_50mN,
        'acc_100mN': acc_100mN,
        'acc_200mN': acc_200mN,
        'gt_min': gt_valid.min(),
        'gt_max': gt_valid.max(),
        'gt_mean': gt_valid.mean(),
        'pred_min': pred_valid.min(),
        'pred_max': pred_valid.max(),
        'pred_mean': pred_valid.mean(),
        'n_valid': valid_mask.sum()
    }

def create_comprehensive_analysis(all_metrics, output_dir, val_r2, model_name):
    """Create comprehensive analysis with multiple detailed plots"""
    
    frames = list(range(len(all_metrics['r2'])))
    
    # ========================================================================
    # Figure 1: Main Metrics Evolution (4 subplots)
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # R² evolution
    ax = axes[0, 0]
    ax.plot(frames, all_metrics['r2'], alpha=0.7, linewidth=1.5, label='Per-frame R²')
    ax.axhline(y=val_r2, color='red', linestyle='--', linewidth=2, label=f'Validation R²={val_r2:.4f}')
    ax.axhline(y=np.mean(all_metrics['r2']), color='green', linestyle=':', linewidth=2, 
               label=f"Mean R²={np.mean(all_metrics['r2']):.4f}")
    ax.fill_between(frames, all_metrics['r2'], alpha=0.2)
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('R² Score Evolution Over Frames', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # MAE evolution
    ax = axes[0, 1]
    ax.plot(frames, all_metrics['mae'], alpha=0.7, linewidth=1.5, color='orange', label='MAE')
    ax.axhline(y=np.mean(all_metrics['mae']), color='red', linestyle='--', linewidth=2,
               label=f"Mean={np.mean(all_metrics['mae']):.4f} N")
    ax.fill_between(frames, all_metrics['mae'], alpha=0.2, color='orange')
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('MAE (N)', fontsize=12)
    ax.set_title('Mean Absolute Error Over Frames', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # RMSE evolution
    ax = axes[1, 0]
    ax.plot(frames, all_metrics['rmse'], alpha=0.7, linewidth=1.5, color='purple', label='RMSE')
    ax.axhline(y=np.mean(all_metrics['rmse']), color='red', linestyle='--', linewidth=2,
               label=f"Mean={np.mean(all_metrics['rmse']):.4f} N")
    ax.fill_between(frames, all_metrics['rmse'], alpha=0.2, color='purple')
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('RMSE (N)', fontsize=12)
    ax.set_title('Root Mean Squared Error Over Frames', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy @ different tolerances
    ax = axes[1, 1]
    ax.plot(frames, all_metrics['acc_50mN'], alpha=0.7, linewidth=1.5, label='±50mN', color='green')
    ax.plot(frames, all_metrics['acc_100mN'], alpha=0.7, linewidth=1.5, label='±100mN', color='blue')
    ax.plot(frames, all_metrics['acc_200mN'], alpha=0.7, linewidth=1.5, label='±200mN', color='red')
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy Within Tolerance Bands', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    plt.suptitle(f'{model_name} - Main Metrics Evolution', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'analysis_1_main_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # Figure 2: Ground Truth vs Prediction Statistics (2x2)
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # GT Force evolution
    ax = axes[0, 0]
    ax.plot(frames, all_metrics['gt_mean'], label='Mean', linewidth=2, color='blue')
    ax.plot(frames, all_metrics['gt_max'], label='Max', linewidth=1.5, color='red', alpha=0.7)
    ax.plot(frames, all_metrics['gt_min'], label='Min', linewidth=1.5, color='green', alpha=0.7)
    ax.fill_between(frames, all_metrics['gt_min'], all_metrics['gt_max'], alpha=0.1, color='blue')
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Force (N)', fontsize=12)
    ax.set_title('Ground Truth Force Statistics', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Pred Force evolution
    ax = axes[0, 1]
    ax.plot(frames, all_metrics['pred_mean'], label='Mean', linewidth=2, color='orange')
    ax.plot(frames, all_metrics['pred_max'], label='Max', linewidth=1.5, color='red', alpha=0.7)
    ax.plot(frames, all_metrics['pred_min'], label='Min', linewidth=1.5, color='green', alpha=0.7)
    ax.fill_between(frames, all_metrics['pred_min'], all_metrics['pred_max'], alpha=0.1, color='orange')
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Force (N)', fontsize=12)
    ax.set_title('Predicted Force Statistics', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mean GT vs Mean Pred comparison
    ax = axes[1, 0]
    ax.plot(frames, all_metrics['gt_mean'], label='GT Mean', linewidth=2, color='blue')
    ax.plot(frames, all_metrics['pred_mean'], label='Pred Mean', linewidth=2, color='orange', linestyle='--')
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Mean Force (N)', fontsize=12)
    ax.set_title('Mean Force Comparison: GT vs Prediction', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Force bias analysis
    ax = axes[1, 1]
    bias = np.array(all_metrics['pred_mean']) - np.array(all_metrics['gt_mean'])
    ax.plot(frames, bias, linewidth=2, color='purple')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.axhline(y=np.mean(bias), color='red', linestyle=':', linewidth=2, 
               label=f'Mean Bias={np.mean(bias):.4f} N')
    ax.fill_between(frames, bias, alpha=0.2, color='purple')
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Bias (Pred - GT) [N]', fontsize=12)
    ax.set_title('Prediction Bias Over Frames', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} - Force Statistics Analysis', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'analysis_2_force_stats.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # Figure 3: Distribution Analysis (2x2)
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # R² distribution
    ax = axes[0, 0]
    ax.hist(all_metrics['r2'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(x=np.mean(all_metrics['r2']), color='red', linestyle='--', linewidth=2,
               label=f'Mean={np.mean(all_metrics["r2"]):.4f}')
    ax.axvline(x=np.median(all_metrics['r2']), color='green', linestyle=':', linewidth=2,
               label=f'Median={np.median(all_metrics["r2"]):.4f}')
    ax.set_xlabel('R² Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('R² Score Distribution', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # MAE distribution
    ax = axes[0, 1]
    ax.hist(all_metrics['mae'], bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax.axvline(x=np.mean(all_metrics['mae']), color='red', linestyle='--', linewidth=2,
               label=f'Mean={np.mean(all_metrics["mae"]):.4f} N')
    ax.axvline(x=np.median(all_metrics['mae']), color='green', linestyle=':', linewidth=2,
               label=f'Median={np.median(all_metrics["mae"]):.4f} N')
    ax.set_xlabel('MAE (N)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('MAE Distribution', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Accuracy boxplot
    ax = axes[1, 0]
    data_to_plot = [all_metrics['acc_50mN'], all_metrics['acc_100mN'], all_metrics['acc_200mN']]
    bp = ax.boxplot(data_to_plot, labels=['±50mN', '±100mN', '±200mN'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['green', 'blue', 'red']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy Distribution Across Tolerance Bands', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Coverage analysis
    ax = axes[1, 1]
    ax.plot(frames, all_metrics['n_valid'], linewidth=2, color='teal')
    ax.axhline(y=np.mean(all_metrics['n_valid']), color='red', linestyle='--', linewidth=2,
               label=f'Mean={np.mean(all_metrics["n_valid"]):.0f} pixels')
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Valid Pixels', fontsize=12)
    ax.set_title('Valid Pixel Coverage Over Frames', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} - Distribution Analysis', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'analysis_3_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # Figure 4: Correlation Analysis (2x2)
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # R² vs Mean GT Force
    ax = axes[0, 0]
    ax.scatter(all_metrics['gt_mean'], all_metrics['r2'], alpha=0.5, s=30, color='blue')
    ax.set_xlabel('Mean GT Force (N)', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('R² vs Mean Ground Truth Force', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # MAE vs Mean GT Force
    ax = axes[0, 1]
    ax.scatter(all_metrics['gt_mean'], all_metrics['mae'], alpha=0.5, s=30, color='orange')
    ax.set_xlabel('Mean GT Force (N)', fontsize=12)
    ax.set_ylabel('MAE (N)', fontsize=12)
    ax.set_title('MAE vs Mean Ground Truth Force', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # R² vs Max GT Force
    ax = axes[1, 0]
    ax.scatter(all_metrics['gt_max'], all_metrics['r2'], alpha=0.5, s=30, color='purple')
    ax.set_xlabel('Max GT Force (N)', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('R² vs Max Ground Truth Force', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Accuracy vs Coverage
    ax = axes[1, 1]
    ax.scatter(all_metrics['n_valid'], all_metrics['acc_100mN'], alpha=0.5, s=30, color='green')
    ax.set_xlabel('Valid Pixels', fontsize=12)
    ax.set_ylabel('Accuracy @ ±100mN (%)', fontsize=12)
    ax.set_title('Accuracy vs Valid Pixel Coverage', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} - Correlation Analysis', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'analysis_4_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # Summary Statistics Text File
    # ========================================================================
    summary_text = f"""
{'='*80}
{model_name} - COMPREHENSIVE ANALYSIS SUMMARY
{'='*80}

DATASET INFORMATION:
  Total frames analyzed:     {len(frames)}
  Frame range:               {min(frames)} - {max(frames)}
  Validation R² (model):     {val_r2:.4f}

{'='*80}
PERFORMANCE METRICS:
{'='*80}

R² Score:
  Mean:                      {np.mean(all_metrics['r2']):.4f}
  Std Dev:                   {np.std(all_metrics['r2']):.4f}
  Median:                    {np.median(all_metrics['r2']):.4f}
  Min:                       {np.min(all_metrics['r2']):.4f} (frame {np.argmin(all_metrics['r2']) + min(frames)})
  Max:                       {np.max(all_metrics['r2']):.4f} (frame {np.argmax(all_metrics['r2']) + min(frames)})
  Q1:                        {np.percentile(all_metrics['r2'], 25):.4f}
  Q3:                        {np.percentile(all_metrics['r2'], 75):.4f}

MAE (Mean Absolute Error):
  Mean:                      {np.mean(all_metrics['mae']):.4f} N
  Std Dev:                   {np.std(all_metrics['mae']):.4f} N
  Median:                    {np.median(all_metrics['mae']):.4f} N
  Min:                       {np.min(all_metrics['mae']):.4f} N (frame {np.argmin(all_metrics['mae']) + min(frames)})
  Max:                       {np.max(all_metrics['mae']):.4f} N (frame {np.argmax(all_metrics['mae']) + min(frames)})

RMSE (Root Mean Squared Error):
  Mean:                      {np.mean(all_metrics['rmse']):.4f} N
  Std Dev:                   {np.std(all_metrics['rmse']):.4f} N
  Median:                    {np.median(all_metrics['rmse']):.4f} N
  Min:                       {np.min(all_metrics['rmse']):.4f} N
  Max:                       {np.max(all_metrics['rmse']):.4f} N

{'='*80}
ACCURACY WITHIN TOLERANCE BANDS:
{'='*80}

@ ±50mN (0.05N):
  Mean:                      {np.mean(all_metrics['acc_50mN']):.2f}%
  Std Dev:                   {np.std(all_metrics['acc_50mN']):.2f}%
  Min:                       {np.min(all_metrics['acc_50mN']):.2f}%
  Max:                       {np.max(all_metrics['acc_50mN']):.2f}%

@ ±100mN (0.1N):
  Mean:                      {np.mean(all_metrics['acc_100mN']):.2f}%
  Std Dev:                   {np.std(all_metrics['acc_100mN']):.2f}%
  Min:                       {np.min(all_metrics['acc_100mN']):.2f}%
  Max:                       {np.max(all_metrics['acc_100mN']):.2f}%

@ ±200mN (0.2N):
  Mean:                      {np.mean(all_metrics['acc_200mN']):.2f}%
  Std Dev:                   {np.std(all_metrics['acc_200mN']):.2f}%
  Min:                       {np.min(all_metrics['acc_200mN']):.2f}%
  Max:                       {np.max(all_metrics['acc_200mN']):.2f}%

{'='*80}
GROUND TRUTH FORCE STATISTICS:
{'='*80}

Mean Force:
  Average:                   {np.mean(all_metrics['gt_mean']):.4f} N
  Range:                     {np.min(all_metrics['gt_mean']):.4f} - {np.max(all_metrics['gt_mean']):.4f} N

Max Force:
  Average:                   {np.mean(all_metrics['gt_max']):.4f} N
  Range:                     {np.min(all_metrics['gt_max']):.4f} - {np.max(all_metrics['gt_max']):.4f} N

Min Force:
  Average:                   {np.mean(all_metrics['gt_min']):.4f} N
  Range:                     {np.min(all_metrics['gt_min']):.4f} - {np.max(all_metrics['gt_min']):.4f} N

{'='*80}
PREDICTION STATISTICS:
{'='*80}

Mean Force:
  Average:                   {np.mean(all_metrics['pred_mean']):.4f} N
  Range:                     {np.min(all_metrics['pred_mean']):.4f} - {np.max(all_metrics['pred_mean']):.4f} N

Max Force:
  Average:                   {np.mean(all_metrics['pred_max']):.4f} N
  Range:                     {np.min(all_metrics['pred_max']):.4f} - {np.max(all_metrics['pred_max']):.4f} N

Min Force:
  Average:                   {np.mean(all_metrics['pred_min']):.4f} N
  Range:                     {np.min(all_metrics['pred_min']):.4f} - {np.max(all_metrics['pred_min']):.4f} N

Prediction Bias (Pred - GT):
  Mean Bias:                 {np.mean(np.array(all_metrics['pred_mean']) - np.array(all_metrics['gt_mean'])):.4f} N
  Std Dev:                   {np.std(np.array(all_metrics['pred_mean']) - np.array(all_metrics['gt_mean'])):.4f} N

{'='*80}
COVERAGE STATISTICS:
{'='*80}

Valid Pixels (GT > 0.05N):
  Mean:                      {np.mean(all_metrics['n_valid']):.0f} pixels
  Std Dev:                   {np.std(all_metrics['n_valid']):.0f} pixels
  Min:                       {np.min(all_metrics['n_valid']):.0f} pixels
  Max:                       {np.max(all_metrics['n_valid']):.0f} pixels

{'='*80}
BEST AND WORST FRAMES:
{'='*80}

Best R² Frames (Top 5):
"""
    
    # Top 5 best R²
    best_r2_idx = np.argsort(all_metrics['r2'])[-5:][::-1]
    for idx in best_r2_idx:
        summary_text += f"  Frame {idx + min(frames):4d}: R²={all_metrics['r2'][idx]:.4f}, MAE={all_metrics['mae'][idx]:.4f} N\n"
    
    summary_text += f"\nWorst R² Frames (Bottom 5):\n"
    worst_r2_idx = np.argsort(all_metrics['r2'])[:5]
    for idx in worst_r2_idx:
        summary_text += f"  Frame {idx + min(frames):4d}: R²={all_metrics['r2'][idx]:.4f}, MAE={all_metrics['mae'][idx]:.4f} N\n"
    
    summary_text += f"\n{'='*80}\n"
    
    # Save summary
    with open(output_dir / 'analysis_summary.txt', 'w') as f:
        f.write(summary_text)
    
    print(summary_text)

def main():
    parser = argparse.ArgumentParser(description='Inference + Overlay Fold 2 - Enhanced')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Chemin vers le checkpoint du modèle')
    parser.add_argument('--npz', type=str, required=True,
                        help='Chemin vers le fichier NPZ')
    parser.add_argument('--images', type=str, required=True,
                        help='Dossier contenant les images')
    parser.add_argument('--output', type=str, required=True,
                        help='Dossier de sortie')
    parser.add_argument('--start_frame', type=int, default=1,
                        help='Frame de début (default=1 pour omettre timestep 0)')
    parser.add_argument('--end_frame', type=int, default=1999,
                        help='Frame de fin')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda ou cpu)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔮 INFERENCE + OVERLAY - FOLD 2 ENHANCED")
    print("=" * 80)
    print(f"⚠️  Omission du timestep 0 (start_frame={args.start_frame})")
    print(f"✨ Cartes d'erreur complètes (smoothing + interpolation)")
    print(f"📊 Analyse détaillée avec multiples graphes")
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Création output dir
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # ÉTAPE 1: Chargement dataset
    # ========================================================================
    print("\n" + "=" * 80)
    print("📦 ÉTAPE 1: Chargement dataset...")
    print("=" * 80)
    
    dataset = StableSpatialDataset(
        npz_files=[args.npz],
        max_timesteps=2000,
        grid_size=(256, 256)
    )
    print(f"✅ Dataset chargé: {len(dataset)} samples")
    
    # ========================================================================
    # ÉTAPE 2: Chargement mapping info
    # ========================================================================
    print("\n" + "=" * 80)
    print("🗺️  ÉTAPE 2: Chargement mapping...")
    print("=" * 80)
    
    data = np.load(args.npz)
    projected_pixels = data['projected_pixels'][0]
    x_coords, y_coords = projected_pixels[:, 0], projected_pixels[:, 1]
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # Grid coordinates
    grid_x = ((x_coords - x_min) / (x_max - x_min + 1e-8) * 255).astype(int)
    grid_y = ((y_coords - y_min) / (y_max - y_min + 1e-8) * 255).astype(int)
    
    mapping_info = {
        'projected_pixels': projected_pixels,
        'grid_x': grid_x,
        'grid_y': grid_y,
        'n_nodes': len(projected_pixels)
    }
    
    print(f"✅ Mapping chargé: {len(projected_pixels)} nodes")
    
    # ========================================================================
    # ÉTAPE 3: Chargement modèle
    # ========================================================================
    print("\n" + "=" * 80)
    print("🏆 ÉTAPE 3: Chargement modèle...")
    print("=" * 80)
    
    model = AdvancedUNet(in_channels=5).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    val_r2 = checkpoint.get('val_r2', checkpoint.get('best_r2', 0.0))
    epoch = checkpoint.get('epoch', 0)
    fold = checkpoint.get('fold', 2)
    
    model_name = f"Fold {fold} (Val R²={val_r2:.4f})"
    
    print(f"✅ Modèle chargé:")
    print(f"   Fold: {fold}")
    print(f"   Validation R²: {val_r2:.4f}")
    print(f"   Epoch: {epoch}")
    
    # ========================================================================
    # ÉTAPE 4: Inférence et génération overlays
    # ========================================================================
    print("\n" + "=" * 80)
    print("🚀 ÉTAPE 4: Inférence et overlays...")
    print("=" * 80)
    
    all_metrics = {
        'r2': [], 'mae': [], 'rmse': [],
        'acc_50mN': [], 'acc_100mN': [], 'acc_200mN': [],
        'gt_min': [], 'gt_max': [], 'gt_mean': [],
        'pred_min': [], 'pred_max': [], 'pred_mean': [],
        'n_valid': []
    }
    
    end_frame = min(args.end_frame, len(dataset) - 1)
    
    for frame_idx in tqdm(range(args.start_frame, end_frame + 1), desc="Processing"):
        # Image de fond
        img_path = Path(args.images) / f"frame_{frame_idx:04d}.png"
        
        if not img_path.exists():
            print(f"⚠️  Image non trouvée: {img_path}")
            continue
        
        brain_img = np.array(Image.open(img_path))
        
        # Dataset (input, target)
        if frame_idx >= len(dataset):
            break
        
        input_grid, target_grid = dataset[frame_idx]
        input_tensor = input_grid.unsqueeze(0).to(device)
        target_grid = target_grid.numpy()
        
        # Prédiction
        with torch.no_grad():
            pred_grid = model(input_tensor).squeeze().cpu().numpy()
        
        # Overlay
        output_file = output_dir / f"overlay_{frame_idx:04d}.png"
        metrics = create_overlay(
            frame_idx, brain_img, target_grid, pred_grid,
            mapping_info, output_file, model_name=model_name
        )
        
        # Store metrics
        for key in all_metrics.keys():
            all_metrics[key].append(metrics[key])
    
    # ========================================================================
    # ÉTAPE 5: Analyse complète et graphes
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 ÉTAPE 5: Analyse complète...")
    print("=" * 80)
    
    create_comprehensive_analysis(all_metrics, output_dir, val_r2, model_name)
    
    # Sauvegarde métriques brutes
    metrics_file = output_dir / "metrics.pkl"
    with open(metrics_file, 'wb') as f:
        pickle.dump(all_metrics, f)
    
    print(f"\n✅ TERMINÉ!")
    print(f"📁 Output: {output_dir}")
    print(f"📊 4 fichiers d'analyse graphique créés")
    print(f"📄 1 fichier résumé texte créé")

if __name__ == '__main__':
    main()
