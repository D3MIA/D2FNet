#!/usr/bin/env python3
"""
Ultra-Stable Spatial CNN Model
===============================
CNN model for force prediction with stable normalization and preprocessing.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import logging
from pathlib import Path
import time
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error
import scipy.ndimage
from scipy.ndimage import binary_dilation, binary_fill_holes
import argparse
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('spatial_cnn_stable.log'),
        logging.StreamHandler()
    ]
)

class StableSpatialDataset(Dataset):
    """Dataset with ultra-stable normalization and preprocessing."""
    
    def __init__(self, npz_files, max_timesteps=1000, grid_size=(256, 256), 
                 exclude_pattern=None):
        self.npz_files = npz_files
        self.max_timesteps = max_timesteps
        self.grid_width, self.grid_height = grid_size
        
        if exclude_pattern:
            self.npz_files = [f for f in self.npz_files if exclude_pattern not in str(f)]
        
        logger = logging.getLogger(__name__)
        logger.info(f"Files after filtering: {len(self.npz_files)}")
        
        self._calculate_stable_statistics()
        self._create_brain_mask()
        self._precalculate_all_data()
    
    def _calculate_stable_statistics(self):
        """Calculate ultra-stable normalization statistics using P99.9 percentiles."""
        logger = logging.getLogger(__name__)
        logger.info("Calculating stable statistics...")
        
        all_dx, all_dy, all_forces = [], [], []
        
        for npz_file in self.npz_files[:min(5, len(self.npz_files))]:
            data = np.load(npz_file)
            sample_disp = data['disp2d'][:500]
            sample_force = data['force_mag'][:500]
            
            all_dx.extend(sample_disp[:, :, 0].flatten())
            all_dy.extend(sample_disp[:, :, 1].flatten())
            all_forces.extend(sample_force.flatten())
        
        self.dx_scale = np.percentile(np.abs(all_dx), 99.9)
        self.dy_scale = np.percentile(np.abs(all_dy), 99.9)
        self.force_scale = 3.0  # Fixed normalization: 2N→0.667, 3N→1.0
        
        self.force_p50 = np.percentile(all_forces, 50)
        self.force_p90 = np.percentile(all_forces, 90)
        self.force_p99 = np.percentile(all_forces, 99)
        
        logger.info(f"Statistics:")
        logger.info(f"   dx_scale (P99.9): {self.dx_scale:.4f}")
        logger.info(f"   dy_scale (P99.9): {self.dy_scale:.4f}")
        logger.info(f"   force_scale: {self.force_scale:.4f}")
        logger.info(f"   force P50/P90/P99: {self.force_p50:.4f}/{self.force_p90:.4f}/{self.force_p99:.4f}")
    
    def _create_brain_mask(self):
        """Create brain mask with border filtering (X > 1400mm excluded)."""
        logger = logging.getLogger(__name__)
        logger.info("Creating brain mask...")
        
        data = np.load(self.npz_files[0])
        coords = data['projected_pixels'][0]
        
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        
        x_threshold = 1400
        valid_nodes_mask = x_coords <= x_threshold
        n_total = len(x_coords)
        n_valid = valid_nodes_mask.sum()
        n_removed = n_total - n_valid
        logger.info(f"Border filtering (X > {x_threshold}):")
        logger.info(f"   Valid nodes: {n_valid:,} ({n_valid/n_total*100:.1f}%)")
        logger.info(f"   Removed nodes: {n_removed:,} ({n_removed/n_total*100:.1f}%)")
        
        x_coords = x_coords[valid_nodes_mask]
        y_coords = y_coords[valid_nodes_mask]
        self.valid_nodes_mask = valid_nodes_mask
        
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        grid_x = ((x_coords - x_min) / (x_max - x_min + 1e-8) * (self.grid_width - 1)).astype(int)
        grid_y = ((y_coords - y_min) / (y_max - y_min + 1e-8) * (self.grid_height - 1)).astype(int)
        
        self.brain_mask = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        self.brain_mask[grid_y, grid_x] = True
        
        self.brain_mask = binary_dilation(self.brain_mask, iterations=3)
        self.brain_mask = binary_fill_holes(self.brain_mask)
        
        from scipy.ndimage import binary_erosion
        self.brain_mask = binary_erosion(self.brain_mask, iterations=1)
        
        active_pixels = self.brain_mask.sum()
        percentage = 100 * active_pixels / (self.grid_width * self.grid_height)
        
        logger.info(f"Mask created: {active_pixels} active pixels ({percentage:.1f}%)")
    
    def _precalculate_all_data(self):
        """Precalculate all grids with stable normalization."""
        logger = logging.getLogger(__name__)
        logger.info(f"Precalculating data...")
        
        self.all_input_grids = []
        self.all_target_grids = []
        self.all_force_categories = []
        
        start_time = time.time()
        file_pbar = tqdm(self.npz_files, desc="Datasets", unit="file")
        
        for file_idx, npz_file in enumerate(file_pbar):
            file_pbar.set_description(f"{Path(npz_file).name[:30]}")
            
            data = np.load(npz_file)
            displacement = data['disp2d']
            force = data['force_mag']
            coordinates = data['projected_pixels'][0]
            
            displacement = displacement[:, self.valid_nodes_mask]
            force = force[:, self.valid_nodes_mask]
            coordinates = coordinates[self.valid_nodes_mask]
            
            n_timesteps = min(len(displacement), self.max_timesteps)
            if len(displacement) > self.max_timesteps:
                indices = np.linspace(0, len(displacement)-1, self.max_timesteps, dtype=int)
                displacement = displacement[indices]
                force = force[indices]
            
            x_coords = coordinates[:, 0]
            y_coords = coordinates[:, 1]
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            
            grid_x = ((x_coords - x_min) / (x_max - x_min + 1e-8) * (self.grid_width - 1)).astype(int)
            grid_y = ((y_coords - y_min) / (y_max - y_min + 1e-8) * (self.grid_height - 1)).astype(int)
            
            grid_x = np.clip(grid_x, 0, self.grid_width - 1)
            grid_y = np.clip(grid_y, 0, self.grid_height - 1)
            
            for t in range(n_timesteps):
                input_grid, target_grid = self._create_stable_grid(
                    displacement[t], force[t], grid_x, grid_y
                )
                
                self.all_input_grids.append(input_grid)
                self.all_target_grids.append(target_grid)
                
                max_force = force[t].max()
                if max_force > self.force_p99:
                    self.all_force_categories.append(2)
                elif max_force > self.force_p90:
                    self.all_force_categories.append(1)
                else:
                    self.all_force_categories.append(0)
            
            elapsed = time.time() - start_time
            eta = elapsed / (file_idx + 1) * (len(self.npz_files) - file_idx - 1)
            file_pbar.set_postfix({
                'Grids': len(self.all_input_grids),
                'ETA': f'{int(eta/60)}min'
            })
        
        total_time = time.time() - start_time
        logger.info(f"{len(self.all_input_grids)} grids created in {total_time/60:.1f} minutes")
        logger.info(f"Distribution: Normal={self.all_force_categories.count(0)}, "
                   f"High={self.all_force_categories.count(1)}, "
                   f"Very high={self.all_force_categories.count(2)}")
    
    def _create_stable_grid(self, disp, force, grid_x, grid_y):
        """Create input/target grids with stable normalization and clipping."""
        
        input_grid = np.zeros((5, self.grid_height, self.grid_width), dtype=np.float32)
        target_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        
        dx, dy = disp[:, 0], disp[:, 1]
        
        dx_norm = dx / self.dx_scale
        dy_norm = dy / self.dy_scale
        magnitude = np.sqrt(dx**2 + dy**2) / self.dx_scale
        magnitude = np.clip(magnitude, 0, 5)
        
        input_grid[0, grid_y, grid_x] = dx_norm
        input_grid[1, grid_y, grid_x] = dy_norm
        input_grid[2, grid_y, grid_x] = magnitude
        input_grid[3] = self.brain_mask.astype(np.float32)
        
        dx_grid = input_grid[0].copy()
        dy_grid = input_grid[1].copy()
        
        dx_smooth = scipy.ndimage.gaussian_filter(dx_grid, sigma=1.0)
        dy_smooth = scipy.ndimage.gaussian_filter(dy_grid, sigma=1.0)
        
        div_x = np.gradient(dx_smooth, axis=1)
        div_y = np.gradient(dy_smooth, axis=0)
        divergence = div_x + div_y
        
        input_grid[4] = divergence * self.brain_mask
        
        force_norm = force / self.force_scale
        force_norm = np.clip(force_norm, 0, 2.0)
        
        target_grid[grid_y, grid_x] = force_norm
        target_grid[~self.brain_mask] = -1
        
        return torch.tensor(input_grid, dtype=torch.float32), \
               torch.tensor(target_grid, dtype=torch.float32)
    
    def __len__(self):
        return len(self.all_input_grids)
    
    def __getitem__(self, idx):
        return self.all_input_grids[idx], self.all_target_grids[idx]


class UltraStableUNet(nn.Module):
    """Ultra-stable U-Net architecture with minimal complexity."""
    
    def __init__(self, in_channels=5):
        super().__init__()
        
        self.enc1 = self._conv_block(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = self._conv_block(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = self._conv_block(32, 64)
        self.pool3 = nn.MaxPool2d(2)
        
        self.bottleneck = self._conv_block(64, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = self._conv_block(128, 64)
        
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = self._conv_block(64, 32)
        
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = self._conv_block(32, 16)
        
        self.output = nn.Conv2d(16, 1, kernel_size=1)
        
        self._initialize_weights()
    
    def _conv_block(self, in_ch, out_ch):
        """Simple and stable convolutional block."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.05)
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
        
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        b = self.bottleneck(self.pool3(e3))
        
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        output = self.output(d1)
        output = torch.clamp(output, min=0, max=3)
        output = output * mask
        
        return output.squeeze(1)


class SimpleMSELoss(nn.Module):
    """Simple MSE loss with mild weighting for high forces."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        valid_mask = target >= 0
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred.device)
        
        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]
        mse = (pred_valid - target_valid) ** 2
        
        if len(target_valid) > 10:
            threshold = torch.quantile(target_valid, 0.9)
            high_mask = target_valid > threshold
            
            weights = torch.ones_like(target_valid)
            weights[high_mask] = 2.0
            
            weighted_mse = mse * weights
            return weighted_mse.mean()
        else:
            return mse.mean()


def train_stable_model(npz_files, max_timesteps=1000, epochs=40, batch_size=8, lr=0.001):
    """Train ultra-stable model with conservative settings."""
    
    logger = logging.getLogger(__name__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info("="*70)
    logger.info("ULTRA-STABLE MODEL TRAINING")
    logger.info("="*70)
    logger.info(f"Device: {device}")
    logger.info(f"Stability features:")
    logger.info(f"   - P99.9 displacement normalization")
    logger.info(f"   - Force clipping: [0, 2.0]")
    logger.info(f"   - Output clamp: [0, 3]")
    logger.info(f"   - Simple MSE with 2x weighting")
    logger.info(f"   - Minimal dropout: 0.05")
    
    dataset = StableSpatialDataset(
        npz_files=npz_files,
        max_timesteps=max_timesteps,
        grid_size=(256, 256),
        exclude_pattern='6842'
    )
    
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == 'cuda')
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == 'cuda')
    )
    
    logger.info(f"Split: {train_size} train, {val_size} validation")
    
    model = UltraStableUNet(in_channels=5).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {total_params:,} parameters")
    
    criterion = SimpleMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=7, min_lr=1e-6
    )
    
    best_val_mae = float('inf')
    patience = 12
    patience_counter = 0
    
    logger.info("Starting training...")
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            train_losses.append(loss.item())
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_train_loss = np.mean(train_losses)
        
        model.eval()
        val_losses = []
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for inputs, targets in pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_losses.append(loss.item())
                
                valid_mask = targets >= 0
                if valid_mask.any():
                    targets_valid = targets[valid_mask] * dataset.force_scale
                    outputs_valid = outputs[valid_mask] * dataset.force_scale
                    
                    all_preds.extend(outputs_valid.cpu().numpy())
                    all_targets.extend(targets_valid.cpu().numpy())
        
        avg_val_loss = np.mean(val_losses)
        avg_val_mae = mean_absolute_error(all_targets, all_preds) if all_targets else 0
        val_r2 = r2_score(all_targets, all_preds) if all_targets else 0
        
        scheduler.step(avg_val_loss)
        
        logger.info(f"Epoch {epoch+1}/{epochs}:")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}")
        logger.info(f"  Val Loss: {avg_val_loss:.4f}")
        logger.info(f"  Val MAE: {avg_val_mae:.4f}")
        logger.info(f"  Val R²: {val_r2:.4f}")
        
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            patience_counter = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_mae': avg_val_mae,
                'val_r2': val_r2,
                'dataset_stats': {
                    'dx_scale': dataset.dx_scale,
                    'dy_scale': dataset.dy_scale,
                    'force_scale': dataset.force_scale,
                    'force_p90': dataset.force_p90,
                    'brain_mask': dataset.brain_mask
                }
            }, 'best_model_stable.pth')
            
            logger.info(f"Best model saved! MAE: {avg_val_mae:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break
    
    logger.info("="*70)
    logger.info(f"Training completed! Best MAE: {best_val_mae:.4f}")
    
    return best_val_mae


def main():
    parser = argparse.ArgumentParser(description='Ultra-Stable Spatial CNN Model')
    parser.add_argument('--datasets_dir', default='datasets_2d', help='Datasets directory')
    parser.add_argument('--max_timesteps', type=int, default=1000, help='Max timesteps per file')
    parser.add_argument('--epochs', type=int, default=40, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    
    datasets_path = Path(args.datasets_dir)
    npz_files = list(datasets_path.glob('**/*_2d.npz'))
    
    if not npz_files:
        logger.error(f"No files found in {datasets_path}")
        return
    
    logger.info(f"{len(npz_files)} files found")
    
    best_mae = train_stable_model(
        npz_files=npz_files,
        max_timesteps=args.max_timesteps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
    
    logger.info(f"Training completed! Best MAE: {best_mae:.4f}")


if __name__ == '__main__':
    main()