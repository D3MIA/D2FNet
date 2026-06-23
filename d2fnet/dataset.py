#!/usr/bin/env python3
"""
Stable Spatial Dataset
======================
Dataset class with stable normalization and preprocessing for force prediction.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from pathlib import Path
import time
from tqdm import tqdm
import scipy.ndimage
from scipy.ndimage import binary_dilation, binary_fill_holes

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
                 exclude_pattern=None,
                 fixed_dx_scale=None,
                 fixed_dy_scale=None,
                 fixed_force_scale=None):
        self.npz_files = npz_files
        self.max_timesteps = max_timesteps
        self.grid_width, self.grid_height = grid_size
        
        if exclude_pattern:
            self.npz_files = [f for f in self.npz_files if exclude_pattern not in str(f)]
        
        logger = logging.getLogger(__name__)
        logger.info(f"Files after filtering: {len(self.npz_files)}")
        
        # If fixed scales provided, use them (for validation/inference consistency)
        if fixed_dx_scale is not None:
            logger.info("Using fixed scales (train normalization):")
            self.dx_scale = fixed_dx_scale
            self.dy_scale = fixed_dy_scale
            self.force_scale = fixed_force_scale
            logger.info(f"   dx_scale: {self.dx_scale:.4f}")
            logger.info(f"   dy_scale: {self.dy_scale:.4f}")
            logger.info(f"   force_scale: {self.force_scale:.4f}")
            # Still need to calculate percentiles for force categories
            self._calculate_force_percentiles()
        else:
            # Calculate scales from data
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
    
    def _calculate_force_percentiles(self):
        """Calculate force percentiles only (when scales are fixed)."""
        logger = logging.getLogger(__name__)
        logger.info("Calculating force percentiles...")
        
        all_forces = []
        
        for npz_file in self.npz_files[:min(5, len(self.npz_files))]:
            data = np.load(npz_file)
            sample_force = data['force_mag'][:500]
            all_forces.extend(sample_force.flatten())
        
        self.force_p50 = np.percentile(all_forces, 50)
        self.force_p90 = np.percentile(all_forces, 90)
        self.force_p99 = np.percentile(all_forces, 99)
        
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

