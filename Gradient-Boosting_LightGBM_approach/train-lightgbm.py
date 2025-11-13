#!/usr/bin/env python3
"""Ultra-Optimized LightGBM Ensemble - 19 DATASETS with 3-FOLD CV
Target: 90%+ R² with intelligent spatial subsampling and advanced features.
Uses datasets_2d_modified (19 datasets) with proper cross-validation.
"""

import numpy as np
import json
import logging
import time
import argparse
import glob
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression

# LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    raise ImportError("LightGBM is required for this script!")

def setup_logging(output_dir: Path, log_level: str = "INFO"):
    """Setup logging configuration"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.FileHandler(output_dir / "training_lightgbm_19datasets_cv.log"),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    return logger

def regular_spatial_subsampling(n_nodes: int, target_nodes: int) -> np.ndarray:
    """Regular spatial subsampling - every Nth node for uniform coverage"""
    if n_nodes <= target_nodes:
        return np.arange(n_nodes)
    
    step = n_nodes // target_nodes
    indices = np.arange(0, n_nodes, step)[:target_nodes]
    
    return indices

def extract_ultra_advanced_features(positions: np.ndarray) -> np.ndarray:
    """Extract 32 ultra-advanced features VECTORIZED (100× faster!)"""
    n_timesteps, n_nodes, n_dims = positions.shape
    
    # ✅ VECTORIZED: Calculate for ALL nodes at once (not in loop!)
    # Shape: (n_timesteps, n_nodes)
    
    # Basic displacement (vectorized)
    dx = positions[:, :, 0] - positions[0:1, :, 0]  # (n_timesteps, n_nodes)
    dy = positions[:, :, 1] - positions[0:1, :, 1]
    disp_mag = np.sqrt(dx**2 + dy**2)
    disp_angle = np.arctan2(dy, dx)
    
    # Time features (broadcast to all nodes)
    time_features = np.linspace(0, 1, n_timesteps)[:, None]  # (n_timesteps, 1)
    time_features = np.broadcast_to(time_features, (n_timesteps, n_nodes))
    time_squared = time_features**2
    time_cubed = time_features**3
    
    # Velocity (optimized with padding)
    vel_x = np.pad(np.diff(positions[:, :, 0], axis=0), ((1, 0), (0, 0)), mode='constant')
    vel_y = np.pad(np.diff(positions[:, :, 1], axis=0), ((1, 0), (0, 0)), mode='constant')
    vel_mag = np.sqrt(vel_x**2 + vel_y**2)
    vel_angle = np.arctan2(vel_y, vel_x)
    
    # Acceleration (optimized)
    acc_x = np.pad(np.diff(vel_x, axis=0), ((1, 0), (0, 0)), mode='constant')
    acc_y = np.pad(np.diff(vel_y, axis=0), ((1, 0), (0, 0)), mode='constant')
    acc_mag = np.sqrt(acc_x**2 + acc_y**2)
    
    # Jerk (optimized)
    jerk_x = np.pad(np.diff(acc_x, axis=0), ((1, 0), (0, 0)), mode='constant')
    jerk_y = np.pad(np.diff(acc_y, axis=0), ((1, 0), (0, 0)), mode='constant')
    jerk_mag = np.sqrt(jerk_x**2 + jerk_y**2)
    
    # Geometric features (vectorized)
    distance_from_origin = np.sqrt(positions[:, :, 0]**2 + positions[:, :, 1]**2)
    angle_from_x_axis = np.arctan2(positions[:, :, 1], positions[:, :, 0])
    
    # Curvature (vectorized)
    curvature = np.zeros_like(vel_mag)
    curvature[2:] = np.abs(vel_x[2:] * acc_y[2:] - vel_y[2:] * acc_x[2:]) / (vel_mag[2:]**3 + 1e-8)
    
    # Energy proxies (vectorized)
    kinetic_energy_proxy = 0.5 * vel_mag**2
    potential_energy_proxy = distance_from_origin
    
    # Directional change (optimized)
    vel_angle_change = np.pad(np.diff(vel_angle, axis=0), ((1, 0), (0, 0)), mode='constant')
    
    # Global statistics per node (no broadcasting needed, just tile later)
    disp_mag_std = np.tile(np.std(disp_mag, axis=0, keepdims=True), (n_timesteps, 1))
    vel_mag_mean = np.tile(np.mean(vel_mag, axis=0, keepdims=True), (n_timesteps, 1))
    
    # Stack all features: (32, n_timesteps, n_nodes)
    features_3d = np.stack([
        dx, dy, disp_mag, disp_angle,
        np.sin(disp_angle), np.cos(disp_angle),
        dx**2, dy**2, time_features,
        vel_x, vel_y, vel_mag,
        acc_x, acc_y, acc_mag, disp_mag**2,
        time_squared, time_cubed, vel_angle, vel_angle_change,
        jerk_x, jerk_y, jerk_mag, curvature,
        distance_from_origin, angle_from_x_axis,
        kinetic_energy_proxy, potential_energy_proxy,
        disp_mag_std, vel_mag_mean,
        np.sin(angle_from_x_axis), np.cos(angle_from_x_axis)
    ], axis=0)  # (32, n_timesteps, n_nodes)
    
    # Reshape to (n_nodes * n_timesteps, 32) - SAME ORDER as targets!
    features_2d = features_3d.transpose(2, 1, 0).reshape(-1, 32)
    
    return features_2d

class UltraLightGBM19DatasetsCV:
    """Ultra LightGBM with 19 datasets and 3-fold CV"""
    
    def __init__(self, data_root: str, output_dir: str, max_samples_per_file: int = 1000000):
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.max_samples_per_file = max_samples_per_file
        
        self.logger = setup_logging(self.output_dir)
        
        # Find all datasets in datasets_2d_modified
        all_files = sorted(glob.glob(str(self.data_root / "**" / "*_2d.npz"), recursive=True))
        
        # ALL 19 DATASETS (no exclusion)
        self.dataset_files = all_files
        
        if not self.dataset_files:
            raise ValueError(f"No dataset files found in {data_root}")
        
        self.logger.info(f"Dataset source: {data_root}")
        self.logger.info(f"Found {len(self.dataset_files)} total dataset files")
        self.logger.info(f"Using ALL {len(self.dataset_files)} datasets for 3-fold CV")
        self.logger.info(f"Max samples per dataset: {self.max_samples_per_file:,}")
        self.logger.info(f"🔄 Spatial subsampling: 20,000 nodes (regular)")
        
        # 3-FOLD CV SPLITS (same as anti-ghost hybrid)
        self.cv_splits = [
            {
                'train': ['run_seed_1324', 'run_seed_2004', 'run_seed_2191', 'run_seed_2222', 
                         'run_seed_321', 'run_seed_3333', 'run_seed_4444', 'run_seed_4509',
                         'run_seed_6842', 'run_seed_7777', 'run_seed_789', 'run_seed_8888', 
                         'run_seed_9999'],
                'val': ['run_seed_1111', 'run_seed_1200', 'run_seed_5555', 'run_seed_6666', 
                       'run_seed_960', 'run_seed_9806']
            },
            {
                'train': ['run_seed_1111', 'run_seed_1200', 'run_seed_2004', 'run_seed_2191',
                         'run_seed_2222', 'run_seed_321', 'run_seed_4444', 'run_seed_5555',
                         'run_seed_6666', 'run_seed_789', 'run_seed_960', 'run_seed_9806',
                         'run_seed_9999'],
                'val': ['run_seed_1324', 'run_seed_3333', 'run_seed_4509', 'run_seed_6842',
                       'run_seed_7777', 'run_seed_8888']
            },
            {
                'train': ['run_seed_1111', 'run_seed_1200', 'run_seed_1324', 'run_seed_3333',
                         'run_seed_4444', 'run_seed_4509', 'run_seed_5555', 'run_seed_6666',
                         'run_seed_6842', 'run_seed_7777', 'run_seed_8888', 'run_seed_960',
                         'run_seed_9806'],
                'val': ['run_seed_2004', 'run_seed_2191', 'run_seed_2222', 'run_seed_321',
                       'run_seed_789', 'run_seed_9999']
            }
        ]
        
        # ULTRA-OPTIMIZED LIGHTGBM MODELS
        self.models = {}
        
        # High Performance model
        self.models['lgb_ultra_perf_cv'] = lgb.LGBMRegressor(
            n_estimators=5000, max_depth=20, learning_rate=0.02,
            num_leaves=200, feature_fraction=0.9, bagging_fraction=0.7,
            bagging_freq=1, reg_alpha=0.01, reg_lambda=0.01,
            min_child_samples=3, min_child_weight=0.001,
            random_state=42, n_jobs=16, verbose=-1
        )
        
        # Ultra Deep model
        self.models['lgb_ultra_deep_cv'] = lgb.LGBMRegressor(
            n_estimators=3500, max_depth=25, learning_rate=0.03,
            num_leaves=300, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=2, reg_alpha=0.02, reg_lambda=0.02,
            min_child_samples=2, min_child_weight=0.001,
            random_state=42, n_jobs=16, verbose=100
        )
        
        # Feature Rich model
        self.models['lgb_feature_rich_cv'] = lgb.LGBMRegressor(
            n_estimators=4000, max_depth=15, learning_rate=0.04,
            num_leaves=150, feature_fraction=1.0, bagging_fraction=0.9,
            bagging_freq=1, reg_alpha=0.05, reg_lambda=0.05,
            min_child_samples=5, random_state=42, n_jobs=16, verbose=100
        )

    def load_dataset_subset(self, dataset_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Load specific datasets by name with preallocated arrays"""
        # Preallocate with estimated max size
        # Worst case: 20k nodes × 1500 timesteps × n_datasets = 30M samples per dataset
        max_samples_total = min(len(dataset_names) * 30_000_000, 100_000_000)  # Cap at 100M
        n_features = 32  # Known from extract_ultra_advanced_features
        
        self.logger.info(f"💾 Preallocating arrays for max {max_samples_total:,} samples")
        X = np.empty((max_samples_total, n_features), dtype=np.float32)
        y = np.empty(max_samples_total, dtype=np.float32)
        sample_count = 0
        
        for dataset_name in tqdm(dataset_names, desc=f"Loading {len(dataset_names)} datasets"):
            # Find file matching dataset name
            matching_files = [f for f in self.dataset_files if dataset_name in f]
            
            if not matching_files:
                self.logger.warning(f"WARNING: Dataset {dataset_name} not found!")
                continue
            
            file_path = matching_files[0]
            self.logger.info(f"📂 {dataset_name}...")
            
            data = np.load(file_path)
            
            # Handle both formats
            if 'disp2d' in data.keys():
                positions = data['disp2d']
                force_mag = data['force_mag']
            else:
                positions = data['positions']
                forces = data['forces']
                force_mag = np.sqrt(forces[:, :, 0]**2 + forces[:, :, 1]**2)
            
            # Timestep subsampling
            n_timesteps = positions.shape[0]
            if n_timesteps > 1500:
                step = n_timesteps // 1500
                positions = positions[::step, :, :]
                force_mag = force_mag[::step, :]
            
            # Spatial subsampling to 20k nodes
            n_nodes = positions.shape[1]
            if n_nodes > 20000:
                indices = regular_spatial_subsampling(n_nodes, 20000)
                positions = positions[:, indices, :]
                force_mag = force_mag[:, indices]
            
            # Extract features
            features = extract_ultra_advanced_features(positions)
            targets = force_mag.T.flatten()
            
            # Subsample if too many
            if len(features) > self.max_samples_per_file:
                idx = np.random.choice(len(features), self.max_samples_per_file, replace=False)
                features = features[idx]
                targets = targets[idx]
            
            # Add to preallocated arrays
            n_samples = len(features)
            X[sample_count:sample_count+n_samples] = features
            y[sample_count:sample_count+n_samples] = targets
            sample_count += n_samples
            
            self.logger.info(f"   {n_samples:,} samples")
        
        # Trim to actual size (in case some datasets were skipped)
        X = X[:sample_count]
        y = y[:sample_count]
        
        return X, y

    def train_with_cv(self) -> Dict:
        """Train with 3-fold cross-validation"""
        self.logger.info("\n" + "="*80)
        self.logger.info("STARTING 3-FOLD CROSS-VALIDATION (19 DATASETS)")
        self.logger.info("="*80)
        
        cv_results = {}
        
        for fold_idx, split in enumerate(self.cv_splits, 1):
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"FOLD {fold_idx}/3")
            self.logger.info(f"{'='*80}")
            self.logger.info(f"🏋️  Train: {len(split['train'])} datasets - {split['train']}")
            self.logger.info(f"Val: {len(split['val'])} datasets - {split['val']}")
            
            # Load data for this fold
            self.logger.info("\nLoading training data...")
            X_train, y_train = self.load_dataset_subset(split['train'])
            
            self.logger.info("\nLoading validation data...")
            X_val, y_val = self.load_dataset_subset(split['val'])
            
            self.logger.info(f"\nFold {fold_idx} data:")
            self.logger.info(f"   Train: {len(X_train):,} samples, {X_train.shape[1]} features")
            self.logger.info(f"   Val: {len(X_val):,} samples, {X_val.shape[1]} features")
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Feature selection
            selector = SelectKBest(score_func=f_regression, k=28)
            X_train_sel = selector.fit_transform(X_train_scaled, y_train)
            X_val_sel = selector.transform(X_val_scaled)
            
            # Save preprocessing for this fold
            joblib.dump(scaler, self.output_dir / f"scaler_fold{fold_idx}.joblib")
            joblib.dump(selector, self.output_dir / f"selector_fold{fold_idx}.joblib")
            
            # Train models for this fold
            fold_results = {}
            
            for model_name, model in self.models.items():
                self.logger.info(f"\n⚡ Training {model_name} (Fold {fold_idx})...")
                
                start = time.time()
                model.fit(
                    X_train_sel, y_train,
                    eval_set=[(X_val_sel, y_val)],
                    callbacks=[lgb.early_stopping(200)],
                    eval_metric='rmse'
                )
                train_time = time.time() - start
                
                # Predictions
                y_train_pred = model.predict(X_train_sel)
                y_val_pred = model.predict(X_val_sel)
                
                # Metrics
                train_r2 = r2_score(y_train, y_train_pred)
                val_r2 = r2_score(y_val, y_val_pred)
                train_mae = mean_absolute_error(y_train, y_train_pred)
                val_mae = mean_absolute_error(y_val, y_val_pred)
                
                fold_results[model_name] = {
                    'train_r2': train_r2,
                    'val_r2': val_r2,
                    'train_mae': train_mae,
                    'val_mae': val_mae,
                    'train_time': train_time,
                    'n_trees': model.booster_.num_trees()
                }
                
                self.logger.info(f"   {model_name}: Val R²={val_r2:.4f}, MAE={val_mae:.4f} ({train_time:.1f}s)")
                
                # Save model
                model_path = self.output_dir / f"{model_name}_fold{fold_idx}.joblib"
                joblib.dump(model, model_path)
            
            cv_results[f'fold_{fold_idx}'] = fold_results
        
        # Aggregate CV results
        self.logger.info("\n" + "="*80)
        self.logger.info("CROSS-VALIDATION SUMMARY")
        self.logger.info("="*80)
        
        for model_name in self.models.keys():
            val_r2s = [cv_results[f'fold_{i}'][model_name]['val_r2'] for i in range(1, 4)]
            mean_r2 = np.mean(val_r2s)
            std_r2 = np.std(val_r2s)
            
            self.logger.info(f"{model_name}:")
            self.logger.info(f"   Val R²: {mean_r2:.4f} ± {std_r2:.4f}")
            self.logger.info(f"   Folds: {[f'{r:.4f}' for r in val_r2s]}")
            
            if mean_r2 >= 0.90:
                self.logger.info(f"   🎉 TARGET ACHIEVED: {mean_r2:.1%}!")
        
        # Save results
        with open(self.output_dir / "cv_results_19datasets.json", 'w') as f:
            json.dump(cv_results, f, indent=2, default=float)
        
        return cv_results

def main():
    parser = argparse.ArgumentParser(description="Ultra LightGBM - 19 Datasets with 3-Fold CV")
    parser.add_argument("--data_root", type=str, default="datasets_2d_modified")
    parser.add_argument("--output_dir", type=str, default="outputs_ultra_lightgbm_19datasets_cv")
    parser.add_argument("--max_samples", type=int, default=1000000)
    
    args = parser.parse_args()
    
    predictor = UltraLightGBM19DatasetsCV(
        data_root=args.data_root,
        output_dir=args.output_dir,
        max_samples_per_file=args.max_samples
    )
    
    predictor.train_with_cv()

if __name__ == "__main__":
    main()
