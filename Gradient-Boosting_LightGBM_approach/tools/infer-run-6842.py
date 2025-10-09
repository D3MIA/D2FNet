import numpy as np
import os
import sys
import time
import joblib
from pathlib import Path
import logging
from tqdm import tqdm
import gc
from datetime import datetime

# Try to import psutil, use fallback if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference_6842_safe.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def print_progress(message, level="INFO"):
    """Print progress with timestamp and formatting"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    if level == "SUCCESS":
        print(f" [{timestamp}] {message}")
    elif level == "PROGRESS":
        print(f" [{timestamp}] {message}")
    elif level == "WARNING":
        print(f"  [{timestamp}] {message}")
    else:
        print(f"  [{timestamp}] {message}")
    logger.info(message)

def get_memory_usage():
    """Get current memory usage with fallback"""
    if PSUTIL_AVAILABLE:
        try:
            process = psutil.Process()
            memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
            return memory_gb
        except:
            return 0.0
    else:
        return 0.0  # Fallback when psutil not available

def extract_ultra_advanced_features(positions: np.ndarray) -> np.ndarray:
    """Extract 32 ultra-advanced features - SAME AS TRAINING
    
    Args:
        positions: Shape (n_timesteps, n_nodes, 2)
    
    Returns:
        features: Shape (n_timesteps * n_nodes, 32)
    """
    print_progress("🧮 Starting ultra feature extraction (32 features)...", "PROGRESS")
    start_time = time.time()
    
    n_timesteps, n_nodes, n_dims = positions.shape
    print_progress(f" Data shape: {n_timesteps} timesteps × {n_nodes} nodes × {n_dims} dims")
    
    all_features = []
    
    # Process each node with progress bar
    for node_idx in tqdm(range(n_nodes), desc=" Extracting features per node"):
        if node_idx % 5000 == 0:
            print_progress(f"Processing node {node_idx}/{n_nodes} ({node_idx/n_nodes*100:.1f}%)", "PROGRESS")
        
        pos = positions[:, node_idx, :]  # Shape: (n_timesteps, 2)
        
        # Basic displacement features
        dx = pos[:, 0] - pos[0, 0]
        dy = pos[:, 1] - pos[0, 1]
        disp_mag = np.sqrt(dx**2 + dy**2)
        disp_angle = np.arctan2(dy, dx)
        
        # Time-based features
        time_features = np.linspace(0, 1, n_timesteps)
        time_squared = time_features**2
        time_cubed = time_features**3
        
        # Velocity features (1st derivative)
        vel_x = np.zeros_like(dx)
        vel_y = np.zeros_like(dy)
        vel_x[1:] = np.diff(pos[:, 0])
        vel_y[1:] = np.diff(pos[:, 1])
        vel_mag = np.sqrt(vel_x**2 + vel_y**2)
        vel_angle = np.arctan2(vel_y, vel_x)
        
        # Acceleration features (2nd derivative)
        acc_x = np.zeros_like(vel_x)
        acc_y = np.zeros_like(vel_y)
        acc_x[1:] = np.diff(vel_x)
        acc_y[1:] = np.diff(vel_y)
        acc_mag = np.sqrt(acc_x**2 + acc_y**2)
        
        # Jerk features (3rd derivative)
        jerk_x = np.zeros_like(acc_x)
        jerk_y = np.zeros_like(acc_y)
        jerk_x[1:] = np.diff(acc_x)
        jerk_y[1:] = np.diff(acc_y)
        jerk_mag = np.sqrt(jerk_x**2 + jerk_y**2)
        
        # Advanced geometric features
        distance_from_origin = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
        angle_from_x_axis = np.arctan2(pos[:, 1], pos[:, 0])
        
        # Curvature and path features
        curvature = np.zeros_like(vel_mag)
        curvature[2:] = np.abs(vel_x[2:] * acc_y[2:] - vel_y[2:] * acc_x[2:]) / (vel_mag[2:]**3 + 1e-8)
        
        # Energy-like features
        kinetic_energy_proxy = 0.5 * vel_mag**2
        potential_energy_proxy = distance_from_origin
        
        # Directional change features
        vel_angle_change = np.zeros_like(vel_angle)
        vel_angle_change[1:] = np.diff(vel_angle)
        
        # Statistical features over local windows
        window_size = min(10, n_timesteps // 4)
        if window_size > 1:
            disp_mag_std = np.zeros_like(disp_mag)
            vel_mag_mean = np.zeros_like(vel_mag)
            
            for i in range(n_timesteps):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(n_timesteps, i + window_size // 2 + 1)
                
                disp_mag_std[i] = np.std(disp_mag[start_idx:end_idx])
                vel_mag_mean[i] = np.mean(vel_mag[start_idx:end_idx])
        else:
            disp_mag_std = np.zeros_like(disp_mag)
            vel_mag_mean = vel_mag.copy()
        
        # Create 32-feature matrix: (n_timesteps, 32)
        node_features = np.column_stack([
            # Original 16 features
            dx,                           # 0: Displacement X
            dy,                           # 1: Displacement Y  
            disp_mag,                     # 2: Displacement magnitude
            disp_angle,                   # 3: Displacement angle
            np.sin(disp_angle),           # 4: Sin of angle
            np.cos(disp_angle),           # 5: Cos of angle
            dx**2,                        # 6: Squared displacement X
            dy**2,                        # 7: Squared displacement Y
            time_features,                # 8: Time progression
            vel_x,                        # 9: Velocity X
            vel_y,                        # 10: Velocity Y
            vel_mag,                      # 11: Velocity magnitude
            acc_x,                        # 12: Acceleration X
            acc_y,                        # 13: Acceleration Y
            acc_mag,                      # 14: Acceleration magnitude
            disp_mag**2,                  # 15: Squared displacement magnitude
            
            # Advanced features (16 more)
            time_squared,                 # 16: Time squared
            time_cubed,                   # 17: Time cubed
            vel_angle,                    # 18: Velocity angle
            vel_angle_change,             # 19: Velocity direction change
            jerk_x,                       # 20: Jerk X
            jerk_y,                       # 21: Jerk Y
            jerk_mag,                     # 22: Jerk magnitude
            curvature,                    # 23: Path curvature
            distance_from_origin,         # 24: Distance from origin
            angle_from_x_axis,            # 25: Angle from X-axis
            kinetic_energy_proxy,         # 26: Kinetic energy proxy
            potential_energy_proxy,       # 27: Potential energy proxy
            disp_mag_std,                 # 28: Local displacement variability
            vel_mag_mean,                 # 29: Local velocity mean
            np.sin(angle_from_x_axis),    # 30: Sin of position angle
            np.cos(angle_from_x_axis),    # 31: Cos of position angle
        ])  # Shape: (n_timesteps, 32)
        
        all_features.append(node_features)
    
    # Stack all nodes: (n_nodes, n_timesteps, 32)
    features_3d = np.stack(all_features, axis=0)
    
    # Flatten to (n_nodes * n_timesteps, 32) - SAME ORDER as targets!
    features_2d = features_3d.reshape(-1, 32)
    
    total_duration = time.time() - start_time
    print_progress(f" Ultra feature extraction completed!", "SUCCESS")
    print_progress(f" Final feature shape: {features_2d.shape}")
    print_progress(f"⏱️  Total time: {total_duration:.2f}s")
    
    return features_2d

def load_run_6842_data():
    """Load run 6842 data"""
    print_progress("🚀 LOADING RUN 6842 DATA", "PROGRESS")
    start_time = time.time()
    
    # Load data
    data_path = '/home/chakibh/scratch/brainforces2/datasets_2d/run_seed_6842/run_seed_6842/brain_surface_7bef69ed_auto_projected_7ed6d666_2d.npz'
    print_progress(f" Loading data from: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    data = np.load(data_path)
    available_keys = list(data.keys())
    print_progress(f" Available keys: {available_keys}")
    
    # Extract data
    positions = data['disp2d']              # Shape: (n_timesteps, n_nodes, 2)
    forces = data['force_mag']              # Shape: (n_timesteps, n_nodes)
    
    print_progress(f" Data shapes:")
    print_progress(f"  - Positions: {positions.shape}")
    print_progress(f"  - Forces: {forces.shape}")
    
    load_duration = time.time() - start_time
    print_progress(f" Data loading completed in {load_duration:.2f}s", "SUCCESS")
    
    return positions, forces

def main():
    """Main inference pipeline - SAVES FIRST, then attempts visualization"""
    print_progress("🚀 STARTING SAFE INFERENCE ON RUN 6842", "SUCCESS")
    print_progress("=" * 70)
    
    overall_start = time.time()
    
    # Setup paths
    model_dir = '/home/chakibh/scratch/brainforces2/outputs_ultra_lightgbm_nocv'
    output_dir = Path('/home/chakibh/scratch/brainforces2/inference_6842_safe')
    output_dir.mkdir(exist_ok=True)

    print_progress(f" Model directory: {model_dir}")
    print_progress(f" Output directory: {output_dir}")
    print_progress(f" Initial memory usage: {get_memory_usage():.2f} GB")

    try:
        # Stage 1: Load data
        print_progress("\n" + "="*50)
        print_progress(" STAGE 1/5: LOADING DATA")
        print_progress("="*50)
        positions, forces = load_run_6842_data()
        
        # Stage 2: Extract features
        print_progress("\n" + "="*50)
        print_progress(" STAGE 2/5: ULTRA FEATURE EXTRACTION")
        print_progress("="*50)
        X = extract_ultra_advanced_features(positions)
        
        print_progress(f" Features extracted: {X.shape}")
        print_progress(f" Memory after features: {get_memory_usage():.2f} GB")
        
        # Stage 3: Load model and preprocessing
        print_progress("\n" + "="*50)
        print_progress(" STAGE 3/5: LOADING MODELS")
        print_progress("="*50)
        
        # Load best model (ultra deep)
        model_path = os.path.join(model_dir, 'lgb_ultra_deep_nocv_ultra.joblib')
        scaler_path = os.path.join(model_dir, 'ultra_scaler_nocv.joblib')
        selector_path = os.path.join(model_dir, 'ultra_feature_selector_nocv.joblib')
        
        print_progress(f" Loading model from: {model_path}")
        model = joblib.load(model_path)
        print_progress(" Ultra deep model loaded successfully!")
        
        print_progress(f" Loading scaler from: {scaler_path}")
        scaler = joblib.load(scaler_path)
        print_progress(" Scaler loaded successfully!")
        
        print_progress(f" Loading feature selector from: {selector_path}")
        selector = joblib.load(selector_path)
        print_progress(" Feature selector loaded successfully!")
        
        # Stage 4: Preprocessing & Predictions
        print_progress("\n" + "="*50)
        print_progress(" STAGE 4/5: PREPROCESSING & PREDICTIONS")
        print_progress("="*50)
        
        # Scale features
        print_progress(" Applying feature scaling...")
        X_scaled = scaler.transform(X)
        print_progress(f" Feature scaling completed")
        
        # Apply feature selection (32 -> 28 features)
        print_progress(" Applying feature selection...")
        X_selected = selector.transform(X_scaled)
        print_progress(f" Features after selection: {X_selected.shape}")
        
        # Clean up memory
        del X, X_scaled
        gc.collect()
        
        # Make predictions
        n_samples = X_selected.shape[0]
        print_progress(f"🔮 Making predictions on {n_samples:,} samples...")
        
        # Batch prediction
        batch_size = 100000
        predictions = []
        
        for i in tqdm(range(0, n_samples, batch_size), desc="🔮 Predicting batches"):
            batch_start = i
            batch_end = min(i + batch_size, n_samples)
            
            batch_pred = model.predict(X_selected[batch_start:batch_end])
            predictions.append(batch_pred)
        
        pred_forces = np.concatenate(predictions)
        print_progress(f" Predictions completed!")
        print_progress(f" Predictions shape: {pred_forces.shape}")
        
        # Calculate overall metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        forces_flat = forces.T.flatten()  # Same order as features
        
        # Ensure both arrays have same length
        if len(forces_flat) != len(pred_forces):
            print_progress(f"  Length mismatch: true={len(forces_flat)}, pred={len(pred_forces)}", "WARNING")
            min_len = min(len(forces_flat), len(pred_forces))
            forces_flat = forces_flat[:min_len]
            pred_forces = pred_forces[:min_len]
            print_progress(f" Trimmed to {min_len} samples for comparison")
        
        rmse = np.sqrt(mean_squared_error(forces_flat, pred_forces))
        mae = mean_absolute_error(forces_flat, pred_forces)
        r2 = r2_score(forces_flat, pred_forces)
        
        # Calculate relative error
        epsilon = 1e-10
        relative_error = np.abs(pred_forces - forces_flat) / (np.abs(forces_flat) + epsilon)
        mean_relative_error = np.mean(relative_error)
        median_relative_error = np.median(relative_error)
        
        # Calculate relative error only for significant forces
        force_threshold = 0.01
        significant_mask = forces_flat > force_threshold
        if np.any(significant_mask):
            relative_error_significant = np.abs(pred_forces[significant_mask] - forces_flat[significant_mask]) / forces_flat[significant_mask]
            mean_relative_error_significant = np.mean(relative_error_significant)
            median_relative_error_significant = np.median(relative_error_significant)
        else:
            mean_relative_error_significant = float('nan')
            median_relative_error_significant = float('nan')

        print_progress(" SAFE INFERENCE RESULTS:")
        print_progress(f"  RMSE: {rmse:.6f}")
        print_progress(f"  MAE: {mae:.6f}")
        print_progress(f"  R² Score: {r2:.6f} ({r2*100:.2f}%)")
        print_progress(f"  Mean Relative Error (all): {mean_relative_error:.4f} ({mean_relative_error*100:.2f}%)")
        print_progress(f"  Median Relative Error (all): {median_relative_error:.4f} ({median_relative_error*100:.2f}%)")
        if not np.isnan(mean_relative_error_significant):
            print_progress(f"  Mean Relative Error (forces > {force_threshold}): {mean_relative_error_significant:.4f} ({mean_relative_error_significant*100:.2f}%)")
            print_progress(f"  Median Relative Error (forces > {force_threshold}): {median_relative_error_significant:.4f} ({median_relative_error_significant*100:.2f}%)")
        
        # Stage 5: SAVE RESULTS IMMEDIATELY (SAFE!)
        print_progress("\n" + "="*50)
        print_progress(" STAGE 5/5: SAVING RESULTS (SAFE MODE)")
        print_progress("="*50)
        
        results_path = output_dir / 'safe_predictions_6842.npz'
        print_progress(f" SAVING results to: {results_path}")
        
        np.savez_compressed(results_path,
                          predictions=pred_forces,
                          true_forces=forces_flat,
                          rmse=rmse,
                          mae=mae,
                          r2=r2,
                          mean_relative_error=mean_relative_error,
                          median_relative_error=median_relative_error,
                          mean_relative_error_significant=mean_relative_error_significant,
                          median_relative_error_significant=median_relative_error_significant,
                          force_threshold=force_threshold,
                          positions_shape=positions.shape,
                          forces_shape=forces.shape)
        
        print_progress(f" RESULTS SAFELY SAVED!", "SUCCESS")
        
        # Create summary text file
        summary_path = output_dir / 'inference_summary.txt'
        with open(summary_path, 'w') as f:
            f.write(f"SAFE INFERENCE RESULTS - RUN 6842\n")
            f.write(f"=" * 40 + "\n")
            f.write(f"Model: LightGBM Ultra Deep (28 features)\n")
            f.write(f"Samples: {n_samples:,}\n")
            f.write(f"RMSE: {rmse:.6f}\n")
            f.write(f"MAE: {mae:.6f}\n")
            f.write(f"R² Score: {r2:.6f} ({r2*100:.2f}%)\n")
            f.write(f"Mean Relative Error: {mean_relative_error:.4f} ({mean_relative_error*100:.2f}%)\n")
            f.write(f"Median Relative Error: {median_relative_error:.4f} ({median_relative_error*100:.2f}%)\n")
            f.write(f"Completed: {datetime.now()}\n")
        
        print_progress(f" Summary saved to: {summary_path}")
        
        # Final summary
        total_duration = time.time() - overall_start
        print_progress("\n" + "="*70)
        print_progress(" SAFE INFERENCE COMPLETED SUCCESSFULLY!", "SUCCESS")
        print_progress("="*70)
        print_progress(f" Total execution time: {total_duration:.2f}s ({total_duration/60:.1f} min)")
        print_progress(f" Final memory usage: {get_memory_usage():.2f} GB")
        print_progress(f" Results saved in: {output_dir}")
        print_progress(f" Model Performance: R² = {r2*100:.2f}% (RMSE = {rmse:.6f})")
        print_progress(f" Predictions file: {results_path}")
        print_progress(f" Summary file: {summary_path}")
        
        print_progress("\n Next steps:")
        print_progress("1. Your predictions are safely saved!")
        print_progress("2. Run overlay visualization separately if needed")
        print_progress("3. Analyze results using saved predictions")
        
    except Exception as e:
        print_progress(f" ERROR during inference: {str(e)}", "WARNING")
        logger.error(f"Safe inference failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()