#!/usr/bin/env python3
"""
CNN Training Script with Hybrid R²-AntiGhost Loss
==================================================
3-Fold Cross-Validation training for force prediction using D2FNet architecture.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
import pickle
from sklearn.metrics import r2_score, mean_absolute_error
import argparse
import sys

sys.path.append('.')
from dataset import StableSpatialDataset
from model import D2FNet
from loss import ImprovedAdaptiveR2Loss

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train_cnn.log'),
        logging.StreamHandler()
    ]
)

def train_epoch(model, loader, optimizer, criterion, device, epoch):
    """Train for one epoch with gradient clipping."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch} [TRAIN]', ncols=100)
    
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return total_loss / n_batches

def validate_epoch(model, loader, criterion, device, brain_mask):
    """
    Validate model on CPU with preallocated arrays for memory efficiency.
    Returns validation loss, R² score, and MAE.
    """
    model.eval()
    total_loss = 0
    n_batches = 0
    
    # Preallocate arrays for all valid pixels
    n_brain_pixels = brain_mask.sum()
    n_samples = len(loader.dataset)
    max_valid_pixels = n_samples * n_brain_pixels
    
    all_preds = np.zeros(max_valid_pixels, dtype=np.float32)
    all_targets = np.zeros(max_valid_pixels, dtype=np.float32)
    pixel_count = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc='[VAL]', ncols=100):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            n_batches += 1
            
            outputs_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            
            for i in range(len(outputs_np)):
                pred = outputs_np[i]
                targ = targets_np[i]
                
                # Apply brain mask and valid target mask
                valid_mask = brain_mask & (targ >= 0)
                pred_valid = pred[valid_mask]
                targ_valid = targ[valid_mask]
                
                # Direct assignment to preallocated arrays
                n_valid = len(pred_valid)
                all_preds[pixel_count:pixel_count+n_valid] = pred_valid
                all_targets[pixel_count:pixel_count+n_valid] = targ_valid
                pixel_count += n_valid
    
    # Trim to actual size
    all_preds = all_preds[:pixel_count]
    all_targets = all_targets[:pixel_count]
    
    r2 = r2_score(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    
    return total_loss / n_batches, r2, mae

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='CNN Training with Hybrid R²-AntiGhost Loss')
    parser.add_argument('--fold', type=int, choices=[1, 2, 3], default=None,
                        help='Run specific fold only (1, 2, or 3). If not specified, runs all folds.')
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("CNN Training - 3-Fold Cross-Validation")
    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info("   - Loss: ImprovedAdaptiveR2Loss (30% R² + 70% AntiGhost)")
    logger.info("   - Ghost penalty: 3.0x on low force frames (<0.05N)")
    logger.info("   - Right border filtering: X > 1400mm excluded")
    logger.info("   - Architecture: D2FNet with Multi-Scale Attention")
    logger.info("   - Validation: CPU with preallocated arrays")
    if args.fold:
        logger.info(f"   - Running single fold: {args.fold}")
    
    # Configuration
    BASE_DIR = Path('datasets_2d_modified')
    ALL_DATASETS = [
        'run_seed_1111', 'run_seed_1200', 'run_seed_1324', 'run_seed_2004', 'run_seed_2191',
        'run_seed_2222', 'run_seed_321', 'run_seed_3333', 'run_seed_4444', 'run_seed_4509', 
        'run_seed_5555', 'run_seed_6666', 'run_seed_6842', 'run_seed_7777', 'run_seed_789',
        'run_seed_8888', 'run_seed_960', 'run_seed_9806', 'run_seed_9999'
    ]
    
    # 3-Fold CV splits (13 train / 6 val per fold)
    CV_SPLITS = [
        {
            'train': ['run_seed_1324', 'run_seed_2004', 'run_seed_2191', 'run_seed_2222', 
                     'run_seed_321', 'run_seed_3333', 'run_seed_4444', 'run_seed_4509', 
                     'run_seed_6842', 'run_seed_7777', 'run_seed_789', 'run_seed_8888', 'run_seed_9999'],
            'val': ['run_seed_1111', 'run_seed_1200', 'run_seed_5555', 'run_seed_6666', 
                   'run_seed_960', 'run_seed_9806']
        },
        {
            'train': ['run_seed_1111', 'run_seed_1200', 'run_seed_2004', 'run_seed_2191', 
                     'run_seed_2222', 'run_seed_321', 'run_seed_4444', 'run_seed_5555',
                     'run_seed_6666', 'run_seed_789', 'run_seed_960', 'run_seed_9806', 'run_seed_9999'],
            'val': ['run_seed_1324', 'run_seed_3333', 'run_seed_4509', 'run_seed_6842', 
                   'run_seed_7777', 'run_seed_8888']
        },
        {
            'train': ['run_seed_2004', 'run_seed_1200', 'run_seed_1324', 'run_seed_3333', 
                     'run_seed_4444', 'run_seed_4509', 'run_seed_5555', 'run_seed_6666',
                     'run_seed_6842', 'run_seed_7777', 'run_seed_321', 'run_seed_960', 'run_seed_9806'],
            'val': ['run_seed_1111', 'run_seed_2191', 'run_seed_2222', 'run_seed_8888', 
                   'run_seed_789', 'run_seed_9999']
        }
    ]
    
    EPOCHS = 50
    BATCH_SIZE = 8
    LR = 0.0005
    OUTPUT_DIR = Path('outputs_cnn')
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Load existing CV results if running single fold
    cv_results = []
    if args.fold and (OUTPUT_DIR / 'cv_results.pkl').exists():
        logger.info(f"\nLoading existing CV results...")
        with open(OUTPUT_DIR / 'cv_results.pkl', 'rb') as f:
            cv_results = pickle.load(f)
        logger.info(f"   Found {len(cv_results)} completed fold(s)")
    
    # Select folds to run
    folds_to_run = [(args.fold, CV_SPLITS[args.fold - 1])] if args.fold else enumerate(CV_SPLITS, 1)
    
    for fold_idx, split in folds_to_run:
        logger.info("\n" + "=" * 80)
        logger.info(f"FOLD {fold_idx}/3")
        logger.info("=" * 80)
        logger.info(f"Train datasets: {', '.join(split['train'])}")
        logger.info(f"Val datasets: {', '.join(split['val'])}")
        
        # Collect NPZ files for train and validation
        import glob
        train_files = []
        for ds in split['train']:
            pattern = str(BASE_DIR / ds / 'brain_surface_*_auto_projected_*_2d.npz')
            files = glob.glob(pattern)
            if files:
                train_files.append(files[0])
            else:
                logger.warning(f"Warning: No NPZ found for train dataset: {ds}")
        
        val_files = []
        for ds in split['val']:
            pattern = str(BASE_DIR / ds / 'brain_surface_*_auto_projected_*_2d.npz')
            files = glob.glob(pattern)
            if files:
                val_files.append(files[0])
            else:
                logger.warning(f"Warning: No NPZ found for val dataset: {ds}")
        
        # Create datasets with border filtering and normalization
        logger.info(f"\nLoading datasets for fold {fold_idx}...")
        train_dataset = StableSpatialDataset(
            npz_files=train_files,
            max_timesteps=2000,
            grid_size=(256, 256)
        )
        
        # VALIDATION: Reuse TRAINING scales for consistency!
        val_dataset = StableSpatialDataset(
            npz_files=val_files,
            max_timesteps=2000,
            grid_size=(256, 256),
            fixed_dx_scale=train_dataset.dx_scale,
            fixed_dy_scale=train_dataset.dy_scale,
            fixed_force_scale=train_dataset.force_scale
        )
        logger.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        # Initialize model for this fold
        logger.info(f"\nInitializing model for fold {fold_idx}...")
        model = D2FNet(in_channels=5).to(device)
        
        # Hybrid loss function
        criterion = ImprovedAdaptiveR2Loss(
            r2_weight=0.3,
            ghost_penalty=3.0,
            low_force_threshold=0.05
        )
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        # Training loop
        logger.info(f"\nStarting training for fold {fold_idx}...")
        best_r2 = -np.inf
        history = {'train_loss': [], 'val_loss': [], 'val_r2': [], 'val_mae': []}
        
        for epoch in range(1, EPOCHS + 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"FOLD {fold_idx} - EPOCH {epoch}/{EPOCHS}")
            logger.info(f"{'='*80}")
            
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
            val_loss, val_r2, val_mae = validate_epoch(model, val_loader, criterion, device, train_dataset.brain_mask)
            scheduler.step(val_r2)
            
            logger.info(f"\nEpoch {epoch} Results:")
            logger.info(f"   Train Loss: {train_loss:.6f}")
            logger.info(f"   Val Loss:   {val_loss:.6f}")
            logger.info(f"   Val R²:     {val_r2:.4f}")
            logger.info(f"   Val MAE:    {val_mae:.4f} N")
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_r2'].append(val_r2)
            history['val_mae'].append(val_mae)
            
            # Save best model
            if val_r2 > best_r2:
                best_r2 = val_r2
                
                checkpoint = {
                    'fold': fold_idx,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_r2': val_r2,
                    'val_mae': val_mae,
                    'val_loss': val_loss,
                    # Save scales for inference!
                    'dx_scale': train_dataset.dx_scale,
                    'dy_scale': train_dataset.dy_scale,
                    'force_scale': train_dataset.force_scale
                }
                
                torch.save(checkpoint, OUTPUT_DIR / f'best_model_fold_{fold_idx}.pth')
                logger.info(f"   New best model saved! R² = {val_r2:.4f}")
        
        # Store fold results
        cv_results.append({
            'fold': fold_idx,
            'best_r2': best_r2,
            'best_mae': min(history['val_mae']),
            'history': history
        })
        
        with open(OUTPUT_DIR / f'history_fold_{fold_idx}.pkl', 'wb') as f:
            pickle.dump(history, f)
        
        logger.info(f"\nFold {fold_idx} completed - Best R²: {best_r2:.4f}")
    
    # JUST A REORGANIZATION STEP
    # Merge with existing results if running single fold
    if args.fold and (OUTPUT_DIR / 'cv_results.pkl').exists():
        logger.info(f"\nUpdating existing CV results...")
        with open(OUTPUT_DIR / 'cv_results.pkl', 'rb') as f:
            existing_results = pickle.load(f)
        existing_results = [r for r in existing_results if r['fold'] != args.fold]
        # Add new result
        cv_results = existing_results + cv_results
        logger.info(f"   Updated results (total: {len(cv_results)} folds)")
    
    # Final CV results summary
    logger.info("\n" + "=" * 80)
    if args.fold:
        logger.info(f"FOLD {args.fold} COMPLETED")
    else:
        logger.info("3-FOLD CROSS-VALIDATION COMPLETED")
    logger.info("=" * 80)
    
    all_r2 = [res['best_r2'] for res in cv_results]
    all_mae = [res['best_mae'] for res in cv_results]
    
    logger.info(f"\nRESULTS:")
    for res in cv_results:
        logger.info(f"   Fold {res['fold']}: R² = {res['best_r2']:.4f}, MAE = {res['best_mae']:.4f} N")
    
    if not args.fold:
        logger.info(f"\nCROSS-VALIDATION AVERAGE:")
        logger.info(f"   Mean R²:  {np.mean(all_r2):.4f} ± {np.std(all_r2):.4f}")
        logger.info(f"   Mean MAE: {np.mean(all_mae):.4f} ± {np.std(all_mae):.4f} N")
    
    # Sauvegarder résultats CV
    with open(OUTPUT_DIR / 'cv_results.pkl', 'wb') as f:
        pickle.dump(cv_results, f)
    
    logger.info(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
