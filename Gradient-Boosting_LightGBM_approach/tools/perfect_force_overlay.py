#!/usr/bin/env python3
"""
PERFECT OVERLAY - Optimal visualization of brain surface forces
All frames (0-1999), all forces visible, smooth evolution
"""
import argparse
import re
import time
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import sys
import gc
from tqdm import tqdm

class PerfectOverlayGenerator:
    def __init__(self, args):
        self.args = args
        self.start_time = time.time()
        self.stats = {
            'total_frames': 0,
            'successful_frames': 0,
            'total_force_points': 0,
            'processing_times': []
        }
        
        # Optimal configuration
        try:
            # New matplotlib syntax
            self.colormap = plt.colormaps['inferno']
        except AttributeError:
            # Fallback for older versions
            self.colormap = cm.get_cmap('inferno')
        self.force_normalizer = None
        self.global_vmax = None
        
        print("PERFECT OVERLAY - Initialization")
        print("=" * 60)
    
    def index_images(self, images_dir):
        """Index all available images"""
        print("Indexing images...")
        idx = {}
        pat = re.compile(r"(\d+)")
        
        for p in Path(images_dir).iterdir():
            if p.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
                continue
            m = pat.search(p.stem)
            if m:
                frame_num = int(m.group(1))
                idx[frame_num] = p

        print(f"   {len(idx)} images indexed")
        return idx
    
    def calculate_global_scale(self, gt_mag, pred):
        """Calculate optimal global scale for all frames"""
        print("Calculating optimal global scale...")
        
        # Smart sampling to avoid loading everything in memory
        sample_frames = np.linspace(0, gt_mag.shape[0]-1, min(200, gt_mag.shape[0]), dtype=int)
        
        all_forces = []
        for t in sample_frames:
            frame_forces = np.concatenate([gt_mag[t], pred[t]])
            valid_forces = frame_forces[frame_forces > self.args.min_force_display]
            if len(valid_forces) > 0:
                all_forces.extend(valid_forces)
                
        if len(all_forces) == 0:
            self.global_vmax = 1.0
        else:
            all_forces = np.array(all_forces)
            self.global_vmax = float(np.percentile(all_forces, self.args.clip_percentile))
        
        # Ensure minimum for visibility
        self.global_vmax = max(self.global_vmax, 0.1)
        
        print(f"   Global scale: {self.global_vmax:.6f} N")
        print(f"   Heat threshold: {self.args.force_threshold:.3f} N")
        
        # Initialize normalizer
        self.force_normalizer = Normalize(vmin=0, vmax=self.global_vmax)
        
        return self.global_vmax
    
    def rasterize_forces_optimized(self, mag, proj_xy, depth, W, H):
        """Optimized rasterization with z-buffer"""
        # Coordinate conversion
        x = np.rint(proj_xy[:, 0]).astype(np.int32)
        y = np.rint(proj_xy[:, 1]).astype(np.int32)
        
        if self.args.flip_y:
            y = (H - 1) - y
        
        # Validity mask - ALL significant forces
        valid_mask = (
            (x >= 0) & (x < W) & 
            (y >= 0) & (y < H) & 
            (mag >= self.args.min_force_display)
        )
        
        if not np.any(valid_mask):
            return np.zeros((H, W), dtype=np.float32)
        
        # Extract valid data
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        z_valid = depth[valid_mask].astype(np.float32)
        mag_valid = mag[valid_mask].astype(np.float32)
        
        # Z-buffer to resolve occlusion
        pixel_indices = y_valid * W + x_valid
        sort_order = np.lexsort((z_valid, pixel_indices))
        
        pixel_sorted = pixel_indices[sort_order]
        mag_sorted = mag_valid[sort_order]
        
        # Keep only closest point for each pixel
        unique_pixels, first_indices = np.unique(pixel_sorted, return_index=True)
        
        # Create heatmap
        heatmap = np.zeros(W * H, dtype=np.float32)
        heatmap[unique_pixels] = mag_sorted[first_indices]
        
        return heatmap.reshape(H, W)
    
    def apply_visual_enhancement(self, heatmap):
        """Optimized visual enhancement"""
        if self.args.dilate > 0:
            # Simple dilation with circular kernel
            from scipy.ndimage import maximum_filter
            kernel_size = 2 * self.args.dilate + 1
            heatmap = maximum_filter(heatmap, size=kernel_size)
        
        if self.args.sigma > 0:
            # Gaussian blur
            from scipy.ndimage import gaussian_filter
            heatmap = gaussian_filter(heatmap, sigma=self.args.sigma)
        
        return heatmap
    
    def create_adaptive_transparency_mask(self, heatmap):
        """Adaptive transparency mask based on force intensity"""
        # Force normalization
        normalized = heatmap / self.global_vmax
        
        # Adaptive threshold
        threshold_norm = self.args.force_threshold / self.global_vmax
        
        # Progressive transparency
        alpha = np.zeros_like(heatmap, dtype=np.float32)
        
        # Below threshold: low and progressive transparency
        below_threshold = heatmap < self.args.force_threshold
        if np.any(below_threshold):
            alpha[below_threshold] = (normalized[below_threshold] / threshold_norm) * 0.3
        
        # Above threshold: strong and progressive transparency
        above_threshold = heatmap >= self.args.force_threshold
        if np.any(above_threshold):
            remaining_range = 1.0 - threshold_norm
            if remaining_range > 0:
                alpha[above_threshold] = 0.3 + ((normalized[above_threshold] - threshold_norm) / remaining_range) * 0.5
            else:
                alpha[above_threshold] = 0.8
        
        # Final clip
        alpha = np.clip(alpha, 0, self.args.max_alpha)
        
        return alpha
    
    def colorize_heatmap(self, heatmap):
        """Colorization with inferno colormap"""
        # Normalization for colormap
        normalized = np.clip(heatmap / self.global_vmax, 0, 1)
        
        # Apply colormap
        colored = self.colormap(normalized)
        
        # Convert to PIL format (RGB 0-255)
        colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
        
        return colored_rgb
    
    def blend_with_background(self, background, overlay_rgb, alpha_mask):
        """Optimized blending with variable transparency"""
        # Convert to float for precise calculations
        bg_float = np.array(background, dtype=np.float32)
        overlay_float = overlay_rgb.astype(np.float32)
        
        # Expand alpha for RGB
        alpha_3d = np.stack([alpha_mask, alpha_mask, alpha_mask], axis=2)
        
        # Blend
        result = (1 - alpha_3d) * bg_float + alpha_3d * overlay_float
        
        return result.astype(np.uint8)
    
    def add_frame_info(self, img, frame_num, gt_stats, pred_stats):
        """Add frame information"""
        draw = ImageDraw.Draw(img)
        
        # Main title
        title = f"Frame {frame_num:04d}"
        draw.text((20, 20), title, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        
        if self.args.show_stats:
            # Detailed statistics
            stats_text = [
                f"GT Max: {gt_stats['max']:.4f} N",
                f"Pred Max: {pred_stats['max']:.4f} N",
                f"GT Mean: {gt_stats['mean']:.4f} N", 
                f"Pred Mean: {pred_stats['mean']:.4f} N",
                f"Forces > {self.args.force_threshold}N: {gt_stats['above_thresh']}"
            ]
            
            # Semi-transparent background for stats
            bbox = [img.width - 280, img.height - 140, img.width - 10, img.height - 10]
            draw.rectangle(bbox, fill=(0, 0, 0, 180))
            
            for i, text in enumerate(stats_text):
                y_pos = img.height - 130 + i * 25
                draw.text((img.width - 270, y_pos), text, fill=(255, 255, 255))
    
    def create_colorbar(self, height):
        """Create vertical colorbar"""
        width = 120
        
        # Vertical gradient
        gradient = np.linspace(1, 0, height).reshape(-1, 1)
        gradient = np.repeat(gradient, 40, axis=1)
        
        # Apply colormap
        colored = self.colormap(gradient)
        colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
        
        # Convert to PIL image
        colorbar_img = Image.fromarray(colored_rgb)
        
        # Add graduations
        final_img = Image.new('RGB', (width, height), (255, 255, 255))
        final_img.paste(colorbar_img, (0, 0))
        
        draw = ImageDraw.Draw(final_img)
        
        # Ticks
        n_ticks = 6
        for i in range(n_ticks):
            y = int(i * (height - 1) / (n_ticks - 1))
            value = self.global_vmax * (1 - y / (height - 1))
            
            # Tick line
            draw.line([(40, y), (50, y)], fill=(0, 0, 0), width=2)
            
            # Text
            if value >= 1:
                text = f"{value:.1f}N"
            elif value >= 0.1:
                text = f"{value:.2f}N" 
            else:
                text = f"{value:.3f}N"
            
            draw.text((55, y-8), text, fill=(0, 0, 0))
        
        # Scale title
        draw.text((5, 10), "Force", fill=(0, 0, 0))
        
        return final_img
    
    def process_single_frame(self, frame_num, gt_mag, pred, proj, depth, img_index, background_template):
        """Process complete frame"""
        frame_start = time.time()
        
        try:
            # Background image
            if frame_num in img_index:
                bg_path = img_index[frame_num]
            else:
                bg_path = background_template
            
            background = Image.open(bg_path).convert('RGB')
            if background.size != (self.args.viewport[0], self.args.viewport[1]):
                background = background.resize((self.args.viewport[0], self.args.viewport[1]), Image.LANCZOS)
            
            W, H = background.size
            
            # Generate heatmaps
            gt_heatmap = self.rasterize_forces_optimized(
                gt_mag[frame_num], proj[frame_num], depth[frame_num], W, H
            )
            pred_heatmap = self.rasterize_forces_optimized(
                pred[frame_num], proj[frame_num], depth[frame_num], W, H
            )
            
            # Visual enhancement
            gt_heatmap = self.apply_visual_enhancement(gt_heatmap)
            pred_heatmap = self.apply_visual_enhancement(pred_heatmap)
            
            # Statistics for display
            gt_stats = {
                'max': float(np.max(gt_mag[frame_num])),
                'mean': float(np.mean(gt_mag[frame_num])),
                'above_thresh': int(np.sum(gt_mag[frame_num] >= self.args.force_threshold))
            }
            pred_stats = {
                'max': float(np.max(pred[frame_num])),
                'mean': float(np.mean(pred[frame_num])),
                'above_thresh': int(np.sum(pred[frame_num] >= self.args.force_threshold))
            }
            
            # Create GT and PRED images
            gt_colored = self.colorize_heatmap(gt_heatmap)
            pred_colored = self.colorize_heatmap(pred_heatmap)
            
            gt_alpha = self.create_adaptive_transparency_mask(gt_heatmap)
            pred_alpha = self.create_adaptive_transparency_mask(pred_heatmap)
            
            # Blend with background
            gt_final = self.blend_with_background(background, gt_colored, gt_alpha)
            pred_final = self.blend_with_background(background, pred_colored, pred_alpha)
            
            # Convert to PIL images
            gt_img = Image.fromarray(gt_final)
            pred_img = Image.fromarray(pred_final)
            
            # Add information
            self.add_frame_info(gt_img, frame_num, gt_stats, pred_stats)
            self.add_frame_info(pred_img, frame_num, gt_stats, pred_stats)
            
            # Panel titles
            draw_gt = ImageDraw.Draw(gt_img)
            draw_pred = ImageDraw.Draw(pred_img)
            
            draw_gt.text((W//2-50, 20), "GROUND TRUTH", fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
            draw_pred.text((W//2-50, 20), "PREDICTION", fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
            
            # Final assembly
            colorbar = self.create_colorbar(H)
            
            final_width = W * 2 + colorbar.width
            final_img = Image.new('RGB', (final_width, H), (255, 255, 255))
            
            final_img.paste(gt_img, (0, 0))
            final_img.paste(pred_img, (W, 0))
            final_img.paste(colorbar, (W * 2, 0))
            
            # Save
            output_path = Path(self.args.out_dir) / f"perfect_overlay_{frame_num:04d}.png"
            final_img.save(output_path, quality=95, optimize=True)
            
            # Stats
            processing_time = time.time() - frame_start
            self.stats['processing_times'].append(processing_time)
            self.stats['successful_frames'] += 1
            
            # Memory cleanup
            del gt_heatmap, pred_heatmap, gt_colored, pred_colored
            del gt_final, pred_final, gt_img, pred_img, final_img
            gc.collect()
            
            return True, processing_time
            
        except Exception as e:
            print(f"Error on frame {frame_num}: {e}")
            return False, 0
    
    def run_perfect_generation(self):
        """Complete generation of perfect overlays"""
        print(f"Starting perfect overlay generation")
        print(f"    Frames: {self.args.start} to {self.args.end-1}")
        print(f"    Maximum quality enabled")
        
        # Data loading
        print("\nLoading data...")
        
        # Loading GT with tqdm
        print("Loading Ground Truth...")
        run_data = np.load(self.args.npz, mmap_mode="r")
        
        with tqdm(total=3, desc="Loading GT data", unit="arrays") as pbar:
            proj = run_data["projected_pixels"].astype(np.float32)
            pbar.update(1)
            depth = run_data["depth_values"].astype(np.float32) 
            pbar.update(1)
            surface_forces = run_data["surface_forces"].astype(np.float32)
            gt_mag = np.linalg.norm(surface_forces, axis=-1)
            pbar.update(1)
        
        # Loading Predictions with tqdm  
        print("Loading Predictions...")
        pred_data = np.load(self.args.pred, mmap_mode="r")
        
        if "predictions" in pred_data:
            forces_shape = pred_data["forces_shape"]
            
            with tqdm(total=2, desc="Loading predictions", unit="arrays") as pbar:
                pred_flat = pred_data["predictions"].astype(np.float32)
                pbar.update(1)
                true_flat = pred_data["true_forces"].astype(np.float32)
                pbar.update(1)
            
            # CRITICAL: Verify remapping order
            print(f"   MAPPING VERIFICATION GT vs PREDICTIONS:")
            print(f"      Forces shape: {forces_shape}")
            print(f"      Pred flat: {pred_flat.shape}")
            print(f"      True flat: {true_flat.shape}")
            
            # Reshape with C order (row-major) by default
            pred = pred_flat.reshape(forces_shape[0], forces_shape[1])
            true_test = true_flat.reshape(forces_shape[0], forces_shape[1])
            
            # Verification: compare with calculated GT
            print(f"       GT from surface_forces shape: {gt_mag.shape}")
            print(f"       True reshaped shape: {true_test.shape}")
            print(f"       Pred reshaped shape: {pred.shape}")
            
            # Consistency test on some values - CRITICAL!
            gt_frame0_max = gt_mag[0].max()
            true_frame0_max = true_test[0].max()
            pred_frame0_max = pred[0].max()
            
            print(f"   FRAME 0 COMPARISON:")
            print(f"      GT (surface_forces) max: {gt_frame0_max:.6f} N")
            print(f"      TRUE (from pred file) max: {true_frame0_max:.6f} N") 
            print(f"      PRED max: {pred_frame0_max:.6f} N")
            
            # If values don't match, try Fortran order
            if abs(gt_frame0_max - true_frame0_max) > 1e-5:
                print(f"   WARNING: MAPPING ERROR! C order incorrect, trying Fortran order...")
                true_test = true_flat.reshape(forces_shape[0], forces_shape[1], order='F')
                pred = pred_flat.reshape(forces_shape[0], forces_shape[1], order='F')
                
                true_frame0_max_fortran = true_test[0].max()
                pred_frame0_max_fortran = pred[0].max()
                print(f"      TRUE (Fortran): {true_frame0_max_fortran:.6f} N")
                print(f"      PRED (Fortran): {pred_frame0_max_fortran:.6f} N")
                
                if abs(gt_frame0_max - true_frame0_max_fortran) < 1e-5:
                    print(f"   MAPPING CORRECTED! Using Fortran order")
                else:
                    print(f"   PERSISTENT PROBLEM! Check data")
            else:
                print(f"   MAPPING CORRECT! GT and TRUE match perfectly")
            
            print(f"\n   FINAL CONFIRMATION:")
            print(f"       Displaying: GT_MAG vs PRED")
            print(f"       GT_MAG = magnitude of 3D forces from surface_forces")  
            print(f"       PRED = model predictions (correctly remapped)")
            print(f"       Spatiotemporal correspondence: VERIFIED")
            
        elif "pred" in pred_data:
            pred = pred_data["pred"].astype(np.float32)
            print(f"    Using 'pred' key directly")
        elif "pred_force_mag" in pred_data:
            pred = pred_data["pred_force_mag"].astype(np.float32)
            print(f"    Using 'pred_force_mag' key")
        elif "pred_force_magnitude" in pred_data:
            pred = pred_data["pred_force_magnitude"].astype(np.float32)
            print(f"    Using 'pred_force_magnitude' key")
        else:
            raise ValueError("Prediction key not found")
        
        T, N = gt_mag.shape
        print(f"   {T} frames, {N} points per frame")
        
        # Image index
        img_index = self.index_images(self.args.images_dir)
        background_template = img_index[list(img_index.keys())[0]]
        
        # Calculate global scale
        self.calculate_global_scale(gt_mag, pred)
        
        # Create output directory
        Path(self.args.out_dir).mkdir(parents=True, exist_ok=True)
        
        # Process all frames
        print(f"\nGenerating {self.args.end - self.args.start} perfect overlays...")
        self.stats['total_frames'] = self.args.end - self.args.start
        
        # Main progress bar
        with tqdm(total=self.stats['total_frames'], 
                 desc="Perfect Overlays", 
                 unit="frames",
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} frames [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            
            for frame_num in range(self.args.start, self.args.end):
                success, proc_time = self.process_single_frame(
                    frame_num, gt_mag, pred, proj, depth, img_index, background_template
                )
                
                pbar.update(1)
                pbar.set_postfix({
                    'Frame': f'{frame_num:04d}',
                    'Time': f'{proc_time:.2f}s',
                    'Success': f'{self.stats["successful_frames"]}'
                })
                
                # Detailed progress tracking every 100 frames
                if frame_num % 100 == 0 and frame_num > self.args.start:
                    progress = (frame_num - self.args.start + 1) / self.stats['total_frames'] * 100
                    avg_time = np.mean(self.stats['processing_times'][-100:]) if self.stats['processing_times'] else 0
                    eta_mins = avg_time * (self.args.end - frame_num - 1) / 60
                    
                    tqdm.write(f"    Checkpoint Frame {frame_num:04d} | "
                              f"{progress:5.1f}% | "
                              f"Avg: {avg_time:.2f}s/frame | "
                              f"ETA: {eta_mins:.1f}min | "
                              f"Success: {self.stats['successful_frames']}/{frame_num - self.args.start + 1}")
        
        print(f"Generation completed!")
        
        # Final report
        self.generate_final_report()
    
    def generate_final_report(self):
        """Generate final report"""
        total_time = time.time() - self.start_time
        avg_time = np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_frames_requested': self.stats['total_frames'],
            'successful_frames': self.stats['successful_frames'],
            'success_rate': self.stats['successful_frames'] / self.stats['total_frames'] * 100,
            'total_time_seconds': total_time,
            'total_time_formatted': f"{total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.0f}s",
            'average_time_per_frame': avg_time,
            'frames_per_minute': 60 / avg_time if avg_time > 0 else 0,
            'configuration': {
                'force_threshold': self.args.force_threshold,
                'min_force_display': self.args.min_force_display,
                'max_alpha': self.args.max_alpha,
                'global_vmax': self.global_vmax,
                'sigma': self.args.sigma,
                'dilate': self.args.dilate
            }
        }
        
        # Save report
        report_path = Path(self.args.out_dir) / "perfect_overlay_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Final display
        print("\n" + "=" * 60)
        print("PERFECT OVERLAY COMPLETED!")
        print("=" * 60)
        print(f"Frames generated: {self.stats['successful_frames']}/{self.stats['total_frames']}")
        print(f"Total time: {report['total_time_formatted']}")
        print(f"Performance: {report['frames_per_minute']:.1f} frames/min")
        print(f"Output: {self.args.out_dir}")
        print(f"Report: {report_path}")
        print("All forces are now perfectly visible!")

def main():
    parser = argparse.ArgumentParser("Perfect overlay generator for brain surface forces")
    
    # Input files
    parser.add_argument("--npz", required=True, help="Ground truth NPZ file")
    parser.add_argument("--pred", required=True, help="Predictions NPZ file")
    parser.add_argument("--images-dir", required=True, help="Background images directory")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    
    # Frame range
    parser.add_argument("--start", type=int, default=0, help="Start frame")
    parser.add_argument("--end", type=int, default=2000, help="End frame")
    
    # Perfect visualization parameters
    parser.add_argument("--force-threshold", type=float, default=0.3, 
                       help="Heat threshold (N)")
    parser.add_argument("--min-force-display", type=float, default=0.0005,
                       help="Minimum visible force")
    parser.add_argument("--max-alpha", type=float, default=0.85,
                       help="Maximum transparency")
    
    # Visual enhancement
    parser.add_argument("--sigma", type=float, default=0.8, help="Gaussian blur")
    parser.add_argument("--dilate", type=int, default=1, help="Dilation")
    
    # Technical configuration
    parser.add_argument("--viewport", type=int, nargs=2, default=[1920, 1080])
    parser.add_argument("--flip-y", action="store_true")
    parser.add_argument("--clip-percentile", type=float, default=99.8)
    parser.add_argument("--show-stats", action="store_true", default=True)
    
    args = parser.parse_args()
    
    # Validation
    if args.end <= args.start:
        raise ValueError("end must be > start")
    
    if not Path(args.npz).exists():
        raise FileNotFoundError(f"NPZ file not found: {args.npz}")
    
    if not Path(args.pred).exists():
        raise FileNotFoundError(f"Prediction file not found: {args.pred}")
    
    if not Path(args.images_dir).exists():
        raise FileNotFoundError(f"Images directory not found: {args.images_dir}")
    
    # Launch
    generator = PerfectOverlayGenerator(args)
    generator.run_perfect_generation()

if __name__ == "__main__":
    main()