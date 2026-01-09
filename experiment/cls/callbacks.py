"""Custom callbacks for classification training.

Includes visualization callback that triggers on best checkpoint saves.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint


def create_visualization_gif(
    vis_dir: Path,
    pattern: str = "class_comparison_epoch*.png",
    output_name: str = "class_comparison_evolution.gif",
    duration: int = 500,  # ms per frame
) -> Optional[Path]:
    """Create a GIF from a series of visualization images.
    
    Args:
        vis_dir: Directory containing visualization images
        pattern: Glob pattern to match image files
        output_name: Output GIF filename
        duration: Duration per frame in milliseconds
        
    Returns:
        Path to generated GIF or None if failed
    """
    try:
        from PIL import Image
        
        # Find all matching images
        images_paths = sorted(vis_dir.glob(pattern))
        
        if len(images_paths) < 2:
            print(f"[GIF] Not enough images to create GIF (found {len(images_paths)})")
            return None
        
        # Sort by epoch number
        def extract_epoch(path: Path) -> int:
            match = re.search(r'epoch(\d+)', path.stem)
            return int(match.group(1)) if match else 0
        
        images_paths = sorted(images_paths, key=extract_epoch)
        
        # Load images
        images = []
        for path in images_paths:
            img = Image.open(path)
            # Convert to RGB if necessary (for consistency)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(img)
        
        # Add black end frame (1 second = 1000ms, use 2 frames at 500ms each)
        if len(images) > 0:
            last_img = images[-1]
            black_frame = Image.new('RGB', last_img.size, (0, 0, 0))
            # Add 2 black frames for 1 second total pause
            images.append(black_frame)
            images.append(black_frame)
        
        # Build duration list: normal frames + longer pause for black frames
        durations = [duration] * (len(images) - 2) + [500, 500]  # 1s total for black
        
        # Save as GIF
        output_path = vis_dir / output_name
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=durations,
            loop=0  # 0 = infinite loop
        )
        
        print(f"[GIF] Created {output_path} with {len(images)} frames (incl. end marker)")
        return output_path
        
    except ImportError:
        print("[GIF] PIL not installed, skipping GIF generation")
        return None
    except Exception as e:
        print(f"[GIF] Error creating GIF: {e}")
        return None


class VisualizationCallback(Callback):
    """Callback to generate Grad-CAM visualizations on best model saves.
    
    Triggers visualization generation whenever a new best checkpoint is saved.
    """
    
    def __init__(
        self,
        output_dir: str = "visualizations",
        class_names: Optional[List[str]] = None,
        max_samples: int = 64,
        every_n_epochs: int = 1,
        frequency_domain: bool = False,
    ):
        """
        Args:
            output_dir: Directory to save visualizations
            class_names: List of class names
            max_samples: Maximum samples to use for visualization
            every_n_epochs: Generate visualizations every N epochs when best improves
            frequency_domain: If True, data is in frequency domain
        """
        super().__init__()
        self.output_dir = Path(output_dir)
        self.class_names = class_names
        self.max_samples = max_samples
        self.every_n_epochs = every_n_epochs
        self.frequency_domain = frequency_domain
        
        self._last_best_score = None
        self._test_dataloader = None
    
    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Setup output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get class names from datamodule if not provided
        if self.class_names is None:
            if hasattr(trainer.datamodule, 'get_class_names'):
                self.class_names = trainer.datamodule.get_class_names()
            else:
                self.class_names = [f"Class_{i}" for i in range(pl_module.num_classes)]
    
    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Check if best model improved and trigger visualization."""
        # Find ModelCheckpoint callback
        checkpoint_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                checkpoint_callback = callback
                break
        
        if checkpoint_callback is None:
            return
        
        # Check if best score improved
        current_best = checkpoint_callback.best_model_score
        if current_best is None:
            return
        
        current_best_val = current_best.item() if hasattr(current_best, 'item') else current_best
        
        # Check if this is a new best
        is_new_best = (
            self._last_best_score is None or 
            current_best_val != self._last_best_score
        )
        
        if is_new_best and trainer.current_epoch % self.every_n_epochs == 0:
            self._last_best_score = current_best_val
            self._generate_visualizations(trainer, pl_module)
    
    def _generate_visualizations(
        self, 
        trainer: L.Trainer, 
        pl_module: L.LightningModule
    ) -> None:
        """Generate all visualizations."""
        print(f"\n[Visualization] Generating Grad-CAM analysis (epoch {trainer.current_epoch})...")
        
        try:
            from .visualize import generate_all_visualizations
            
            # Get test dataloader
            if self._test_dataloader is None:
                if hasattr(trainer.datamodule, 'val_dataloader'):
                    self._test_dataloader = trainer.datamodule.val_dataloader()
                elif hasattr(trainer.datamodule, 'test_dataloader'):
                    trainer.datamodule.setup('test')
                    self._test_dataloader = trainer.datamodule.test_dataloader()
            
            if self._test_dataloader is None:
                print("[Visualization] No dataloader available, skipping")
                return
            
            # Generate visualizations
            device = pl_module.device
            saved_files = generate_all_visualizations(
                model=pl_module,
                dataloader=self._test_dataloader,
                class_names=self.class_names,
                output_dir=str(self.output_dir),
                device=str(device),
                max_samples=self.max_samples,
                epoch=trainer.current_epoch,
                frequency_domain=self.frequency_domain,
            )
            
            # Log to tensorboard if available
            for logger in trainer.loggers:
                if hasattr(logger, 'experiment') and hasattr(logger.experiment, 'add_image'):
                    for name, path in saved_files.items():
                        try:
                            import matplotlib.pyplot as plt
                            import matplotlib.image as mpimg
                            img = mpimg.imread(path)
                            # Convert to CHW format for tensorboard
                            if img.ndim == 3:
                                img = img.transpose(2, 0, 1)
                            logger.experiment.add_image(
                                f"gradcam/{name}", 
                                img, 
                                global_step=trainer.current_epoch
                            )
                        except Exception as e:
                            print(f"[Visualization] Failed to log {name} to tensorboard: {e}")
            
            print(f"[Visualization] Saved {len(saved_files)} figures to {self.output_dir}")
            
        except Exception as e:
            print(f"[Visualization] Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()
    
    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Generate GIF from all visualizations at end of training."""
        print("\n[Visualization] Generating evolution GIFs...")
        
        # Generate GIF for class_comparison images
        gif_path = create_visualization_gif(
            self.output_dir,
            pattern="class_comparison_epoch*.png",
            output_name="class_comparison_evolution.gif",
            duration=500,  # 500ms per frame
        )
        if gif_path:
            print(f"[Visualization] Class comparison GIF: {gif_path}")
        
        # Generate GIF for single_sample_cam images
        gif_path2 = create_visualization_gif(
            self.output_dir,
            pattern="single_sample_cam_epoch*.png",
            output_name="single_sample_cam_evolution.gif",
            duration=500,  # 500ms per frame
        )
        if gif_path2:
            print(f"[Visualization] Single sample CAM GIF: {gif_path2}")


class TestOnBestCallback(Callback):
    """Callback to log when best model improves (test runs at end of training)."""
    
    def __init__(self):
        super().__init__()
        self._last_best_score = None
    
    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Log when best model improves."""
        # Find ModelCheckpoint callback
        checkpoint_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                checkpoint_callback = callback
                break
        
        if checkpoint_callback is None:
            return
        
        current_best = checkpoint_callback.best_model_score
        if current_best is None:
            return
        
        current_best_val = current_best.item() if hasattr(current_best, 'item') else current_best
        
        is_new_best = (
            self._last_best_score is None or 
            current_best_val != self._last_best_score
        )
        
        if is_new_best:
            self._last_best_score = current_best_val
            print(f"\n[Best] New best model! score={current_best_val:.4f}, epoch={trainer.current_epoch}")
