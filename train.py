"""
YOLO Training Script with Enhanced Features
============================================
Supports: YOLOv8, YOLOv10, YOLOv11
Model Sizes: n, s, m, l, x
Features: WandB integration, validation visualization, flexible configuration
"""

import os
import argparse
import random
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import wandb
import cv2
import shutil

import glob


class TrainingConfig:
    """Configuration class for YOLO training"""
    
    # Supported YOLO versions and model sizes
    SUPPORTED_VERSIONS = ['8', '10', '11']
    SUPPORTED_SIZES = ['n', 's', 'm', 'l', 'x']
    
    def __init__(self, args):
        self.base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        
        # Model configuration
        self.yolo_version = args.version
        self.model_size = args.size
        self.model_name = f"yolov{self.yolo_version}{self.model_size}"
        
        # Paths
        if args.weights:
            self.model_path = Path(args.weights)
        else:
            # Check models directory if not in root
            root_path = self.base_dir / f"{self.model_name}.pt"
            models_path = self.base_dir / "models" / f"{self.model_name}.pt"
            
            if not root_path.exists() and models_path.exists():
                self.model_path = models_path
            else:
                self.model_path = root_path
        
        self.data_path = self.base_dir / args.data
        


        # Training parameters
        self.epochs = args.epochs
        self.batch_size = args.batch
        self.img_size = args.imgsz
        self.device = args.device
        self.workers = args.workers
        
        # Optimization parameters
        self.lr0 = args.lr0
        self.lrf = args.lrf
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.optimizer = args.optimizer
        
        # Augmentation parameters
        self.hsv_h = args.hsv_h
        self.hsv_s = args.hsv_s
        self.hsv_v = args.hsv_v
        self.degrees = args.degrees
        self.translate = args.translate
        self.scale = args.scale
        self.flipud = args.flipud
        self.fliplr = args.fliplr
        self.mosaic = args.mosaic
        self.mixup = args.mixup
        
        # Project configuration
        dataset_name = self.data_path.parent.name.lower()
        self.project_name = f"{dataset_name}-{self.model_name}"
        self.run_name = args.name if args.name else f"{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # Make the path absolute so it strictly avoids YOLO's default 'detect' directory
        self.project_dir = args.project if args.project else str(self.base_dir / 'runs' / 'train')
        
        # WandB configuration
        self.use_wandb = not args.no_wandb
        self.wandb_project = args.wandb_project if args.wandb_project else self.project_name
        
        # Validation visualization
        self.save_val_images = args.save_val_images
        self.num_val_images = args.num_val_images
        
        # Other options
        self.resume = args.resume
        self.patience = args.patience
        self.save_period = args.save_period
        self.exist_ok = args.exist_ok
        self.pretrained = args.pretrained
        self.verbose = args.verbose
        
    def validate(self):
        """Validate configuration"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        if not self.model_path.exists() and not self.pretrained:
            print(f"⚠️  Warning: Model file not found: {self.model_path}")
            print(f"    Will download pretrained {self.model_name} model from Ultralytics")
        
        if self.yolo_version not in self.SUPPORTED_VERSIONS:
            raise ValueError(f"Unsupported YOLO version: {self.yolo_version}. Supported: {self.SUPPORTED_VERSIONS}")
        
        if self.model_size not in self.SUPPORTED_SIZES:
            raise ValueError(f"Unsupported model size: {self.model_size}. Supported: {self.SUPPORTED_SIZES}")
        
        return True
    
    def print_config(self):
        """Print training configuration"""
        print("\n" + "="*60)
        print("🚀 YOLO Training Configuration")
        print("="*60)
        print(f"📦 Model: {self.model_name}")
        print(f"📁 Weights: {self.model_path}")
        print(f"📊 Dataset: {self.data_path}")
        print(f"🎯 Project: {self.wandb_project}")
        print(f"📝 Run Name: {self.run_name}")
        print(f"\n⚙️  Training Parameters:")
        print(f"   - Epochs: {self.epochs}")
        print(f"   - Batch Size: {self.batch_size}")
        print(f"   - Image Size: {self.img_size}")
        print(f"   - Device: {self.device}")
        print(f"   - Optimizer: {self.optimizer}")
        print(f"   - Learning Rate: {self.lr0}")
        print(f"\n📈 WandB: {'Enabled' if self.use_wandb else 'Disabled'}")
        print(f"📸 Save Validation Images: {'Yes' if self.save_val_images else 'No'}")
        if self.save_val_images:
            print(f"   - Images per epoch: {self.num_val_images}")
        print("="*60 + "\n")


class ValidationImageSaver:
    """Callback to save random validation images during training"""
    
    def __init__(self, save_dir, num_images=3):
        self.save_dir = Path(save_dir)
        self.num_images = num_images
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def on_val_end(self, validator):
        """Called at the end of validation"""
        try:
            # Get validation results
            if hasattr(validator, 'save_dir'):
                val_img_dir = Path(validator.save_dir)
                
                # Find validation images (usually in val_batch*_pred.jpg)
                pred_images = list(val_img_dir.glob("val_batch*_pred.jpg"))
                
                if pred_images:
                    # Get current epoch
                    epoch = validator.epoch if hasattr(validator, 'epoch') else 'unknown'
                    
                    # Create epoch directory
                    epoch_dir = self.save_dir / f"epoch_{epoch}"
                    epoch_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Randomly select images
                    selected_images = random.sample(pred_images, min(self.num_images, len(pred_images)))
                    
                    # Copy selected images
                    for idx, img_path in enumerate(selected_images):
                        dest_path = epoch_dir / f"val_sample_{idx+1}.jpg"
                        shutil.copy2(img_path, dest_path)
                    
                    print(f"✅ Saved {len(selected_images)} validation images to {epoch_dir}")
                    
        except Exception as e:
            print(f"⚠️  Warning: Failed to save validation images: {e}")


def setup_wandb(config):
    """Setup WandB configuration"""
    if config.use_wandb:
        try:
            # Initialize WandB manually to ensure it works
            wandb.init(
                project=config.wandb_project,
                name=config.run_name,
                config={
                    "model": config.model_name,
                    "epochs": config.epochs,
                    "batch_size": config.batch_size,
                    "img_size": config.img_size,
                    "optimizer": config.optimizer,
                    "lr0": config.lr0,
                    "device": config.device,
                }
            )
            print("✅ WandB initialized successfully (Manual Mode)")
            return True
        except Exception as e:
            print(f"⚠️  Warning: WandB initialization failed: {e}")
            print("    Continuing without WandB logging...")
            config.use_wandb = False
            return False
    return False


def on_fit_epoch_end(trainer):
    """Callback to log metrics at the end of each fit epoch"""
    if wandb.run:
        # Helper to safely get float value
        def get_val(val):
            if hasattr(val, 'item'):
                return val.item()
            return val

        # Log training metrics
        metrics = {}
        
        # Training losses
        if hasattr(trainer, 'loss_items'):
             metrics["train/box_loss"] = get_val(trainer.loss_items[0])
             metrics["train/cls_loss"] = get_val(trainer.loss_items[1])
             metrics["train/dfl_loss"] = get_val(trainer.loss_items[2])
        
        # Learning rate
        if hasattr(trainer, 'optimizer') and trainer.optimizer.param_groups:
            metrics["lr/pg0"] = teacher_val = trainer.optimizer.param_groups[0]['lr']
        
        # Validation metrics
        if hasattr(trainer, 'metrics'):
            # Copy all validation metrics
            for k, v in trainer.metrics.items():
                metrics[k] = get_val(v)
            
        # Debug print to ensure we are logging
        print(f"\r🚀 WandB Log: {len(metrics)} metrics pushed (Epoch {trainer.epoch})", end="")
        
        try:
            wandb.log(metrics)
        except Exception as e:
            print(f"\n❌ WandB Log Failed: {e}")


def train_model(config):
    """Main training function"""
    
    # Validate configuration
    config.validate()
    config.print_config()
    
    # Setup WandB
    setup_wandb(config)
    
    # Load YOLO model
    print(f"\n📦 Loading {config.model_name} model...")
    if config.model_path.exists():
        model = YOLO(str(config.model_path))
        print(f"✅ Loaded model from: {config.model_path}")
    else:
        model = YOLO(f"{config.model_name}.pt")  # Will download from Ultralytics
        print(f"✅ Loaded pretrained {config.model_name} model")
    
    # Add WandB callback if enabled
    if config.use_wandb:
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    
    # Prepare training arguments
    train_args = {
        'data': str(config.data_path),
        'epochs': config.epochs,
        'imgsz': config.img_size,
        'batch': config.batch_size,
        'device': config.device,
        'workers': config.workers,
        'optimizer': config.optimizer,
        'lr0': config.lr0,
        'lrf': config.lrf,
        'momentum': config.momentum,
        'weight_decay': config.weight_decay,
        'hsv_h': config.hsv_h,
        'hsv_s': config.hsv_s,
        'hsv_v': config.hsv_v,
        'degrees': config.degrees,
        'translate': config.translate,
        'scale': config.scale,
        'flipud': config.flipud,
        'fliplr': config.fliplr,
        'mosaic': config.mosaic,
        'mixup': config.mixup,
        'name': config.run_name,
        'project': config.project_dir,
        'exist_ok': config.exist_ok,
        'pretrained': config.pretrained,
        'verbose': config.verbose,
        'patience': config.patience,
        'save_period': config.save_period,
        'save': True,
        'plots': True,
    }
    
    # Add resume if specified
    if config.resume:
        train_args['resume'] = config.resume
    
    print("\n🚀 Starting training...")
    print("-" * 60)
    
    # Train model
    results = model.train(**train_args)
    
    # Get the save directory
    save_dir = Path(model.trainer.save_dir)
    
    # Save validation images if enabled
    if config.save_val_images:
        val_save_dir = save_dir / "validation_samples"
        print(f"\n📸 Validation images saved in: {val_save_dir}")
    
    print("\n" + "="*60)
    print("✅ Training completed!")
    print("="*60)
    print(f"📁 Results saved to: {save_dir}")
    print(f"🏆 Best weights: {save_dir / 'weights' / 'best.pt'}")
    print(f"📊 Last weights: {save_dir / 'weights' / 'last.pt'}")
    
    # Run validation
    print("\n📊 Running final validation...")
    val_project_dir = str(config.base_dir / 'runs' / 'results' / 'summary')
    val_run_name = "sum_" + config.run_name.split('/')[-1]
    val_results = model.val(project=val_project_dir, name=val_run_name)
    


    # Print metrics
    if hasattr(val_results, 'box'):
        print("\n📈 Validation Metrics:")
        print(f"   - mAP50: {val_results.box.map50:.4f}")
        print(f"   - mAP50-95: {val_results.box.map:.4f}")
        print(f"   - Precision: {val_results.box.mp:.4f}")
        print(f"   - Recall: {val_results.box.mr:.4f}")
    
    # Close WandB run
    if config.use_wandb and wandb.run:
        wandb.finish()
    
    return results, save_dir


def main():
    parser = argparse.ArgumentParser(
        description='YOLO Training Script with Enhanced Features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train YOLOv11n with default settings
  python train.py --version 11 --size n

  # Train YOLOv8s with custom epochs and batch size
  python train.py --version 8 --size s --epochs 200 --batch 32

  # Train with custom weights and data
  python train.py --version 11 --size m --weights custom.pt --data path/to/data.yaml

  # Resume training
  python train.py --resume runs/train/exp/weights/last.pt

  # Train without WandB
  python train.py --version 11 --size n --no-wandb
        """
    )
    
    # Model configuration
    parser.add_argument('--version', type=str, default='11', choices=['8', '10', '11'],
                        help='YOLO version (default: 11)')
    parser.add_argument('--size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help='Model size: n(nano), s(small), m(medium), l(large), x(xlarge) (default: n)')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to custom weights file (default: auto yolo{version}{size}.pt)')
    parser.add_argument('--data', type=str, default='dataset/Etching_M1/data.yaml',
                        help='Path to dataset YAML file (default: dataset/Etching_M1/data.yaml)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch', type=int, default=4,
                        help='Batch size (default: 16)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size for training (default: 640)')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to train on: 0, 1, 2, or cpu (default: 0)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of worker threads for data loading (default: 8)')
    
    # Optimizer parameters
    parser.add_argument('--optimizer', type=str, default='auto',
                        choices=['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp', 'auto'],
                        help='Optimizer (default: auto)')
    parser.add_argument('--lr0', type=float, default=0.01,
                        help='Initial learning rate (default: 0.01)')
    parser.add_argument('--lrf', type=float, default=0.01,
                        help='Final learning rate (lr0 * lrf) (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.937,
                        help='SGD momentum/Adam beta1 (default: 0.937)')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='Optimizer weight decay (default: 0.0005)')
    
    # Data augmentation parameters
    parser.add_argument('--hsv-h', type=float, default=0.015,
                        help='HSV-Hue augmentation (default: 0.015)')
    parser.add_argument('--hsv-s', type=float, default=0.7,
                        help='HSV-Saturation augmentation (default: 0.7)')
    parser.add_argument('--hsv-v', type=float, default=0.4,
                        help='HSV-Value augmentation (default: 0.4)')
    parser.add_argument('--degrees', type=float, default=0.0,
                        help='Image rotation (+/- degrees) (default: 0.0)')
    parser.add_argument('--translate', type=float, default=0.1,
                        help='Image translation (+/- fraction) (default: 0.1)')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='Image scale (+/- gain) (default: 0.5)')
    parser.add_argument('--flipud', type=float, default=0.0,
                        help='Image flip up-down probability (default: 0.0)')
    parser.add_argument('--fliplr', type=float, default=0.5,
                        help='Image flip left-right probability (default: 0.5)')
    parser.add_argument('--mosaic', type=float, default=1.0,
                        help='Mosaic augmentation probability (default: 1.0)')
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='MixUp augmentation probability (default: 0.0)')
    
    # Project configuration
    parser.add_argument('--project', type=str, default=None,
                        help='Project directory (default: workspace/runs/train)')
    parser.add_argument('--name', type=str, default=None,
                        help='Run name (default: auto-generated)')
    parser.add_argument('--exist-ok', action='store_true',
                        help='Allow overwriting existing project/name')
    
    # WandB configuration
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable WandB logging')
    parser.add_argument('--wandb-project', type=str, default=None,
                        help='WandB project name (default: auto-generated)')
    
    # Validation visualization
    parser.add_argument('--save-val-images', action='store_true', default=True,
                        help='Save random validation images each epoch (default: True)')
    parser.add_argument('--num-val-images', type=int, default=3,
                        help='Number of validation images to save per epoch (default: 3)')

    
    # Other options
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience (default: 50)')
    parser.add_argument('--save-period', type=int, default=-1,
                        help='Save checkpoint every x epochs (default: -1, disabled)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained model (default: True)')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    # Create configuration
    config = TrainingConfig(args)
    
    # Train model
    try:
        results, save_dir = train_model(config)
        print("\n✨ All done! Happy training! ✨\n")
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
