"""
YOLO Prediction Script
======================
Perform inference on images, videos, or directories using trained YOLO models
"""

import os
import argparse
from pathlib import Path
from datetime import datetime
import cv2
from ultralytics import YOLO
import time


class PredictionConfig:
    """Configuration for prediction"""
    
    SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']
    SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.m4v', '.wmv']
    
    def __init__(self, args):
        self.source = Path(args.source)
        self.weights = Path(args.weights)
        self.conf = args.conf
        self.iou = args.iou
        self.device = args.device
        self.save_dir = Path(args.save_dir) if args.save_dir else None
        self.show = args.show
        self.save = args.save
        self.save_txt = args.save_txt
        self.save_conf = args.save_conf
        self.line_width = args.line_width
        self.max_det = args.max_det
        self.classes = args.classes
        self.agnostic_nms = args.agnostic_nms
        self.verbose = args.verbose
        
        # Determine source type
        self.source_type = self._determine_source_type()
        
        # Setup save directory
        if self.save_dir is None:
            model_folder = self.weights.parent.parent.name
            if 'yolo' in model_folder.lower():
                self.save_dir = Path(f'runs/results/video/video_{model_folder}')
            else:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                self.save_dir = Path(f'runs/results/video/video_{self.weights.stem}_{timestamp}')
        
        if self.save:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def _determine_source_type(self):
        """Determine if source is image, video, or directory"""
        if not self.source.exists():
            raise FileNotFoundError(f"Source not found: {self.source}")
        
        if self.source.is_dir():
            return 'directory'
        
        suffix = self.source.suffix.lower()
        if suffix in self.SUPPORTED_IMAGE_FORMATS:
            return 'image'
        elif suffix in self.SUPPORTED_VIDEO_FORMATS:
            return 'video'
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def validate(self):
        """Validate configuration"""
        if not self.weights.exists():
            raise FileNotFoundError(f"Weights file not found: {self.weights}")
        
        if not self.source.exists():
            raise FileNotFoundError(f"Source not found: {self.source}")
        
        return True
    
    def print_config(self):
        """Print prediction configuration"""
        print("\n" + "="*60)
        print("🎯 YOLO Prediction Configuration")
        print("="*60)
        print(f"📦 Weights: {self.weights}")
        print(f"📁 Source: {self.source}")
        print(f"📊 Source Type: {self.source_type}")
        print(f"💾 Save Directory: {self.save_dir}")
        print(f"\n⚙️  Inference Parameters:")
        print(f"   - Confidence Threshold: {self.conf}")
        print(f"   - IoU Threshold: {self.iou}")
        print(f"   - Device: {self.device}")
        print(f"   - Max Detections: {self.max_det}")
        print(f"   - Save Results: {self.save}")
        print(f"   - Show Results: {self.show}")
        print("="*60 + "\n")


def predict_image(model, config, image_path):
    """Predict on a single image"""
    print(f"📸 Processing image: {image_path.name}")
    
    start_time = time.time()
    
    # Run prediction
    results = model.predict(
        source=str(image_path),
        conf=config.conf,
        iou=config.iou,
        device=config.device,
        max_det=config.max_det,
        classes=config.classes,
        agnostic_nms=config.agnostic_nms,
        line_width=config.line_width,
        save=config.save,
        save_txt=config.save_txt,
        save_conf=config.save_conf,
        show=config.show,
        verbose=config.verbose,
    )
    
    inference_time = time.time() - start_time
    
    # Get result
    result = results[0]
    
    # Print statistics
    num_detections = len(result.boxes)
    if num_detections > 0:
        classes_detected = result.boxes.cls.unique().cpu().numpy()
        class_names = [result.names[int(c)] for c in classes_detected]
        
        print(f"   ✅ Found {num_detections} object(s)")
        print(f"   📋 Classes: {', '.join(class_names)}")
    else:
        print(f"   ℹ️  No objects detected")
    
    print(f"   ⏱️  Inference time: {inference_time:.3f}s")
    
    # Save annotated image if requested
    if config.save:
        save_path = config.save_dir / f"pred_{image_path.stem}.jpg"
        annotated_img = result.plot()
        cv2.imwrite(str(save_path), annotated_img)
        print(f"   💾 Saved to: {save_path}")
    
    return result


def predict_video(model, config, video_path):
    """Predict on a video"""
    print(f"🎥 Processing video: {video_path.name}")
    
    # Get video info
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    print(f"   📊 Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Setup output video writer if saving
    if config.save:
        save_path = config.save_dir / f"pred_{video_path.stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))
    
    start_time = time.time()
    
    # Run prediction on video
    results = model.predict(
        source=str(video_path),
        conf=config.conf,
        iou=config.iou,
        device=config.device,
        max_det=config.max_det,
        classes=config.classes,
        agnostic_nms=config.agnostic_nms,
        line_width=config.line_width,
        save=False,  # We'll handle saving manually
        save_txt=config.save_txt,
        save_conf=config.save_conf,
        show=config.show,
        verbose=config.verbose,
        stream=True,  # Use streaming for videos
    )
    
    # Process results
    total_detections = 0
    frame_count = 0
    
    for result in results:
        frame_count += 1
        num_detections = len(result.boxes)
        total_detections += num_detections
        
        # Show progress every 30 frames
        if frame_count % 30 == 0:
            print(f"   ⏳ Processed {frame_count}/{total_frames} frames...", end='\r')
        
        # Save frame if requested
        if config.save:
            annotated_frame = result.plot()
            out.write(annotated_frame)
    
    processing_time = time.time() - start_time
    
    if config.save:
        out.release()
        print(f"\n   💾 Saved to: {save_path}")
    
    # Print statistics
    avg_detections = total_detections / frame_count if frame_count > 0 else 0
    print(f"\n   ✅ Processed {frame_count} frames")
    print(f"   📊 Total detections: {total_detections}")
    print(f"   📈 Average detections per frame: {avg_detections:.2f}")
    print(f"   ⏱️  Processing time: {processing_time:.2f}s ({frame_count/processing_time:.2f} fps)")


def predict_directory(model, config, directory):
    """Predict on all images in a directory"""
    print(f"📁 Processing directory: {directory}")
    
    # Find all images
    image_files = []
    for ext in config.SUPPORTED_IMAGE_FORMATS:
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print("   ⚠️  No images found in directory")
        return
    
    print(f"   📊 Found {len(image_files)} image(s)")
    
    # Create subdirectory for results
    if config.save:
        results_dir = config.save_dir / directory.name
        results_dir.mkdir(parents=True, exist_ok=True)
        original_save_dir = config.save_dir
        config.save_dir = results_dir
    
    # Process each image
    total_detections = 0
    for idx, image_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}]", end=" ")
        result = predict_image(model, config, image_path)
        total_detections += len(result.boxes)
    
    # Restore original save_dir
    if config.save:
        config.save_dir = original_save_dir
    
    # Print summary
    print(f"\n📊 Summary:")
    print(f"   - Total images processed: {len(image_files)}")
    print(f"   - Total detections: {total_detections}")
    print(f"   - Average detections per image: {total_detections/len(image_files):.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='YOLO Prediction Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict on a single image
  python predict.py --source image.jpg --weights runs/train/exp/weights/best.pt

  # Predict on a video with custom confidence
  python predict.py --source video.mp4 --weights best.pt --conf 0.5

  # Predict on all images in a directory
  python predict.py --source images/ --weights best.pt --save

  # Predict and show results in real-time
  python predict.py --source image.jpg --weights best.pt --show

  # Predict with specific classes only
  python predict.py --source image.jpg --weights best.pt --classes 0 1 2
        """
    )
    
    # Required arguments
    parser.add_argument('--source', type=str, required=True,
                        help='Path to image, video, or directory')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to trained model weights (.pt file)')
    
    # Inference parameters
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS (default: 0.45)')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to run on: 0, 1, 2, or cpu (default: 0)')
    parser.add_argument('--max-det', type=int, default=300,
                        help='Maximum number of detections per image (default: 300)')
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                        help='Filter by class: --classes 0 1 2')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='Class-agnostic NMS')
    
    # Output parameters
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save results (default: runs/predict/exp_<timestamp>)')
    parser.add_argument('--save', action='store_true', default=True,
                        help='Save prediction results (default: True)')
    parser.add_argument('--save-txt', action='store_true',
                        help='Save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='Save confidences in --save-txt labels')
    parser.add_argument('--show', action='store_true',
                        help='Display results in real-time')
    
    # Visualization parameters
    parser.add_argument('--line-width', type=int, default=None,
                        help='Bounding box line width (default: auto)')
    
    # Other options
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    # Create configuration
    config = PredictionConfig(args)
    
    # Validate configuration
    config.validate()
    config.print_config()
    
    # Load model
    print(f"📦 Loading model from: {config.weights}")
    model = YOLO(str(config.weights))
    print(f"✅ Model loaded successfully")
    
    # Run prediction based on source type
    try:
        if config.source_type == 'image':
            predict_image(model, config, config.source)
        elif config.source_type == 'video':
            predict_video(model, config, config.source)
        elif config.source_type == 'directory':
            predict_directory(model, config, config.source)
        
        print("\n" + "="*60)
        print("✅ Prediction completed!")
        print("="*60)
        if config.save:
            print(f"📁 Results saved to: {config.save_dir}")
        print()
        
    except KeyboardInterrupt:
        print("\n⚠️  Prediction interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
