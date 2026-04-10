import cv2
from ultralytics import YOLO
import argparse
from pathlib import Path
from datetime import datetime
import os


def main():
    """
    Real-time object detection using webcam and YOLO model.
    Press 'q' to quit.
    """
    parser = argparse.ArgumentParser(description='Real-time object detection with YOLO')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to YOLO model weights file (.pt)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device index (default: 0)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS (default: 0.45)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Inference image size (default: 640)')
    parser.add_argument('--save', action='store_true',
                        help='Save the detection video')
    parser.add_argument('--output', type=str, default='runs/results/live detection/camera_output.mp4',
                        help='Output video filename (default: runs/results/live detection/camera_output.mp4)')
    parser.add_argument('--auto-save', action='store_true',
                        help='Automatically save frames when detections are found')
    parser.add_argument('--save-crops', action='store_true',
                        help='Save individual cropped objects (in addition to full frames)')
    parser.add_argument('--output-dir', type=str, default='runs/results/live detection',
                        help='Base directory for saving detections (default: runs/results/live detection)')
    
    args = parser.parse_args()
    
    # Check if weights file exists
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"❌ Error: Weights file not found: {args.weights}")
        print(f"💡 Please provide a valid path to your trained model weights.")
        return
    
    # Load YOLO model
    print(f"📦 Loading YOLO model from: {args.weights}")
    try:
        model = YOLO(str(weights_path))
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Open camera
    print(f"📹 Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"❌ Error: Cannot open camera {args.camera}")
        print("💡 Try a different camera index (--camera 1, --camera 2, etc.)")
        return
    
    # Get camera properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Default to 30 if fps is 0
    
    print(f"📐 Camera resolution: {frame_width}x{frame_height} @ {fps} FPS")
    print("✅ Camera opened successfully!")
    
    # Create timestamped output folder if auto-save is enabled
    save_folder = None
    crops_folder = None
    if args.auto_save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_folder = Path(args.output_dir) / timestamp / "frames"
        save_folder.mkdir(parents=True, exist_ok=True)
        
        if args.save_crops:
            crops_folder = Path(args.output_dir) / timestamp / "crops"
            crops_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Created output folder: {Path(args.output_dir) / timestamp}")
        print(f"💾 Frames will be saved to: {save_folder}")
        if args.save_crops:
            print(f"✂️ Cropped objects will be saved to: {crops_folder}")
    
    print("\n🎥 Starting real-time detection...")
    print("💡 Press 'q' to quit")
    
    # Initialize video writer if save is enabled
    video_writer = None
    if args.save:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))
        print(f"💾 Saving video to: {args.output}")
    
    frame_count = 0
    
    try:
        while True:
            # Read frame from camera
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Error: Cannot read frame from camera")
                break
            
            frame_count += 1
            
            # Run YOLO inference
            results = model(frame, 
                           conf=args.conf, 
                           iou=args.iou,
                           imgsz=args.img_size,
                           verbose=False)
            
            # Annotate frame with detections
            annotated_frame = results[0].plot()
            
            # Display detection count
            num_detections = len(results[0].boxes)
            
            # Auto-save frame if detections are found
            if args.auto_save and num_detections > 0 and save_folder is not None:
                frame_filename = save_folder / f"frame_{frame_count:06d}_det{num_detections}.jpg"
                cv2.imwrite(str(frame_filename), annotated_frame)
                
                # Save cropped objects if enabled
                if args.save_crops and crops_folder is not None:
                    boxes = results[0].boxes
                    for idx, box in enumerate(boxes):
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Ensure coordinates are within frame bounds
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame_width, x2), min(frame_height, y2)
                        
                        # Crop object from original frame
                        cropped_obj = frame[y1:y2, x1:x2]
                        
                        # Get class name and confidence
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        cls_name = results[0].names[cls_id]
                        
                        # Save cropped object
                        crop_filename = crops_folder / f"frame_{frame_count:06d}_obj{idx}_{cls_name}_{conf:.2f}.jpg"
                        cv2.imwrite(str(crop_filename), cropped_obj)
            
            cv2.putText(annotated_frame, 
                       f"Detections: {num_detections} | Frame: {frame_count}",
                       (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, 
                       (0, 255, 0), 
                       2)
            
            # Show saved status if auto-save is enabled
            if args.auto_save:
                save_status = f"Auto-save: {'ON' if num_detections > 0 else 'Waiting...'}"
                cv2.putText(annotated_frame,
                           save_status,
                           (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.7,
                           (0, 255, 255) if num_detections > 0 else (128, 128, 128),
                           2)
            
            # Show frame
            cv2.imshow('YOLO Real-time Detection (Press Q to quit)', annotated_frame)
            
            # Save frame if enabled
            if video_writer is not None:
                video_writer.write(annotated_frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n👋 Quitting...")
                break
                
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during detection: {e}")
    finally:
        # Release resources
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        print(f"✅ Processed {frame_count} frames")
        if args.save:
            print(f"✅ Video saved to: {args.output}")
        print("🎬 Done!")


if __name__ == "__main__":
    main()
