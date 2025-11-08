#!/usr/bin/env python3
"""
Complete Floorplan Parser Pipeline
===================================

End-to-end pipeline that:
1. Trains detection model on COCO annotations
2. Trains ML room classifier (optional enhancement)
3. Runs inference with OCR and structured output generation

Usage:
    # Train everything
    python complete_pipeline.py --mode train
    
    # Inference on single image
    python complete_pipeline.py --mode infer --image path/to/image.jpg
    
    # Batch inference
    python complete_pipeline.py --mode batch --image_dir path/to/images/
"""

import argparse
import os
import json
import glob
from pathlib import Path


def setup_directories():
    """Create necessary directories"""
    dirs = ['checkpoints', 'logs', 'models', 'results']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✓ Created directories")


def train_detection_model(config):
    """Train the object detection model"""
    from floorplan_trainer import Trainer
    
    print("\n" + "="*60)
    print("TRAINING DETECTION MODEL")
    print("="*60)
    
    trainer = Trainer(config)
    trainer.train()
    
    print("\n✓ Detection model training completed!")
    print(f"  Best model saved at: {config['checkpoint_dir']}/best.pth")


def train_room_classifier():
    """Train the ML room classifier"""
    from room_classifier import train_room_classifier
    
    print("\n" + "="*60)
    print("TRAINING ROOM CLASSIFIER")
    print("="*60)
    
    classifier = train_room_classifier(save_path='./models/room_classifier.pkl')
    
    print("\n✓ Room classifier training completed!")


def run_inference(image_path, checkpoint_path, output_dir='./results'):
    """Run inference on a single image"""
    from floorplan_inference import FloorplanParser
    
    print(f"\nProcessing: {image_path}")
    
    # Initialize parser
    parser = FloorplanParser(
        checkpoint_path=checkpoint_path,
        confidence_threshold=0.2
    )
    
    # Parse
    basename = os.path.basename(image_path)
    viz_path = os.path.join(output_dir, basename.replace('.jpg', '_viz.jpg'))
    
    output, detections = parser.parse(
        image_path=image_path,
        save_visualization=viz_path
    )
    
    # Save JSON
    json_path = os.path.join(output_dir, basename.replace('.jpg', '.json'))
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Print results
    print("\n" + "="*60)
    print("STRUCTURED OUTPUT:")
    print("="*60)
    print(json.dumps(output, indent=2))
    
    print("\n" + "="*60)
    print("DETECTIONS:")
    print("="*60)
    for i, det in enumerate(detections[:10], 1):  # Show first 10
        print(f"{i}. {det['category_name']}: '{det['text']}' (conf: {det['confidence']:.2f})")
    if len(detections) > 10:
        print(f"... and {len(detections) - 10} more")
    
    print(f"\n✓ Results saved to:")
    print(f"  - JSON: {json_path}")
    print(f"  - Visualization: {viz_path}")
    
    return output


def batch_inference(image_dir, checkpoint_path, output_dir='./results'):
    """Run inference on multiple images"""
    from floorplan_inference import FloorplanParser
    
    print("\n" + "="*60)
    print("BATCH INFERENCE")
    print("="*60)
    
    # Find all images
    image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))
    print(f"Found {len(image_paths)} images")
    
    if len(image_paths) == 0:
        print("⚠ No images found!")
        return
    
    # Initialize parser
    parser = FloorplanParser(
        checkpoint_path=checkpoint_path,
        confidence_threshold=0.2
    )
    
    # Process all images
    results = []
    for i, img_path in enumerate(image_paths, 1):
        basename = os.path.basename(img_path)
        print(f"\n[{i}/{len(image_paths)}] Processing {basename}...")
        
        try:
            viz_path = os.path.join(output_dir, basename.replace('.jpg', '_viz.jpg'))
            output, detections = parser.parse(img_path, save_visualization=viz_path)
            
            # Save JSON
            json_path = os.path.join(output_dir, basename.replace('.jpg', '.json'))
            with open(json_path, 'w') as f:
                json.dump(output, f, indent=2)
            
            results.append({
                'image': basename,
                'output': output,
                'num_detections': len(detections)
            })
            
            print(f"  ✓ Rooms: {output['rooms']}, Bathrooms: {output['bathrooms']}, Kitchens: {output['kitchens']}")
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                'image': basename,
                'error': str(e)
            })
    
    # Save summary
    summary_path = os.path.join(output_dir, 'batch_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("BATCH INFERENCE COMPLETED")
    print("="*60)
    print(f"✓ Processed {len(results)} images")
    print(f"✓ Results saved to: {output_dir}")
    print(f"✓ Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Floorplan Parser Pipeline')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'train_classifier', 'infer', 'batch'],
                       help='Pipeline mode')
    parser.add_argument('--image', type=str, help='Path to input image (for infer mode)')
    parser.add_argument('--image_dir', type=str, help='Path to image directory (for batch mode)')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    
    # Training config
    parser.add_argument('--data_dir', type=str, 
                       default='/home/mithlesh/smartsense/assets/train',
                       help='Directory containing training images')
    parser.add_argument('--annotation_file', type=str,
                       default='/home/mithlesh/smartsense/assets/train/annotations.coco.json',
                       help='COCO annotations file')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    # Setup
    setup_directories()
    
    # Execute mode
    if args.mode == 'train':
        config = {
            'image_dir': args.data_dir,
            'annotation_file': args.annotation_file,
            'train_split': 0.8,
            'max_objects': 50,
            'iou_threshold': 0.5,
            'batch_size': args.batch_size,
            'num_epochs': args.epochs,
            'learning_rate': args.lr,
            'weight_decay': 1e-5,
            'num_workers': 4,
            'checkpoint_dir': './checkpoints',
            'log_dir': './logs',
            'save_every': 5
        }
        
        train_detection_model(config)
        
        # Optionally train room classifier
        print("\n" + "="*60)
        print("Would you like to train the ML room classifier? (y/n)")
        response = input().strip().lower()
        if response == 'y':
            train_room_classifier()
    
    elif args.mode == 'train_classifier':
        train_room_classifier()
    
    elif args.mode == 'infer':
        if not args.image:
            print("Error: --image required for infer mode")
            return
        
        if not os.path.exists(args.image):
            print(f"Error: Image not found: {args.image}")
            return
        
        if not os.path.exists(args.checkpoint):
            print(f"Error: Checkpoint not found: {args.checkpoint}")
            print("Please train the model first using --mode train")
            return
        
        os.makedirs(args.output_dir, exist_ok=True)
        run_inference(args.image, args.checkpoint, args.output_dir)
    
    elif args.mode == 'batch':
        if not args.image_dir:
            print("Error: --image_dir required for batch mode")
            return
        
        if not os.path.exists(args.image_dir):
            print(f"Error: Directory not found: {args.image_dir}")
            return
        
        if not os.path.exists(args.checkpoint):
            print(f"Error: Checkpoint not found: {args.checkpoint}")
            print("Please train the model first using --mode train")
            return
        
        os.makedirs(args.output_dir, exist_ok=True)
        batch_inference(args.image_dir, args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()


# ============================================================================
# QUICK START EXAMPLES
# ============================================================================

"""
EXAMPLE 1: Train from scratch
------------------------------
python complete_pipeline.py --mode train \
    --data_dir /home/mithlesh/smartsense/assets/train \
    --annotation_file /home/mithlesh/smartsense/assets/train/annotations.coco.json \
    --batch_size 8 \
    --epochs 100


EXAMPLE 2: Inference on single image
-------------------------------------
python complete_pipeline.py --mode infer \
    --image /path/to/floorplan.jpg \
    --checkpoint ./checkpoints/best.pth \
    --output_dir ./results


EXAMPLE 3: Batch inference
---------------------------
python complete_pipeline.py --mode batch \
    --image_dir /path/to/test/images \
    --checkpoint ./checkpoints/best.pth \
    --output_dir ./results


EXAMPLE 4: Train only the room classifier
------------------------------------------
python complete_pipeline.py --mode train_classifier


EXPECTED OUTPUT STRUCTURE:
--------------------------
results/
├── floorplan1.json          # Structured JSON output
├── floorplan1_viz.jpg       # Visualization with boxes
├── floorplan2.json
├── floorplan2_viz.jpg
└── batch_summary.json       # Summary of all processed images


JSON OUTPUT FORMAT:
-------------------
{
  "rooms": 3,
  "halls": 1,
  "kitchens": 1,
  "bathrooms": 2,
  "rooms_detail": [
    {
      "label": "Bedroom",
      "count": 2,
      "approx_area": 25.6
    },
    {
      "label": "Bathroom",
      "count": 2,
      "approx_area": 8.4
    },
    {
      "label": "Kitchen",
      "count": 1,
      "approx_area": 12.0
    },
    {
      "label": "Living",
      "count": 1,
      "approx_area": 35.2
    }
  ]
}
"""