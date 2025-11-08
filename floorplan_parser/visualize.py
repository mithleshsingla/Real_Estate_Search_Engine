import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Import your model (adjust the import path as needed)
from floorplan_model import FloorplanDetector  # Replace with actual import

class PredictionVisualizer:
    """Visualize model predictions vs ground truth"""
    
    def __init__(self, model_path, category_lookup, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.category_lookup = category_lookup
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = FloorplanDetector(
            num_classes=len(category_lookup), 
            max_objects=50
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Device: {self.device}")
    
    def preprocess_image(self, img_path, target_size=640):
        """Load and preprocess image for model input"""
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        # Resize
        img_resized = cv2.resize(img_rgb, (target_size, target_size))
        
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        return img_tensor, img_rgb, (h, w)
    
    @torch.no_grad()
    def predict(self, img_tensor, conf_threshold=0.5):
        """Run inference and return predictions"""
        objectness, bboxes, class_logits = self.model(img_tensor)
        
        # Get predictions above confidence threshold
        objectness = objectness[0].cpu().numpy()  # [max_objects]
        bboxes = bboxes[0].cpu().numpy()  # [max_objects, 4]
        class_probs = torch.softmax(class_logits[0], dim=-1).cpu().numpy()  # [max_objects, num_classes]
        
        # Filter by confidence
        mask = objectness > conf_threshold
        
        predictions = []
        for i in range(len(objectness)):
            if mask[i]:
                pred_class = np.argmax(class_probs[i])
                predictions.append({
                    'bbox': bboxes[i],  # [x1, y1, x2, y2] normalized
                    'confidence': objectness[i],
                    'class_id': pred_class,
                    'class_name': self.category_lookup.get(pred_class, 'unknown')
                })
        
        return predictions
    
    def visualize_single_image(self, img_path, annotations, conf_threshold=0.5, 
                              show_gt=True, show_pred=True):
        """Visualize predictions and ground truth for a single image"""
        
        # Preprocess and predict
        img_tensor, img_rgb, (orig_h, orig_w) = self.preprocess_image(img_path)
        predictions = self.predict(img_tensor, conf_threshold)
        
        # Create figure with subplots
        if show_gt and show_pred:
            fig, axes = plt.subplots(1, 3, figsize=(20, 7))
            ax_gt, ax_pred, ax_both = axes
        elif show_gt or show_pred:
            fig, axes = plt.subplots(1, 2, figsize=(14, 7))
            if show_gt:
                ax_gt, ax_both = axes
            else:
                ax_pred, ax_both = axes
        else:
            fig, ax_both = plt.subplots(1, 1, figsize=(10, 10))
        
        # Ground truth
        if show_gt:
            img_gt = img_rgb.copy()
            for ann in annotations:
                x, y, w, h = ann["bbox"]
                cat_name = self.category_lookup.get(ann["category_id"], 'unknown')
                cv2.rectangle(img_gt, (int(x), int(y)), (int(x+w), int(y+h)), 
                            (0, 255, 0), 2)
                cv2.putText(img_gt, cat_name, (int(x), int(y-5)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            ax_gt.imshow(img_gt)
            ax_gt.set_title(f'Ground Truth ({len(annotations)} objects)', fontsize=14)
            ax_gt.axis('off')
        
        # Predictions
        if show_pred:
            img_pred = img_rgb.copy()
            for pred in predictions:
                x1, y1, x2, y2 = pred['bbox']
                # Convert normalized coords to pixel coords
                x1, x2 = int(x1 * orig_w), int(x2 * orig_w)
                y1, y2 = int(y1 * orig_h), int(y2 * orig_h)
                
                cv2.rectangle(img_pred, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"{pred['class_name']} {pred['confidence']:.2f}"
                cv2.putText(img_pred, label, (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            ax_pred.imshow(img_pred)
            ax_pred.set_title(f'Predictions ({len(predictions)} objects, conf>{conf_threshold})', 
                            fontsize=14)
            ax_pred.axis('off')
        
        # Both overlaid
        img_both = img_rgb.copy()
        
        # Draw ground truth in green
        for ann in annotations:
            x, y, w, h = ann["bbox"]
            cv2.rectangle(img_both, (int(x), int(y)), (int(x+w), int(y+h)), 
                        (0, 255, 0), 2)
        
        # Draw predictions in red
        for pred in predictions:
            x1, y1, x2, y2 = pred['bbox']
            x1, x2 = int(x1 * orig_w), int(x2 * orig_w)
            y1, y2 = int(y1 * orig_h), int(y2 * orig_h)
            cv2.rectangle(img_both, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        ax_both.imshow(img_both)
        ax_both.set_title('Overlay (Green=GT, Red=Pred)', fontsize=14)
        ax_both.axis('off')
        
        plt.tight_layout()
        return fig, predictions
    
    def visualize_batch(self, data, img_dir, num_images=4, conf_threshold=0.5):
        """Visualize predictions for multiple images"""
        
        for i in range(min(num_images, len(data["images"]))):
            img_info = data["images"][i]
            img_path = Path(img_dir) / img_info["file_name"]
            
            # Get annotations for this image
            annotations = [ann for ann in data["annotations"] 
                          if ann["image_id"] == img_info["id"]]
            
            print(f"\n{'='*60}")
            print(f"Image {i+1}: {img_info['file_name']}")
            print(f"Ground truth objects: {len(annotations)}")
            
            fig, predictions = self.visualize_single_image(
                img_path, annotations, conf_threshold
            )
            
            print(f"Predicted objects: {len(predictions)}")
            if predictions:
                print("Predictions:")
                for j, pred in enumerate(predictions[:10], 1):  # Show top 10
                    print(f"  {j}. {pred['class_name']}: {pred['confidence']:.3f}")
            
            plt.show()


def main():
    """Main function to run visualization"""
    
    # Configuration
    MODEL_PATH = "/home/mithlesh/smartsense/floorplan_parser/checkpoints/latest.pth"
    DATA_PATH = "/home/mithlesh/smartsense/assets/train/annotations.coco.json"  # Update this
    IMG_DIR = "/home/mithlesh/smartsense/assets/train/"
    
    # Load your data
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)
    
    # Create category lookup
    category_lookup = {cat["id"]: cat["name"] for cat in data["categories"]}
    print(f"Categories: {category_lookup}")
    
    # Create visualizer
    visualizer = PredictionVisualizer(
        model_path=MODEL_PATH,
        category_lookup=category_lookup,
        device='cuda'
    )
    
    # Visualize multiple images
    visualizer.visualize_batch(
        data=data,
        img_dir=IMG_DIR,
        num_images=4,
        conf_threshold=0.01  # Lower threshold to see more predictions
    )
    
    # Or visualize a specific image
    print("\n" + "="*60)
    print("Visualizing specific image...")
    img_info = data["images"][3]
    img_path = Path(IMG_DIR) / img_info["file_name"]
    annotations = [ann for ann in data["annotations"] 
                  if ann["image_id"] == img_info["id"]]
    
    fig, preds = visualizer.visualize_single_image(
        img_path, annotations, conf_threshold=0.2
    )
    plt.show()


if __name__ == "__main__":
    main()