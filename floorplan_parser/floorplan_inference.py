import torch
import cv2
import numpy as np
import easyocr
import json
import re
import os
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from floorplan_model import FloorplanDetector
from room_classifier import RoomTypeClassifier


class FloorplanParser:
    """End-to-end floorplan parser with OCR and hybrid room classification"""
    
    def __init__(self, checkpoint_path, confidence_threshold=0.5, device='cuda', 
                 use_ml_classifier=True, ml_classifier_path='./models/room_classifier.pkl',
                 ml_confidence_threshold=0.6):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        self.ml_confidence_threshold = ml_confidence_threshold
        
        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.categories = checkpoint['categories']
        self.num_classes = len(self.categories)
        
        self.model = FloorplanDetector(
            num_classes=self.num_classes,
            max_objects=checkpoint['config']['max_objects']
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded model from {checkpoint_path}")
        print(f"Categories: {self.categories}")
        
        # Initialize OCR
        print("Initializing EasyOCR...")
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        
        # Image transforms (same as validation)
        self.transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Room type keywords for classification
        self.room_keywords = {
            'bedroom': ['bed', 'bedroom', 'br', 'bdrm', 'master', 'guest'],
            'bathroom': ['bath', 'bathroom', 'wc', 'toilet', 'shower', 'washroom'],
            'kitchen': ['kitchen', 'pantry', 'cook'],
            'living': ['living', 'hall', 'lounge', 'family', 'drawing'],
            'dining': ['dining', 'dinner'],
            'balcony': ['balcony', 'terrace', 'deck'],
            'storage': ['store', 'storage', 'utility', 'closet'],
            'entrance': ['entrance', 'foyer', 'entry', 'porch'],
            'garage': ['garage', 'parking', 'carport'],
            'office': ['office', 'study', 'den', 'library']
        }
        
        # Initialize ML classifier (optional)
        self.use_ml_classifier = use_ml_classifier
        self.ml_classifier = None
        if use_ml_classifier:
            if os.path.exists(ml_classifier_path):
                try:
                    self.ml_classifier = RoomTypeClassifier()
                    self.ml_classifier.load(ml_classifier_path)
                    print(f"✓ Loaded ML room classifier from {ml_classifier_path}")
                except Exception as e:
                    print(f"⚠ Failed to load ML classifier: {e}")
                    print("  Falling back to keyword-only classification")
                    self.use_ml_classifier = False
            else:
                print(f"⚠ ML classifier not found at {ml_classifier_path}")
                print("  Using keyword-only classification")
                self.use_ml_classifier = False
        
        # Statistics tracking
        self.classification_stats = {
            'keyword_matches': 0,
            'ml_fallback': 0,
            'unclassified': 0
        }
    
    def preprocess_image(self, image_path):
        """Load and preprocess image"""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to 640x640 if needed
        if image.shape[:2] != (640, 640):
            image = cv2.resize(image, (640, 640))
        
        original_image = image.copy()
        
        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        return image_tensor, original_image
    
    def detect_boxes(self, image_tensor):
        """Run detection model to get bounding boxes"""
        with torch.no_grad():
            objectness, bboxes, class_logits = self.model(image_tensor)
        
        # Get predictions above threshold
        objectness = objectness.squeeze(0).cpu().numpy()
        bboxes = bboxes.squeeze(0).cpu().numpy()
        class_logits = class_logits.squeeze(0).cpu().numpy()
        
        # Filter by confidence
        mask = objectness > self.confidence_threshold
        filtered_bboxes = bboxes[mask]
        filtered_classes = np.argmax(class_logits[mask], axis=1)
        filtered_scores = objectness[mask]
        
        # Convert normalized coords to pixel coords (640x640)
        filtered_bboxes = filtered_bboxes * 640
        
        detections = []
        for bbox, cls, score in zip(filtered_bboxes, filtered_classes, filtered_scores):
            detections.append({
                'bbox': bbox.tolist(),
                'category_id': int(cls),
                'category_name': self.categories[int(cls)],
                'confidence': float(score)
            })
        
        return detections
    
    def extract_text_from_boxes(self, image, detections):
        """Run OCR on detected bounding boxes"""
        for det in detections:
            x_min, y_min, x_max, y_max = [int(coord) for coord in det['bbox']]
            
            # Clip to image bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(640, x_max)
            y_max = min(640, y_max)
            
            # Skip invalid boxes
            if x_max <= x_min or y_max <= y_min:
                det['text'] = ''
                continue
            
            # Crop region
            roi = image[y_min:y_max, x_min:x_max]
            
            # Skip very small regions
            if roi.shape[0] < 5 or roi.shape[1] < 5:
                det['text'] = ''
                continue
            
            # Run OCR
            try:
                result = self.reader.readtext(roi, detail=0)
                det['text'] = ' '.join(result).strip().lower()
            except Exception as e:
                print(f"OCR error: {e}")
                det['text'] = ''
        
        return detections
    
    def classify_room_type(self, text):
        """
        Classify room type using hybrid approach:
        1. Try keyword matching first (fast & precise)
        2. Fall back to ML classifier for ambiguous cases
        
        Returns:
            tuple: (room_type, classification_method)
                   room_type: str or None
                   classification_method: 'keyword', 'ml', or None
        """
        if not text:
            return None, None
        
        text_lower = text.lower()
        
        # STEP 1: Try keyword matching first
        for room_type, keywords in self.room_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    self.classification_stats['keyword_matches'] += 1
                    return room_type, 'keyword'
        
        # STEP 2: Fall back to ML classifier if available
        if self.use_ml_classifier and self.ml_classifier:
            try:
                room_type, confidence = self.ml_classifier.predict(text)
                if confidence >= self.ml_confidence_threshold:
                    self.classification_stats['ml_fallback'] += 1
                    return room_type, 'ml'
            except Exception as e:
                print(f"ML classifier error: {e}")
        
        # STEP 3: Unable to classify
        self.classification_stats['unclassified'] += 1
        return None, None
    
    def parse_area(self, text):
        """Extract area from dimension text (e.g., '3.5m x 4m' or '12 sq.ft')"""
        if not text:
            return None
        
        # Pattern 1: "X x Y" or "X*Y"
        pattern1 = r'(\d+\.?\d*)\s*[xX*×]\s*(\d+\.?\d*)'
        match = re.search(pattern1, text)
        if match:
            try:
                w = float(match.group(1))
                h = float(match.group(2))
                return round(w * h, 2)
            except:
                pass
        
        # Pattern 2: "XX sq" or "XX sqm" or "XX sq.ft"
        pattern2 = r'(\d+\.?\d*)\s*sq'
        match = re.search(pattern2, text)
        if match:
            try:
                return float(match.group(1))
            except:
                pass
        
        return None
    
    def generate_structured_output(self, detections):
        """Generate final JSON output from detections and OCR results"""
        
        # Reset stats for this image
        self.classification_stats = {
            'keyword_matches': 0,
            'ml_fallback': 0,
            'unclassified': 0
        }
        
        # Group by room types
        room_counts = {}
        room_details = []
        
        for det in detections:
            text = det.get('text', '')
            category = det['category_name']
            
            # Skip non-room-name categories for room detection
            if category != 'room_name':
                continue
            
            # Classify room type using hybrid approach
            room_type, method = self.classify_room_type(text)
            
            # Store classification method in detection for debugging
            det['classification_method'] = method
            det['room_type'] = room_type
            
            if room_type:
                # Count rooms
                if room_type not in room_counts:
                    room_counts[room_type] = 0
                room_counts[room_type] += 1
                
                # Try to extract area
                area = self.parse_area(text)
                
                room_details.append({
                    'label': room_type.title(),
                    'text': text,
                    'approx_area': area,
                    'classification_method': method
                })
        
        # Aggregate counts
        total_rooms = sum(room_counts.get(rt, 0) for rt in ['bedroom', 'living', 'dining', 'office'])
        halls = room_counts.get('living', 0)
        kitchens = room_counts.get('kitchen', 0)
        bathrooms = room_counts.get('bathroom', 0)
        
        # Consolidate room_details by type
        consolidated_details = {}
        for detail in room_details:
            label = detail['label']
            if label not in consolidated_details:
                consolidated_details[label] = {
                    'label': label,
                    'count': 0,
                    'approx_area': None
                }
            consolidated_details[label]['count'] += 1
            if detail['approx_area']:
                if consolidated_details[label]['approx_area']:
                    consolidated_details[label]['approx_area'] += detail['approx_area']
                else:
                    consolidated_details[label]['approx_area'] = detail['approx_area']
        
        output = {
            'rooms': total_rooms,
            'halls': halls,
            'kitchens': kitchens,
            'bathrooms': bathrooms,
            'rooms_detail': list(consolidated_details.values())
        }
        
        return output
    
    def parse(self, image_path, save_visualization=None):
        """
        Complete parsing pipeline
        
        Args:
            image_path: Path to floorplan image
            save_visualization: Optional path to save annotated image
        
        Returns:
            dict: Structured JSON output
        """
        print(f"Processing {image_path}...")
        
        # 1. Load and preprocess
        image_tensor, original_image = self.preprocess_image(image_path)
        
        # 2. Detect bounding boxes
        detections = self.detect_boxes(image_tensor)
        print(f"Detected {len(detections)} objects")
        
        # 3. Extract text via OCR
        detections = self.extract_text_from_boxes(original_image, detections)
        
        # 4. Generate structured output
        output = self.generate_structured_output(detections)
        
        # 5. Print classification statistics
        total_classified = self.classification_stats['keyword_matches'] + self.classification_stats['ml_fallback']
        if total_classified > 0:
            print(f"\nClassification Statistics:")
            print(f"  ✓ Keyword matches: {self.classification_stats['keyword_matches']}")
            if self.use_ml_classifier:
                print(f"  ✓ ML fallback: {self.classification_stats['ml_fallback']}")
            print(f"  ✗ Unclassified: {self.classification_stats['unclassified']}")
            
            keyword_pct = (self.classification_stats['keyword_matches'] / total_classified) * 100
            print(f"  → Keyword success rate: {keyword_pct:.1f}%")
            if self.use_ml_classifier and self.classification_stats['ml_fallback'] > 0:
                ml_pct = (self.classification_stats['ml_fallback'] / total_classified) * 100
                print(f"  → ML classifier helped: {ml_pct:.1f}%")
        
        # 6. Optional: Save visualization
        if save_visualization:
            self.visualize(original_image, detections, save_visualization)
        
        return output, detections
    
    def visualize(self, image, detections, save_path):
        """Draw bounding boxes and text on image with classification method"""
        vis_image = image.copy()
        
        for det in detections:
            x_min, y_min, x_max, y_max = [int(coord) for coord in det['bbox']]
            
            # Color based on classification method
            method = det.get('classification_method')
            if method == 'keyword':
                color = (0, 255, 0)  # Green for keyword
            elif method == 'ml':
                color = (255, 165, 0)  # Orange for ML
            else:
                color = (128, 128, 128)  # Gray for unclassified
            
            # Draw box
            cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), color, 2)
            
            # Draw label
            room_type = det.get('room_type', 'unknown')
            label = f"{room_type}: {det['text'][:15]}"
            if method:
                label += f" [{method[0]}]"  # Add [k] or [m] indicator
            
            cv2.putText(vis_image, label, (x_min, y_min - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        print(f"Saved visualization to {save_path}")
        print("  Legend: Green=[keyword], Orange=[ML], Gray=[unclassified]")
    
    def print_classification_summary(self):
        """Print overall classification performance"""
        total = sum(self.classification_stats.values())
        if total > 0:
            print("\n" + "="*60)
            print("CLASSIFICATION PERFORMANCE SUMMARY")
            print("="*60)
            for method, count in self.classification_stats.items():
                pct = (count / total) * 100
                print(f"{method:20s}: {count:4d} ({pct:5.1f}%)")
            print("="*60)


def main():
    # Initialize parser with ML classifier enabled
    parser = FloorplanParser(
        checkpoint_path='./checkpoints/best.pth',
        confidence_threshold=0.3,
        use_ml_classifier=True,  # Enable ML classifier
        ml_classifier_path='./models/room_classifier.pkl',
        ml_confidence_threshold=0.6  # Minimum confidence for ML predictions
    )
    
    # Test on a single image
    image_path = '/home/mithlesh/smartsense/assets/train/0_1_jpg.rf.0009e96605638d686a93c2b8be5d1db2.jpg'
    
    output, detections = parser.parse(
        image_path=image_path,
        save_visualization='output_visualization.jpg'
    )
    
    # Print results
    print("\n" + "="*50)
    print("STRUCTURED OUTPUT:")
    print(json.dumps(output, indent=2))
    print("="*50)
    
    print("\nDETECTIONS WITH OCR:")
    for i, det in enumerate(detections, 1):
        method = det.get('classification_method', 'N/A')
        room = det.get('room_type', 'unknown')
        print(f"{i}. Category: {det['category_name']}, Text: '{det['text']}', "
              f"Room: {room}, Method: {method}, Confidence: {det['confidence']:.3f}")


if __name__ == "__main__":
    main()