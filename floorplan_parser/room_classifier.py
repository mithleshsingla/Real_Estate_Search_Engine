import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import json


class RoomTypeClassifier:
    """
    ML-based room type classifier for ambiguous/noisy OCR text.
    Falls back when keyword matching fails.
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),
            min_df=1,
            lowercase=True
        )
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.label_encoder = {}
        self.is_trained = False
    
    def train(self, texts, labels):
        """
        Train the classifier on labeled text samples
        
        Args:
            texts: List of OCR text strings
            labels: List of room type labels (e.g., 'bedroom', 'kitchen')
        """
        # Encode labels
        unique_labels = sorted(set(labels))
        self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        self.reverse_label_encoder = {idx: label for label, idx in self.label_encoder.items()}
        
        encoded_labels = [self.label_encoder[label] for label in labels]
        
        # Vectorize text
        X = self.vectorizer.fit_transform(texts)
        y = np.array(encoded_labels)
        
        # Train classifier
        self.classifier.fit(X, y)
        self.is_trained = True
        
        print(f"✓ Trained classifier on {len(texts)} samples")
        print(f"  Room types: {unique_labels}")
    
    def predict(self, text):
        """
        Predict room type from OCR text
        
        Args:
            text: OCR text string
        
        Returns:
            room_type: Predicted room type
            confidence: Prediction confidence (0-1)
        """
        if not self.is_trained:
            return None, 0.0
        
        # Vectorize
        X = self.vectorizer.transform([text])
        
        # Predict
        pred = self.classifier.predict(X)[0]
        proba = self.classifier.predict_proba(X)[0]
        confidence = proba[pred]
        
        room_type = self.reverse_label_encoder[pred]
        
        return room_type, confidence
    
    def save(self, filepath):
        """Save trained classifier"""
        if not self.is_trained:
            raise ValueError("Classifier not trained yet")
        
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'label_encoder': self.label_encoder,
            'reverse_label_encoder': self.reverse_label_encoder
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Saved classifier to {filepath}")
    
    def load(self, filepath):
        """Load trained classifier"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        self.label_encoder = model_data['label_encoder']
        self.reverse_label_encoder = model_data['reverse_label_encoder']
        self.is_trained = True
        
        print(f"✓ Loaded classifier from {filepath}")


# ============================================================================
# TRAINING DATA GENERATION
# ============================================================================

def generate_training_data():
    """
    Generate synthetic training data for room classification.
    In practice, you should collect real OCR outputs from your floorplans.
    """
    
    # Common OCR variations and misspellings
    training_samples = [
        # Bedrooms
        ("bedroom", "bedroom"),
        ("bed room", "bedroom"),
        ("br", "bedroom"),
        ("bdrm", "bedroom"),
        ("master bedroom", "bedroom"),
        ("guest bedroom", "bedroom"),
        ("bed", "bedroom"),
        ("sleeping room", "bedroom"),
        ("bedrm", "bedroom"),
        ("bedrom", "bedroom"),
        
        # Bathrooms
        ("bathroom", "bathroom"),
        ("bath room", "bathroom"),
        ("bath", "bathroom"),
        ("wc", "bathroom"),
        ("toilet", "bathroom"),
        ("washroom", "bathroom"),
        ("restroom", "bathroom"),
        ("powder room", "bathroom"),
        ("bathrm", "bathroom"),
        ("bathrom", "bathroom"),
        
        # Kitchen
        ("kitchen", "kitchen"),
        ("kitch", "kitchen"),
        ("pantry", "kitchen"),
        ("cooking area", "kitchen"),
        ("kitchenette", "kitchen"),
        ("cook", "kitchen"),
        ("ktchn", "kitchen"),
        
        # Living/Hall
        ("living room", "living"),
        ("hall", "living"),
        ("living", "living"),
        ("lounge", "living"),
        ("family room", "living"),
        ("drawing room", "living"),
        ("sitting room", "living"),
        ("livin", "living"),
        
        # Dining
        ("dining room", "dining"),
        ("dining", "dining"),
        ("dinner room", "dining"),
        ("dinning", "dining"),
        
        # Balcony
        ("balcony", "balcony"),
        ("terrace", "balcony"),
        ("deck", "balcony"),
        ("patio", "balcony"),
        ("balcny", "balcony"),
        
        # Storage
        ("storage", "storage"),
        ("store", "storage"),
        ("utility", "storage"),
        ("closet", "storage"),
        ("storeroom", "storage"),
        
        # Office
        ("office", "office"),
        ("study", "office"),
        ("den", "office"),
        ("work room", "office"),
        
        # Garage
        ("garage", "garage"),
        ("parking", "garage"),
        ("car park", "garage"),
        
        # Entrance
        ("entrance", "entrance"),
        ("foyer", "entrance"),
        ("entry", "entrance"),
        ("porch", "entrance"),
    ]
    
    texts, labels = zip(*training_samples)
    return list(texts), list(labels)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def train_room_classifier(save_path='./models/room_classifier.pkl'):
    """Train and save the room classifier"""
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Generate or load training data
    texts, labels = generate_training_data()
    
    # Create and train classifier
    classifier = RoomTypeClassifier()
    classifier.train(texts, labels)
    
    # Save
    classifier.save(save_path)
    
    # Test on some examples
    test_texts = [
        "bedrm",
        "bathrm",
        "living",
        "ktchen",
        "balcny"
    ]
    
    print("\n" + "="*50)
    print("TEST PREDICTIONS:")
    print("="*50)
    for text in test_texts:
        room_type, confidence = classifier.predict(text)
        print(f"'{text}' → {room_type} (confidence: {confidence:.2f})")
    
    return classifier


# ============================================================================
# INTEGRATION WITH FLOORPLAN PARSER
# ============================================================================

class HybridRoomClassifier:
    """
    Combines keyword matching (fast, precise) with ML classifier (handles noise)
    """
    
    def __init__(self, ml_classifier_path='./models/room_classifier.pkl'):
        # Keyword rules (same as in inference)
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
        
        # Load ML classifier
        self.ml_classifier = RoomTypeClassifier()
        try:
            self.ml_classifier.load(ml_classifier_path)
            self.use_ml = True
        except:
            print("⚠ ML classifier not found, using keywords only")
            self.use_ml = False
    
    def classify(self, text, min_ml_confidence=0.6):
        """
        Classify room type using hybrid approach
        
        Args:
            text: OCR text
            min_ml_confidence: Minimum confidence for ML prediction
        
        Returns:
            room_type: Classified room type or None
            method: 'keyword' or 'ml' or None
        """
        if not text:
            return None, None
        
        text_lower = text.lower()
        
        # 1. Try keyword matching first (fast & precise)
        for room_type, keywords in self.room_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return room_type, 'keyword'
        
        # 2. Fall back to ML classifier for ambiguous cases
        if self.use_ml:
            room_type, confidence = self.ml_classifier.predict(text)
            if confidence >= min_ml_confidence:
                return room_type, 'ml'
        
        return None, None


def test_hybrid_classifier():
    """Test the hybrid classifier"""
    
    # Train ML classifier first
    classifier = train_room_classifier()
    
    # Create hybrid classifier
    hybrid = HybridRoomClassifier()
    
    # Test cases
    test_cases = [
        "bedroom",          # Clear keyword
        "bedrm",            # Misspelling - should use ML
        "kitchen area",     # Keyword
        "ktchn",            # Misspelling - should use ML
        "master suite",     # Keyword
        "xyz123",           # Noise - should return None
        "bath room",        # Keyword
        "bathrm"            # Misspelling - should use ML
    ]
    
    print("\n" + "="*50)
    print("HYBRID CLASSIFIER TEST:")
    print("="*50)
    for text in test_cases:
        room_type, method = hybrid.classify(text)
        print(f"'{text}' → {room_type or 'None'} (method: {method or 'N/A'})")


if __name__ == "__main__":
    test_hybrid_classifier()