# 🏗️ Floorplan Parser

> **AI-powered floorplan image parser that extracts structured room information using computer vision and OCR**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
  - [Training](#1-training)
  - [Inference](#2-inference)
  - [Batch Processing](#3-batch-processing)
- [Output Format](#-output-format)
- [Configuration](#-configuration)
- [Customization](#-customization)
- [Troubleshooting](#-troubleshooting)
- [Performance](#-performance)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project trains a custom computer vision model to parse floorplan images and automatically extract structured information including:
- Number of bedrooms, bathrooms, kitchens, and halls
- Room types and their approximate areas
- Text annotations on floorplans

**Input:** Floorplan image (640×640 pixels)  
**Output:** Structured JSON with room counts, types, and dimensions

---

## ✨ Features

- 🔍 **Custom CNN Detector** - ResNet18 backbone with Feature Pyramid Network (FPN)
- 📝 **OCR Integration** - EasyOCR for text extraction from floorplans
- 🧠 **Hybrid Classification** - Keyword matching + ML fallback for room type identification
- 📊 **Structured Output** - Clean JSON format for easy integration
- 🚀 **GPU Accelerated** - Fast training and inference
- 📈 **TensorBoard Logging** - Real-time training monitoring
- 🔄 **Batch Processing** - Process multiple floorplans at once
- 🎨 **Visualization** - Annotated output images with bounding boxes

---

## 🏛️ Architecture

```
┌─────────────────┐
│ Floorplan Image │
│   (640×640)     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Detection Model        │
│  (ResNet18 + FPN)       │
│  - Detect text boxes    │
│  - Classify categories  │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  OCR Engine             │
│  (EasyOCR)              │
│  - Extract text content │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Room Classifier        │
│  (Hybrid: Keywords+ML)  │
│  - Identify room types  │
│  - Parse dimensions     │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Structured JSON Output │
│  {rooms: 3, halls: 1,   │
│   kitchens: 1, ...}     │
└─────────────────────────┘
```

### Model Components

1. **Detection Model** (`floorplan_model.py`)
   - Backbone: ResNet18 (pretrained on ImageNet)
   - Feature extractor: Feature Pyramid Network (FPN)
   - Output: Bounding boxes + category classifications
   - ~11M parameters

2. **OCR Engine** (EasyOCR)
   - Reads text from detected regions
   - Supports English language
   - GPU-accelerated

3. **Room Classifier** (`room_classifier.py`)
   - Primary: Keyword matching (fast, precise)
   - Fallback: ML classifier (TF-IDF + Random Forest)
   - Handles OCR errors and variations

---

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 10GB+ disk space

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/floorplan-parser.git
cd floorplan-parser
```

### Step 2: Create Virtual Environment

```bash
python -m venv floorplan_env
source floorplan_env/bin/activate  # On Windows: floorplan_env\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install opencv-python numpy pillow albumentations easyocr tqdm tensorboard scikit-learn
```

Or install all at once:

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

---

## 📁 Project Structure

```
floorplan-parser/
├── floorplan_dataloader.py      # COCO data loading & augmentation
├── floorplan_model.py            # Detection model architecture
├── floorplan_trainer.py          # Training script
├── floorplan_inference.py        # Inference pipeline with OCR
├── room_classifier.py            # ML-based room type classifier
├── complete_pipeline.py          # Unified CLI interface
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── checkpoints/                  # Model checkpoints
│   ├── best.pth                 # Best validation model
│   └── latest.pth               # Latest checkpoint
│
├── logs/                         # TensorBoard logs
│   └── events.out.tfevents.*
│
├── models/                       # Trained classifiers
│   └── room_classifier.pkl      # ML room classifier
│
└── results/                      # Inference outputs
    ├── image1.json              # Structured output
    ├── image1_viz.jpg           # Annotated image
    └── batch_summary.json       # Batch results
```

---

## 🚀 Quick Start

### Data Preparation

Your data should be organized as:

```
/path/to/dataset/
├── image1.jpg                    # 640×640 floorplan images
├── image2.jpg
├── ...
└── annotations.coco.json         # COCO format annotations
```

**COCO annotations format:**
```json
{
  "images": [
    {
      "id": 0,
      "file_name": "image1.jpg",
      "height": 640,
      "width": 640
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 0,
      "category_id": 5,
      "bbox": [100, 200, 150, 250]
    }
  ],
  "categories": [
    {"id": 0, "name": "room_name"},
    {"id": 1, "name": "clg"},
    {"id": 2, "name": "floor_name"},
    ...
  ]
}
```

### Training in 3 Steps

```bash
# 1. Train detection model
python complete_pipeline.py --mode train \
    --data_dir /path/to/dataset \
    --annotation_file /path/to/annotations.coco.json \
    --epochs 100 \
    --batch_size 8

# 2. (Optional) Train ML room classifier
python complete_pipeline.py --mode train_classifier

# 3. Run inference
python complete_pipeline.py --mode infer \
    --image /path/to/test_image.jpg \
    --checkpoint ./checkpoints/best.pth
```

---

## 📖 Usage

### 1. Training

#### Basic Training

```bash
python floorplan_trainer.py
```

This uses default configuration. The model will:
- Train for 100 epochs
- Save checkpoints every 5 epochs
- Use batch size of 8
- Save best model based on validation loss

#### Custom Training

Edit configuration in `floorplan_trainer.py`:

```python
config = {
    'image_dir': '/your/path/here',
    'annotation_file': '/your/annotations.coco.json',
    'batch_size': 8,          # Reduce if OOM
    'num_epochs': 100,        # Increase for better convergence
    'learning_rate': 1e-4,    # Adjust if needed
    'max_objects': 50,        # Max detections per image
    'train_split': 0.8,       # 80% train, 20% val
}
```

Or use the unified CLI:

```bash
python complete_pipeline.py --mode train \
    --data_dir /path/to/data \
    --annotation_file /path/to/annotations.coco.json \
    --batch_size 8 \
    --epochs 100 \
    --lr 1e-4
```

#### Monitor Training

```bash
tensorboard --logdir=./logs
```

Then open http://localhost:6006 in your browser.

#### Resume Training

```python
# In floorplan_trainer.py, uncomment:
trainer.load_checkpoint('./checkpoints/latest.pth')
trainer.train()
```

---

### 2. Inference

#### Single Image

```bash
python complete_pipeline.py --mode infer \
    --image /path/to/floorplan.jpg \
    --checkpoint ./checkpoints/best.pth \
    --output_dir ./results
```

Or use the inference script directly:

```python
from floorplan_inference import FloorplanParser

parser = FloorplanParser(
    checkpoint_path='./checkpoints/best.pth',
    confidence_threshold=0.3  # Lower = more detections
)

output, detections = parser.parse(
    image_path='floorplan.jpg',
    save_visualization='output_viz.jpg'
)

print(output)  # Structured JSON
```

#### Adjust Detection Threshold

```python
parser = FloorplanParser(
    checkpoint_path='./checkpoints/best.pth',
    confidence_threshold=0.2  # Lower threshold = more detections
)
```

- `0.2-0.3`: More detections (may include false positives)
- `0.4-0.5`: Balanced (recommended)
- `0.6-0.7`: Fewer detections (higher precision)

---

### 3. Batch Processing

Process multiple floorplans at once:

```bash
python complete_pipeline.py --mode batch \
    --image_dir /path/to/floorplans/ \
    --checkpoint ./checkpoints/best.pth \
    --output_dir ./results
```

This will:
1. Process all `.jpg` images in the directory
2. Generate individual JSON files for each image
3. Create annotated visualizations
4. Save a summary in `batch_summary.json`

**Output structure:**
```
results/
├── image1.json              # Individual results
├── image1_viz.jpg
├── image2.json
├── image2_viz.jpg
└── batch_summary.json       # Combined results
```

---

## 📄 Output Format

### Structured JSON

```json
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
```

### Fields Explained

- `rooms`: Total count of bedrooms + living rooms + dining rooms + offices
- `halls`: Number of halls/living rooms
- `kitchens`: Number of kitchens
- `bathrooms`: Number of bathrooms
- `rooms_detail`: Detailed breakdown with areas (if dimensions found in text)

---

## ⚙️ Configuration

### Training Configuration

```python
config = {
    # Data paths
    'image_dir': '/path/to/images',
    'annotation_file': '/path/to/annotations.coco.json',
    'train_split': 0.8,                    # 80% train, 20% validation
    
    # Model architecture
    'max_objects': 50,                     # Maximum detections per image
    'iou_threshold': 0.5,                  # IoU threshold for matching
    
    # Training hyperparameters
    'batch_size': 8,                       # Batch size (reduce if OOM)
    'num_epochs': 100,                     # Training epochs
    'learning_rate': 1e-4,                 # Initial learning rate
    'weight_decay': 1e-5,                  # L2 regularization
    'num_workers': 4,                      # Data loading threads
    
    # Checkpointing
    'checkpoint_dir': './checkpoints',
    'log_dir': './logs',
    'save_every': 5                        # Save checkpoint every N epochs
}
```

### Inference Configuration

```python
parser = FloorplanParser(
    checkpoint_path='./checkpoints/best.pth',
    confidence_threshold=0.5,              # Detection confidence threshold
    device='cuda',                         # 'cuda' or 'cpu'
    use_ml_classifier=True,                # Enable ML fallback classifier
    ml_classifier_path='./models/room_classifier.pkl',
    ml_confidence_threshold=0.6            # ML prediction confidence
)
```

---

## 🎨 Customization

### 1. Add New Room Keywords

Edit `floorplan_inference.py`:

```python
self.room_keywords = {
    'bedroom': ['bed', 'bedroom', 'br', 'bdrm', 'master', 'guest'],
    'bathroom': ['bath', 'bathroom', 'wc', 'toilet', 'washroom'],
    'kitchen': ['kitchen', 'pantry', 'cook'],
    # Add your custom keywords here
    'study': ['study', 'office', 'work room'],
    'patio': ['patio', 'outdoor', 'garden'],
}
```

### 2. Train Custom ML Classifier

Collect room text samples and train:

```python
from room_classifier import RoomTypeClassifier

# Prepare training data
texts = ['bedroom', 'bedrm', 'bed room', 'master bd', ...]
labels = ['bedroom', 'bedroom', 'bedroom', 'bedroom', ...]

# Train classifier
classifier = RoomTypeClassifier()
classifier.train(texts, labels)
classifier.save('./models/room_classifier.pkl')
```

### 3. Modify Detection Threshold

For more/fewer detections:

```python
# In floorplan_inference.py
parser = FloorplanParser(
    checkpoint_path='./checkpoints/best.pth',
    confidence_threshold=0.3  # Default: 0.5
)
```

### 4. Change Model Architecture

Edit `floorplan_model.py` to use different backbones:

```python
# Replace ResNet18 with ResNet50
resnet = models.resnet50(pretrained=True)
```

### 5. Custom Data Augmentation

Edit `floorplan_dataloader.py`:

```python
train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    # Add your custom augmentations
    A.GaussianBlur(p=0.2),
    A.ElasticTransform(p=0.2),
    ...
])
```

---

## 🔍 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
```bash
# Reduce batch size
python floorplan_trainer.py  # Edit config: batch_size=4 or 2

# Reduce max_objects
# Edit config: max_objects=30
```

#### 2. Poor Detection Accuracy

**Solutions:**
- Train for more epochs (100-150)
- Lower confidence threshold (0.2-0.3)
- Add more training data
- Check data quality and annotations

#### 3. OCR Not Reading Text

**Solutions:**
- Check image quality and resolution
- Enhance image contrast before OCR
- Try different OCR confidence thresholds
- Preprocess images (denoising, sharpening)

#### 4. Wrong Room Classification

**Solutions:**
- Add more keywords to `room_keywords` dictionary
- Train the ML classifier with your specific text patterns
- Review OCR output for systematic errors
- Lower `ml_confidence_threshold` to accept more ML predictions

#### 5. Model Not Converging

**Solutions:**
- Check data quality and annotation correctness
- Reduce learning rate (5e-5 or 1e-5)
- Use learning rate warmup
- Increase batch size if possible
- Train for more epochs

#### 6. Import Errors

**Error:** `ModuleNotFoundError: No module named 'X'`

**Solution:**
```bash
pip install -r requirements.txt
# or install specific package
pip install package_name
```

---

## 📊 Performance

### Expected Results (500 training images)

| Metric | Value |
|--------|-------|
| **Training Time** | 1-2 hours (100 epochs, single GPU) |
| **Detection mAP** | 70-85% |
| **OCR Accuracy** | 80-90% |
| **Room Classification** | 85-95% |
| **Inference Speed** | 0.5-1 sec/image (GPU) |
| **Model Size** | ~45 MB |

### Hardware Requirements

**Minimum:**
- GPU: NVIDIA GTX 1060 (6GB VRAM)
- RAM: 8GB
- Storage: 10GB

**Recommended:**
- GPU: NVIDIA RTX 3060 or better
- RAM: 16GB
- Storage: 20GB SSD

### Optimization Tips

1. **Faster Training:**
   - Use larger batch size (if GPU memory allows)
   - Increase `num_workers` for data loading
   - Use mixed precision training (FP16)

2. **Better Accuracy:**
   - Collect more training data (1000+ images)
   - Use stronger backbone (ResNet50)
   - Apply more data augmentation
   - Train longer (150-200 epochs)

3. **Faster Inference:**
   - Reduce `confidence_threshold` (fewer NMS operations)
   - Use TensorRT for deployment
   - Batch process multiple images

---

## 🤝 Contributing

We welcome contributions! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/floorplan-parser.git
cd floorplan-parser

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **PyTorch** - Deep learning framework
- **EasyOCR** - OCR engine
- **Albumentations** - Data augmentation library
- **ResNet** - Pre-trained backbone architecture

---

## 📧 Contact & Support

- **Issues:** [GitHub Issues](https://github.com/yourusername/floorplan-parser/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/floorplan-parser/discussions)
- **Email:** your.email@example.com

---

## 🗺️ Roadmap

- [ ] Support for more floorplan formats (PDF, DWG)
- [ ] Multi-language OCR support
- [ ] Web-based demo interface
- [ ] Mobile app for on-site scanning
- [ ] Integration with CAD software
- [ ] 3D floorplan generation

