import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class FloorplanDataset(Dataset):
    """Dataset for floorplan images with COCO annotations"""
    
    def __init__(self, image_dir, annotation_file, transform=None, max_objects=50):
        """
        Args:
            image_dir: Directory containing images
            annotation_file: Path to COCO format JSON
            transform: Albumentations transforms
            max_objects: Maximum number of objects per image (for padding)
        """
        self.image_dir = image_dir
        self.transform = transform
        self.max_objects = max_objects
        
        # Load COCO annotations
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        self.images = {img['id']: img for img in coco_data['images']}
        self.categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # Group annotations by image_id
        self.image_annotations = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_annotations:
                self.image_annotations[img_id] = []
            self.image_annotations[img_id].append(ann)
        
        self.image_ids = list(self.images.keys())
        print(f"Loaded {len(self.image_ids)} images with {len(coco_data['annotations'])} annotations")
        print(f"Categories: {self.categories}")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        
        # Load image
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get annotations for this image
        annotations = self.image_annotations.get(img_id, [])
        
        # Extract bboxes and labels
        bboxes = []
        labels = []
        for ann in annotations:
            # COCO format: [x, y, width, height]
            x, y, w, h = ann['bbox']
            # Convert to [x_min, y_min, x_max, y_max] and normalize
            x_min = x / 640.0
            y_min = y / 640.0
            x_max = (x + w) / 640.0
            y_max = (y + h) / 640.0
            
            # Clip to [0, 1]
            x_min = np.clip(x_min, 0, 1)
            y_min = np.clip(y_min, 0, 1)
            x_max = np.clip(x_max, 0, 1)
            y_max = np.clip(y_max, 0, 1)
            
            if x_max > x_min and y_max > y_min:  # Valid box
                bboxes.append([x_min, y_min, x_max, y_max])
                labels.append(ann['category_id'])
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=bboxes,
                labels=labels
            )
            image = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['labels']
        
        # Pad/truncate to max_objects
        num_objects = len(bboxes)
        if num_objects > self.max_objects:
            bboxes = bboxes[:self.max_objects]
            labels = labels[:self.max_objects]
            num_objects = self.max_objects
        
        # Create padded tensors
        padded_bboxes = torch.zeros(self.max_objects, 4)
        padded_labels = torch.zeros(self.max_objects, dtype=torch.long)
        
        if num_objects > 0:
            bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            padded_bboxes[:num_objects] = bboxes_tensor
            padded_labels[:num_objects] = labels_tensor
        
        # print(f"Image shape: {image.shape}")  # should be (3, 640, 640) tensor 
        # print(f"Bboxes after transform: {bboxes}")  # normalized boxes [N,4]

        return {
            'image': image,
            'bboxes': padded_bboxes,
            'labels': padded_labels,
            'num_objects': torch.tensor(num_objects, dtype=torch.long),
            'image_id': img_id
        }

def get_transforms(train=True):
    """Get augmentation transforms with explicit resize"""
    if train:
        return A.Compose([
            A.Resize(640, 640),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Rotate(limit=10, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='albumentations', label_fields=['labels'],
                                   min_visibility=0.3))  # discards boxes mostly cropped out
    else:
        return A.Compose([
            A.Resize(640, 640),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='albumentations', label_fields=['labels']))

def get_transforms_ol(train=True):
    """Get augmentation transforms"""
    if train:
        return A.Compose([
            A.Resize(640, 640),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Rotate(limit=10, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='albumentations', label_fields=['labels']))
    else:
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='albumentations', label_fields=['labels']))


def create_dataloaders(image_dir, annotation_file, batch_size=8, train_split=0.8, num_workers=4):
    """Create train and validation dataloaders"""
    
    # Create full dataset
    full_dataset = FloorplanDataset(
        image_dir=image_dir,
        annotation_file=annotation_file,
        transform=None
    )
    
    # Split into train and val
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    
    indices = torch.randperm(total_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create datasets with transforms
    train_dataset = FloorplanDataset(
        image_dir=image_dir,
        annotation_file=annotation_file,
        transform=get_transforms(train=True)
    )
    train_dataset.image_ids = [full_dataset.image_ids[i] for i in train_indices]
    
    val_dataset = FloorplanDataset(
        image_dir=image_dir,
        annotation_file=annotation_file,
        transform=get_transforms(train=False)
    )
    val_dataset.image_ids = [full_dataset.image_ids[i] for i in val_indices]
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda x: x  # Return list of dicts
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda x: x
    )
    
    print(f"Train set: {len(train_dataset)} images")
    print(f"Val set: {len(val_dataset)} images")
    
    return train_loader, val_loader, train_dataset.categories


if __name__ == "__main__":
    # Test the dataloader
    image_dir = "/home/mithlesh/smartsense/assets/train"
    annotation_file = "/home/mithlesh/smartsense/assets/train/annotations.coco.json"
    
    train_loader, val_loader, categories = create_dataloaders(
        image_dir=image_dir,
        annotation_file=annotation_file,
        batch_size=4
    )
    
    # Test loading a batch
    for batch in train_loader:
        print(f"Batch size: {len(batch)}")
        print(f"Image shape: {batch[0]['image'].shape}")
        print(f"Bboxes shape: {batch[0]['bboxes'].shape}")
        print(f"Labels shape: {batch[0]['labels'].shape}")
        print(f"Num objects: {batch[0]['num_objects']}")
        break