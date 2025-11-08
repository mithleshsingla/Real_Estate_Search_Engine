import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import json
from datetime import datetime

# Import our modules
from floorplan_dataloader import create_dataloaders
from floorplan_model import FloorplanDetector, DetectionLoss


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create directories
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['log_dir'], exist_ok=True)
        
        # Initialize dataloaders
        self.train_loader, self.val_loader, self.categories = create_dataloaders(
            image_dir=config['image_dir'],
            annotation_file=config['annotation_file'],
            batch_size=config['batch_size'],
            train_split=config['train_split'],
            num_workers=config['num_workers']
        )
        
        # Initialize model
        self.model = FloorplanDetector(
            num_classes=len(self.categories),
            max_objects=config['max_objects']
        ).to(self.device)
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
        
        # Loss and optimizer
        self.criterion = DetectionLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Tensorboard
        self.writer = SummaryWriter(config['log_dir'])
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.global_step = 0

    # Helper: safely convert tensor or float to float
    @staticmethod
    def to_float(x):
        return x.item() if isinstance(x, torch.Tensor) else float(x)
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_obj_loss = 0
        total_bbox_loss = 0
        total_class_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Prepare batch
            self.criterion.set_epoch(self.epoch)
            images = torch.stack([item['image'] for item in batch]).to(self.device)
            gt_bboxes = torch.stack([item['bboxes'] for item in batch]).to(self.device)
            gt_labels = torch.stack([item['labels'] for item in batch]).to(self.device)
            num_objects = torch.stack([item['num_objects'] for item in batch]).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred_objectness, pred_bboxes, pred_class_logits = self.model(images)
            
            # Compute loss
            loss, obj_loss, bbox_loss, class_loss = self.criterion(
                pred_objectness, pred_bboxes, pred_class_logits,
                gt_bboxes, gt_labels, num_objects
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Convert all to float safely
            loss_val = self.to_float(loss)
            obj_val = self.to_float(obj_loss)
            bbox_val = self.to_float(bbox_loss)
            cls_val = self.to_float(class_loss)
            
            # Accumulate losses
            total_loss += loss_val
            total_obj_loss += obj_val
            total_bbox_loss += bbox_val
            total_class_loss += cls_val
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_val:.4f}",
                'obj': f"{obj_val:.4f}",
                'bbox': f"{bbox_val:.4f}",
                'cls': f"{cls_val:.4f}"
            })
            
            # Log to tensorboard
            if batch_idx % 10 == 0:
                self.writer.add_scalar('train/total_loss', loss_val, self.global_step)
                self.writer.add_scalar('train/obj_loss', obj_val, self.global_step)
                self.writer.add_scalar('train/bbox_loss', bbox_val, self.global_step)
                self.writer.add_scalar('train/class_loss', cls_val, self.global_step)
            
            self.global_step += 1
        
        # Average losses
        n_batches = len(self.train_loader)
        return {
            'total_loss': total_loss / n_batches,
            'obj_loss': total_obj_loss / n_batches,
            'bbox_loss': total_bbox_loss / n_batches,
            'class_loss': total_class_loss / n_batches
        }
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        total_obj_loss = 0
        total_bbox_loss = 0
        total_class_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Prepare batch
                images = torch.stack([item['image'] for item in batch]).to(self.device)
                gt_bboxes = torch.stack([item['bboxes'] for item in batch]).to(self.device)
                gt_labels = torch.stack([item['labels'] for item in batch]).to(self.device)
                num_objects = torch.stack([item['num_objects'] for item in batch]).to(self.device)
                
                # Forward pass
                pred_objectness, pred_bboxes, pred_class_logits = self.model(images)
                
                # Compute loss
                loss, obj_loss, bbox_loss, class_loss = self.criterion(
                    pred_objectness, pred_bboxes, pred_class_logits,
                    gt_bboxes, gt_labels, num_objects
                )
                
                # Convert safely
                total_loss += self.to_float(loss)
                total_obj_loss += self.to_float(obj_loss)
                total_bbox_loss += self.to_float(bbox_loss)
                total_class_loss += self.to_float(class_loss)
        
        # Average losses
        n_batches = len(self.val_loader)
        return {
            'total_loss': total_loss / n_batches,
            'obj_loss': total_obj_loss / n_batches,
            'bbox_loss': total_bbox_loss / n_batches,
            'class_loss': total_class_loss / n_batches
        }
    
    def save_checkpoint(self, is_best=False):
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'categories': self.categories
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], 'latest.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config['checkpoint_dir'], 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint to {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Loaded checkpoint from {checkpoint_path} (epoch {self.epoch})")
    
    def train(self):
        print(f"Starting training for {self.config['num_epochs']} epochs")
        
        for epoch in range(self.config['num_epochs']):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            print(f"\nEpoch {epoch} - Train Loss: {train_metrics['total_loss']:.4f}")
            
            # Validate
            val_metrics = self.validate()
            print(f"Epoch {epoch} - Val Loss: {val_metrics['total_loss']:.4f}")
            
            # Log to tensorboard
            self.writer.add_scalar('epoch/train_loss', train_metrics['total_loss'], epoch)
            self.writer.add_scalar('epoch/val_loss', val_metrics['total_loss'], epoch)
            self.writer.add_scalar('epoch/learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['total_loss'])
            
            # Save checkpoint
            is_best = val_metrics['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['total_loss']
            
            if (epoch + 1) % self.config['save_every'] == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
        
        print(f"Training completed! Best validation loss: {self.best_val_loss:.4f}")
        self.writer.close()


def main():
    # Configuration
    config = {
        # Data
        'image_dir': '/home/mithlesh/smartsense/assets/train',
        'annotation_file': '/home/mithlesh/smartsense/assets/train/annotations.coco.json',
        'train_split': 0.8,
        
        # Model
        'max_objects': 50,
        'iou_threshold': 0.5,
        
        # Training
        'batch_size': 8,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_workers': 4,
        
        # Saving
        'checkpoint_dir': './checkpoints',
        'log_dir': './logs',
        'save_every': 5
    }
    
    # Create trainer
    trainer = Trainer(config)
    
    # Optional: Load checkpoint to resume training
    # trainer.load_checkpoint('./checkpoints/latest.pth')
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
