import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class FloorplanDetector(nn.Module):
    """
    Floorplan detector optimized for small horizontal text boxes.
    Uses dense grid-based predictions to handle many objects (50-92 per image).
    """
    def __init__(self, num_classes=8, max_objects=150, grid_size=None):
        super(FloorplanDetector, self).__init__()
        self.num_classes = num_classes
        self.max_objects = max_objects
        
        # Auto-calculate grid size to get at least max_objects predictions
        # For 150 max objects, use at least 13x13=169 grid
        if grid_size is None:
            self.grid_size = int((max_objects ** 0.5) * 1.1) + 1
        else:
            self.grid_size = grid_size
        
        print(f"Using grid size: {self.grid_size}x{self.grid_size} = {self.grid_size**2} predictions")

        # Backbone
        resnet = models.resnet18(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # FPN
        self.fpn_conv1 = nn.Conv2d(512, 256, 1)
        self.fpn_conv2 = nn.Conv2d(256, 256, 1)
        self.fpn_conv3 = nn.Conv2d(128, 256, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv_reduce = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Detection heads (convolutional for spatial preservation)
        self.objectness_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1)  # 1 channel for objectness
        )
        
        self.bbox_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 4, 1)  # 4 channels for bbox offsets
        )
        
        self.class_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, 1)
        )
        
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for small object detection"""
        # Small random initialization for bbox offsets
        for m in self.bbox_head:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Standard init for objectness and class
        for head in [self.objectness_head, self.class_head]:
            for m in head:
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, 0, 0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)
        
        # Backbone + FPN
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p5 = self.fpn_conv1(c5)
        p4 = self.fpn_conv2(c4) + self.upsample(p5)
        p3 = self.fpn_conv3(c3) + self.upsample(p4)
        features = self.conv_reduce(p3)
        
        # Resize to desired grid size
        features = F.interpolate(features, size=(self.grid_size, self.grid_size), 
                                mode='bilinear', align_corners=False)
        
        # Predict at each grid cell
        objectness_map = self.objectness_head(features)  # [B, 1, H, W]
        bbox_map = self.bbox_head(features)  # [B, 4, H, W]
        class_map = self.class_head(features)  # [B, num_classes, H, W]
        
        # Flatten spatial dimensions - now we get grid_size^2 predictions
        num_predictions = self.grid_size * self.grid_size
        objectness = torch.sigmoid(objectness_map).view(batch_size, num_predictions)
        bbox_offsets = bbox_map.view(batch_size, 4, num_predictions).permute(0, 2, 1)
        class_logits = class_map.view(batch_size, self.num_classes, num_predictions).permute(0, 2, 1)
        
        # Convert offsets to absolute coordinates
        bboxes = self._offsets_to_boxes(bbox_offsets)
        
        return objectness, bboxes, class_logits
    
    def _offsets_to_boxes(self, offsets):
        """
        Convert grid cell offsets to absolute box coordinates.
        Optimized for small horizontal text boxes.
        offsets: [B, num_predictions, 4] where 4 = [dx, dy, dw, dh]
        """
        batch_size = offsets.size(0)
        device = offsets.device
        
        # Create grid cell centers
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, self.grid_size, device=device),
            torch.linspace(0, 1, self.grid_size, device=device),
            indexing='ij'
        )
        grid_centers = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)  # [num_predictions, 2]
        grid_centers = grid_centers.unsqueeze(0).expand(batch_size, -1, -1)  # [B, num_predictions, 2]
        
        # Parse offsets
        dx_dy = torch.tanh(offsets[..., :2]) * 0.5  # Limit to ±0.5 grid cells
        
        # Separate width and height with different constraints
        # Width: can be larger (up to 20% of image)
        # Height: much smaller (up to 5% of image) for horizontal text
        dw = torch.sigmoid(offsets[..., 2:3]) * 0.10  # Max 15% width
        dh = torch.sigmoid(offsets[..., 3:4]) * 0.025  # Max 2.5% height (small!)

        # Compute box centers
        centers = grid_centers + dx_dy / self.grid_size
        
        # Compute box corners with aspect ratio constraint
        half_w = dw / 2
        half_h = dh / 2
        
        x1 = (centers[..., 0:1] - half_w).clamp(0, 1)
        y1 = (centers[..., 1:2] - half_h).clamp(0, 1)
        x2 = (centers[..., 0:1] + half_w).clamp(0, 1)
        y2 = (centers[..., 1:2] + half_h).clamp(0, 1)
        
        # Ensure minimum size
        x2 = torch.max(x2, x1 + 0.01)  # Min 1% width
        y2 = torch.max(y2, y1 + 0.005)  # Min 0.5% height
        
        boxes = torch.cat([x1, y1, x2, y2], dim=-1)
        return boxes


def compute_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes"""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    x_min = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    y_min = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    x_max = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    y_max = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = (x_max - x_min).clamp(min=0) * (y_max - y_min).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / (union + 1e-6)
    return iou


def compute_giou(boxes1, boxes2):
    """Compute Generalized IoU (better for small/non-overlapping boxes)"""
    # Standard IoU computation
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    x_min = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    y_min = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    x_max = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    y_max = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    
    inter = (x_max - x_min).clamp(min=0) * (y_max - y_min).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / (union + 1e-6)
    
    # Enclosing box
    x1_min = torch.min(boxes1[:, None, 0], boxes2[None, :, 0])
    y1_min = torch.min(boxes1[:, None, 1], boxes2[None, :, 1])
    x2_max = torch.max(boxes1[:, None, 2], boxes2[None, :, 2])
    y2_max = torch.max(boxes1[:, None, 3], boxes2[None, :, 3])
    
    enclosing_area = (x2_max - x1_min) * (y2_max - y1_min)
    
    giou = iou - (enclosing_area - union) / (enclosing_area + 1e-6)
    return giou


class DetectionLoss(nn.Module):
    """
    Improved loss for dense small object detection (50-92 objects per image).
    Includes aspect ratio loss to encourage horizontal boxes.
    """
    def __init__(self, num_classes=8, max_objects=150, alpha=0.25, gamma=2.0, w_aspect=1):
        super(DetectionLoss, self).__init__()
        self.num_classes = num_classes
        self.max_objects = max_objects
        self.alpha = alpha
        self.gamma = gamma
        self.w_aspect = w_aspect  # Weight for aspect ratio loss
        self.current_epoch = 0
        
    def set_epoch(self, epoch):
        """Update epoch for curriculum learning"""
        self.current_epoch = epoch
        
    def focal_loss(self, pred, target):
        """Focal loss for handling class imbalance"""
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        p_t = pred * target + (1 - pred) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * bce
        return focal_loss.mean()

    def forward(self, pred_objectness, pred_bboxes, pred_class_logits, 
                gt_bboxes, gt_labels, num_objects):
        batch_size = pred_objectness.size(0)
        num_predictions = pred_objectness.size(1)
        device = pred_objectness.device

        total_obj_loss = 0.0
        total_bbox_loss = 0.0
        total_class_loss = 0.0
        total_matched_boxes = 0

        for b in range(batch_size):
            n_obj = num_objects[b].item()
            
            if n_obj == 0:
                # No objects - only objectness loss
                target_obj = torch.zeros_like(pred_objectness[b])
                total_obj_loss += self.focal_loss(pred_objectness[b], target_obj)
                continue
            
            gt_boxes = gt_bboxes[b, :n_obj]
            gt_lbls = gt_labels[b, :n_obj]
            pred_boxes = pred_bboxes[b]
            
            # Compute matching cost matrix
            # 1. Center distance cost
            gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2  # [n_obj, 2]
            pred_centers = (pred_boxes[:, :2] + pred_boxes[:, 2:]) / 2  # [num_pred, 2]
            dist_matrix = torch.cdist(pred_centers, gt_centers, p=2)  # [num_pred, n_obj]
            
            # 2. GIoU cost
            giou_matrix = compute_giou(pred_boxes, gt_boxes)  # [num_pred, n_obj]
            
            # 3. Classification cost (for matched predictions)
            pred_probs = F.softmax(pred_class_logits[b], dim=-1)  # [num_pred, num_classes]
            cls_cost = -pred_probs[:, gt_lbls]  # [num_pred, n_obj]
            
            # Normalize costs
            max_dist = torch.sqrt(torch.tensor(2.0, device=device))
            norm_dist = dist_matrix / max_dist
            
            # Combined cost: distance + giou + classification
            # Use adaptive weights based on epoch
            if self.current_epoch < 10:
                # Early training: focus on distance
                cost_matrix = 2.0 * norm_dist - giou_matrix + 0.5 * cls_cost
            else:
                # Later training: balance all costs
                cost_matrix = norm_dist - giou_matrix + cls_cost
            
            # Simple greedy assignment (optimal for most cases with many objects)
            # For each GT box, find the best prediction
            min_cost, assigned_pred_idx = cost_matrix.min(dim=0)  # [n_obj]
            
            # Create targets
            target_obj = torch.zeros(num_predictions, device=device)
            
            # Use adaptive threshold based on number of objects
            # More objects -> more lenient threshold
            if self.current_epoch < 5:
                threshold = 1.5  # Very lenient initially
            elif self.current_epoch < 20:
                threshold = 1.2
            else:
                threshold = 1.0
            
            valid_matches = min_cost < threshold
            matched_pred_idx = assigned_pred_idx[valid_matches]
            matched_gt_idx = torch.arange(n_obj, device=device)[valid_matches]
            
            # Handle duplicate assignments (multiple GTs assigned to same prediction)
            # Keep only the best match for each prediction
            unique_pred_idx, inverse = torch.unique(matched_pred_idx, return_inverse=True)
            final_matched_pred = []
            final_matched_gt = []
            
            for i, pred_idx in enumerate(unique_pred_idx):
                # Find all GTs assigned to this prediction
                gt_indices = matched_gt_idx[inverse == i]
                # Pick the GT with lowest cost
                costs = min_cost[matched_gt_idx == gt_indices[0]]
                if len(gt_indices) > 1:
                    best_gt_local_idx = cost_matrix[pred_idx, gt_indices].argmin()
                    best_gt = gt_indices[best_gt_local_idx]
                else:
                    best_gt = gt_indices[0]
                
                final_matched_pred.append(pred_idx)
                final_matched_gt.append(best_gt)
            
            if len(final_matched_pred) > 0:
                matched_pred_idx = torch.tensor(final_matched_pred, device=device, dtype=torch.long)
                matched_gt_idx = torch.tensor(final_matched_gt, device=device, dtype=torch.long)
                
                target_obj[matched_pred_idx] = 1.0
                total_matched_boxes += len(matched_pred_idx)
                
                # Bbox loss (GIoU loss)
                matched_pred_boxes = pred_boxes[matched_pred_idx]
                matched_gt_boxes = gt_boxes[matched_gt_idx]
                
                giou = compute_giou(matched_pred_boxes, matched_gt_boxes).diag()
                giou_loss = (1 - giou).mean()
                total_bbox_loss += giou_loss
                
                # Aspect ratio loss - encourage horizontal boxes
                pred_widths = matched_pred_boxes[:, 2] - matched_pred_boxes[:, 0]
                pred_heights = matched_pred_boxes[:, 3] - matched_pred_boxes[:, 1]
                gt_widths = matched_gt_boxes[:, 2] - matched_gt_boxes[:, 0]
                gt_heights = matched_gt_boxes[:, 3] - matched_gt_boxes[:, 1]
                
                # Predict aspect ratios should match GT aspect ratios
                pred_aspect = pred_widths / (pred_heights + 1e-6)
                gt_aspect = gt_widths / (gt_heights + 1e-6)
                aspect_loss = F.smooth_l1_loss(pred_aspect, gt_aspect)
                total_bbox_loss += self.w_aspect * aspect_loss
                
                # Class loss
                matched_pred_logits = pred_class_logits[b][matched_pred_idx]
                matched_gt_labels = gt_lbls[matched_gt_idx]
                total_class_loss += F.cross_entropy(matched_pred_logits, matched_gt_labels)
            
            # Objectness loss with focal loss
            total_obj_loss += self.focal_loss(pred_objectness[b], target_obj)

        # Average losses
        obj_loss = total_obj_loss / batch_size
        bbox_loss = total_bbox_loss / max(total_matched_boxes, 1)
        class_loss = total_class_loss / max(total_matched_boxes, 1)
        
        # Total loss with balanced weights
        total_loss = obj_loss + 1.0 * bbox_loss + class_loss
        
        # Logging
        avg_matched = total_matched_boxes / batch_size
        avg_gt = sum(num_objects).item() / batch_size
        match_rate = (avg_matched / avg_gt * 100) if avg_gt > 0 else 0
        
        print(f"Matched {total_matched_boxes}/{sum(num_objects).item()} boxes ({match_rate:.1f}%) | "
              f"Obj: {obj_loss.item():.4f}, BBox: {bbox_loss.item():.4f}, Cls: {class_loss.item():.4f}")
        
        return total_loss, obj_loss, bbox_loss, class_loss


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Configure for 50-92 objects per image
    max_objects = 150  # Buffer above max 92
    model = FloorplanDetector(num_classes=8, max_objects=max_objects).to(device)
    
    x = torch.randn(2, 3, 640, 640, device=device)
    objectness, bboxes, class_logits = model(x)
    
    print(f"\nModel Configuration:")
    print(f"Max objects: {max_objects}")
    print(f"Grid size: {model.grid_size}x{model.grid_size}")
    print(f"Total predictions per image: {model.grid_size**2}")
    print(f"\nOutput shapes:")
    print(f"Objectness: {objectness.shape}")
    print(f"Bboxes: {bboxes.shape}")
    print(f"Class logits: {class_logits.shape}")
    print(f"\nBbox statistics:")
    print(f"Bbox range: [{bboxes.min().item():.3f}, {bboxes.max().item():.3f}]")
    
    # Analyze box sizes
    widths = bboxes[0, :, 2] - bboxes[0, :, 0]
    heights = bboxes[0, :, 3] - bboxes[0, :, 1]
    print(f"Width range: [{widths.min().item():.3f}, {widths.max().item():.3f}], mean: {widths.mean().item():.3f}")
    print(f"Height range: [{heights.min().item():.3f}, {heights.max().item():.3f}], mean: {heights.mean().item():.3f}")

    # Test with realistic scenario: 50-92 small horizontal boxes
    criterion = DetectionLoss(num_classes=13, max_objects=max_objects, w_aspect=1)
    criterion.set_epoch(0)  # Start of training
    
    # Create realistic small horizontal boxes (like text labels)
    gt_bboxes = torch.zeros(2, max_objects, 4, device=device)
    num_objects = torch.tensor([50, 75], device=device)  # Typical case
    
    for b in range(2):
        n_obj = num_objects[b].item()
        for i in range(n_obj):
            cx = torch.rand(1, device=device).item()
            cy = torch.rand(1, device=device).item()
            w = 0.08 + torch.rand(1, device=device).item() * 0.08  # 8-16% width
            h = 0.02 + torch.rand(1, device=device).item() * 0.02  # 2-4% height
            
            x1, y1 = max(0, cx - w/2), max(0, cy - h/2)
            x2, y2 = min(1, cx + w/2), min(1, cy + h/2)
            gt_bboxes[b, i] = torch.tensor([x1, y1, x2, y2], device=device)
    
    gt_labels = torch.randint(0, 8, (2, max_objects), device=device)
    
    print(f"\n{'='*60}")
    print("Testing with realistic data:")
    print(f"Batch 1: {num_objects[0]} objects, Batch 2: {num_objects[1]} objects")
    print(f"{'='*60}")
    
    loss, obj_loss, bbox_loss, class_loss = criterion(
        objectness, bboxes, class_logits,
        gt_bboxes, gt_labels, num_objects
    )
    print(f"\nTotal loss: {loss.item():.4f}")
    print(f"Component losses - Obj: {obj_loss.item():.4f}, BBox: {bbox_loss.item():.4f}, Cls: {class_loss.item():.4f}")