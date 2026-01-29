#!/usr/bin/env python3
# Point-Supervised Semantic Segmentation for LandCover.ai v1
# Using DeepLabV3+ with EfficientNet-B4 backbone, Focal Loss, and class-balanced loss

import os
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from collections import defaultdict
from torchvision import models
from torchvision.models import ResNet50_Weights, EfficientNet_B4_Weights

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Configuration
IMG_SIZE = 512  # larger size for better results
NUM_CLASSES = 5  # 0=background, 1=building, 2=woodland, 3=water, 4=road
CLASS_NAMES = ['background', 'building', 'woodland', 'water', 'road']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class weights for imbalanced data (inverse frequency based)
# background is most common, building/road are rare
CLASS_WEIGHTS = torch.tensor([0.5, 2.0, 1.0, 1.5, 2.5], dtype=torch.float32)

# Color map for visualization
COLORS = np.array([
    [0, 0, 0],       # background - black
    [255, 0, 0],     # building - red
    [0, 255, 0],     # woodland - green
    [0, 0, 255],     # water - blue
    [255, 255, 0],   # road - yellow
], dtype=np.uint8)


# ============== Dataset with Data Augmentation ==============
class LandCoverDataset(Dataset):
    def __init__(self, data_dir, split_file, img_size=IMG_SIZE, augment=False):
        self.data_dir = data_dir
        self.img_size = img_size
        self.augment = augment
        # Read tile IDs
        with open(split_file, 'r') as f:
            self.tile_ids = [line.strip() for line in f.readlines()]
        # ImageNet normalization for pretrained backbone
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def __len__(self):
        return len(self.tile_ids)
    
    def _augment(self, img, mask):
        # Random horizontal flip
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
        # Random vertical flip
        if random.random() > 0.5:
            img = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)
        # Random 90 degree rotation
        k = random.randint(0, 3)
        if k > 0:
            img = np.rot90(img, k).copy()
            mask = np.rot90(mask, k).copy()
        # Random brightness/contrast
        if random.random() > 0.5:
            alpha = 0.8 + random.random() * 0.4  # contrast
            beta = -0.1 + random.random() * 0.2  # brightness
            img = np.clip(alpha * img + beta, 0, 1)
        return img, mask
    
    def __getitem__(self, idx):
        tile_id = self.tile_ids[idx]
        # Load image
        img_path = os.path.join(self.data_dir, 'output', f'{tile_id}.jpg')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        
        # Load mask
        mask_path = os.path.join(self.data_dir, 'output', f'{tile_id}_m.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        
        # Apply augmentation
        if self.augment:
            img, mask = self._augment(img, mask)
        
        # Normalize with ImageNet stats for pretrained backbone
        img = (img - self.mean) / self.std
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask.copy()).long()
        
        return img, mask, tile_id


# ============== Sparse Point Label Generator ==============
def generate_sparse_labels(mask, points_per_class):
    # mask: (H, W) tensor with class labels 0-4
    # Returns sparse mask with -1 for unlabeled pixels
    H, W = mask.shape
    sparse_mask = torch.full((H, W), -1, dtype=torch.long)
    
    for cls in range(NUM_CLASSES):
        if cls == 0:  # skip background for point sampling
            continue
        # Find all pixels of this class
        cls_pixels = (mask == cls).nonzero(as_tuple=False)
        if len(cls_pixels) == 0:
            continue
        # Randomly sample points
        num_points = min(points_per_class, len(cls_pixels))
        if num_points > 0:
            indices = torch.randperm(len(cls_pixels))[:num_points]
            selected = cls_pixels[indices]
            sparse_mask[selected[:, 0], selected[:, 1]] = cls
    
    # Also sample some background points for balance
    bg_pixels = (mask == 0).nonzero(as_tuple=False)
    if len(bg_pixels) > 0:
        num_bg = min(points_per_class * 2, len(bg_pixels))  # more bg points
        indices = torch.randperm(len(bg_pixels))[:num_bg]
        selected = bg_pixels[indices]
        sparse_mask[selected[:, 0], selected[:, 1]] = 0
    
    return sparse_mask


def generate_sparse_labels_batch(masks, points_per_class):
    # masks: (B, H, W) tensor
    batch_size = masks.shape[0]
    sparse_masks = []
    for i in range(batch_size):
        sparse_masks.append(generate_sparse_labels(masks[i], points_per_class))
    return torch.stack(sparse_masks)


# ============== Class-Balanced Partial Cross Entropy Loss ==============
class PartialCELoss(nn.Module):
    def __init__(self, class_weights=None, ignore_index=-1):
        super().__init__()
        self.ignore_index = ignore_index
        self.class_weights = class_weights
    
    def forward(self, pred, target):
        # pred: (B, C, H, W), target: (B, H, W) with -1 for unlabeled
        if self.class_weights is not None:
            weights = self.class_weights.to(pred.device)
            self.ce = nn.CrossEntropyLoss(weight=weights, ignore_index=self.ignore_index, reduction='mean')
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='mean')
        return self.ce(pred, target)


# ============== ASPP Module for DeepLabV3+ ==============
class ASPPConv(nn.Module):
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class ASPPPooling(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        size = x.shape[-2:]
        x = self.conv(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch=256, dilations=[6, 12, 18]):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv2 = ASPPConv(in_ch, out_ch, dilations[0])
        self.conv3 = ASPPConv(in_ch, out_ch, dilations[1])
        self.conv4 = ASPPConv(in_ch, out_ch, dilations[2])
        self.pool = ASPPPooling(in_ch, out_ch)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.pool(x)
        x = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        return self.project(x)


# ============== DeepLabV3+ with ResNet-50 Backbone ==============
class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        # Load pretrained ResNet-50
        if pretrained:
            resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            resnet = models.resnet50(weights=None)
        
        # Encoder (ResNet-50 backbone)
        self.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1  # 256 channels, stride 4
        self.layer2 = resnet.layer2  # 512 channels, stride 8
        self.layer3 = resnet.layer3  # 1024 channels, stride 16
        self.layer4 = resnet.layer4  # 2048 channels, stride 32
        
        # Modify layer4 to use dilated convolutions (output stride 16)
        for name, module in self.layer4.named_modules():
            if isinstance(module, nn.Conv2d):
                if module.stride == (2, 2):
                    module.stride = (1, 1)
                if module.kernel_size == (3, 3):
                    module.dilation = (2, 2)
                    module.padding = (2, 2)
        
        # ASPP module
        self.aspp = ASPP(2048, 256)
        
        # Low-level feature projection
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )
    
    def forward(self, x):
        size = x.shape[-2:]
        # Encoder
        x = self.layer0(x)
        low_level = self.layer1(x)  # low-level features
        x = self.layer2(low_level)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # ASPP
        x = self.aspp(x)
        
        # Upsample and concatenate with low-level features
        x = F.interpolate(x, size=low_level.shape[-2:], mode='bilinear', align_corners=False)
        low_level = self.low_level_proj(low_level)
        x = torch.cat([x, low_level], dim=1)
        
        # Decoder
        x = self.decoder(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return x


# ============== DeepLabV3+ with EfficientNet-B4 Backbone ==============
class EfficientNetB4DeepLabV3Plus(nn.Module):
    """DeepLabV3+ with EfficientNet-B4 backbone.
    
    EfficientNet-B4 provides better feature extraction than ResNet-50 with:
    - Compound scaling (depth, width, resolution)
    - Better accuracy/efficiency trade-off
    - 19M parameters vs 25M for ResNet-50
    """
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        # Load pretrained EfficientNet-B4
        if pretrained:
            efficientnet = models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        else:
            efficientnet = models.efficientnet_b4(weights=None)
        
        # EfficientNet-B4 features structure:
        # features[0]: Conv stem (48 channels)
        # features[1]: MBConv blocks stage 1 (24 channels)
        # features[2]: MBConv blocks stage 2 (32 channels) - low-level features
        # features[3]: MBConv blocks stage 3 (56 channels)
        # features[4]: MBConv blocks stage 4 (112 channels)
        # features[5]: MBConv blocks stage 5 (160 channels)
        # features[6]: MBConv blocks stage 6 (272 channels)
        # features[7]: MBConv blocks stage 7 (448 channels)
        # features[8]: Conv head (1792 channels)
        
        self.features = efficientnet.features
        
        # Get channel dimensions from EfficientNet-B4
        self.low_level_channels = 32  # From stage 2
        self.high_level_channels = 1792  # From final conv
        
        # ASPP module for high-level features
        self.aspp = ASPP(self.high_level_channels, 256)
        
        # Low-level feature projection
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(self.low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )
    
    def forward(self, x):
        size = x.shape[-2:]
        
        # EfficientNet encoder
        # Stage 0-1: Initial conv and first MBConv
        x = self.features[0](x)  # stem
        x = self.features[1](x)  # stage 1
        
        # Stage 2: Low-level features (1/4 resolution)
        low_level = self.features[2](x)  # 32 channels
        
        # Stages 3-8: High-level features
        x = self.features[3](low_level)
        x = self.features[4](x)
        x = self.features[5](x)
        x = self.features[6](x)
        x = self.features[7](x)
        x = self.features[8](x)  # 1792 channels
        
        # ASPP
        x = self.aspp(x)
        
        # Upsample and concatenate with low-level features
        x = F.interpolate(x, size=low_level.shape[-2:], mode='bilinear', align_corners=False)
        low_level = self.low_level_proj(low_level)
        x = torch.cat([x, low_level], dim=1)
        
        # Decoder
        x = self.decoder(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return x


# ============== Simple U-Net (kept for comparison) ==============
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=NUM_CLASSES):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.bottleneck = ConvBlock(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = ConvBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)
        self.out_conv = nn.Conv2d(64, num_classes, 1)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


# ============== Evaluation Metrics ==============
def compute_iou(pred, target, num_classes=NUM_CLASSES):
    # pred: (B, H, W) predictions, target: (B, H, W) ground truth
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        if union > 0:
            ious.append((intersection / union).item())
        else:
            ious.append(float('nan'))
    return ious


def evaluate(model, dataloader, device):
    model.eval()
    class_ious = defaultdict(list)
    
    with torch.no_grad():
        for imgs, masks, _ in dataloader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            
            ious = compute_iou(preds, masks)
            for cls, iou in enumerate(ious):
                if not np.isnan(iou):
                    class_ious[cls].append(iou)
    
    # Compute mean IoU per class
    mean_ious = {}
    for cls in range(NUM_CLASSES):
        if class_ious[cls]:
            mean_ious[cls] = np.mean(class_ious[cls])
        else:
            mean_ious[cls] = 0.0
    
    miou = np.mean([v for v in mean_ious.values() if v > 0])
    return mean_ious, miou


# ============== Training ==============
def train_one_epoch(model, dataloader, criterion, optimizer, points_per_class, device):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    print_interval = max(1, num_batches // 5)  # Print 5 times per epoch
    
    for batch_idx, (imgs, masks, _) in enumerate(dataloader):
        imgs = imgs.to(device)
        masks = masks.to(device)
        
        # Generate sparse point labels from full masks
        sparse_masks = generate_sparse_labels_batch(masks, points_per_class).to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, sparse_masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Progress indicator every 20% of batches
        if (batch_idx + 1) % print_interval == 0 or batch_idx == 0:
            avg_loss = total_loss / (batch_idx + 1)
            progress = (batch_idx + 1) / num_batches * 100
            print(f'  Batch {batch_idx+1}/{num_batches} ({progress:.0f}%) - Loss: {avg_loss:.4f}', flush=True)
    
    return total_loss / len(dataloader)


def train_model(model, train_loader, val_loader, points_per_class, epochs=30, lr=1e-4, patience=10, 
                save_path=None):
    # Class-balanced loss (simple CE, not Focal)
    criterion = PartialCELoss(class_weights=CLASS_WEIGHTS, ignore_index=-1)
    
    # Different learning rates for backbone and head
    if hasattr(model, 'layer0'):  # ResNet DeepLabV3+
        backbone_params = list(model.layer0.parameters()) + list(model.layer1.parameters()) + \
                          list(model.layer2.parameters()) + list(model.layer3.parameters()) + \
                          list(model.layer4.parameters())
        head_params = list(model.aspp.parameters()) + list(model.low_level_proj.parameters()) + \
                      list(model.decoder.parameters())
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': lr * 0.1},  # lower lr for pretrained
            {'params': head_params, 'lr': lr}
        ], weight_decay=1e-4)
    else:  # UNet or other
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_miou = 0
    best_model_state = None
    history = {'train_loss': [], 'val_miou': []}
    epochs_without_improvement = 0  # for early stopping
    
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, points_per_class, DEVICE)
        scheduler.step()
        
        # Evaluate every 3 epochs for early stopping (more frequent)
        if (epoch + 1) % 3 == 0 or epoch == 0 or epoch == epochs - 1:
            class_ious, miou = evaluate(model, val_loader, DEVICE)
            history['val_miou'].append(miou)
            
            if miou > best_miou:
                best_miou = miou
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0  # reset counter
                
                # Save best model checkpoint
                if save_path:
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': best_model_state,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_miou': best_miou,
                        'class_ious': class_ious,
                        'points_per_class': points_per_class,
                    }, save_path)
                    print(f'  ** New best model saved to {save_path} **')
            else:
                epochs_without_improvement += 3  # we evaluate every 3 epochs
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | mIoU: {miou:.4f} | LR: {current_lr:.2e} | No improvement: {epochs_without_improvement}')
            
            # Early stopping check
            if epochs_without_improvement >= patience:
                print(f'\n*** Early stopping triggered! No improvement for {patience} epochs ***')
                break
        else:
            print(f'Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f}')
        
        history['train_loss'].append(train_loss)
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    class_ious, _ = evaluate(model, val_loader, DEVICE)
    
    # Save final best model
    if save_path and best_model_state is not None:
        torch.save({
            'epoch': epochs,
            'model_state_dict': best_model_state,
            'best_miou': best_miou,
            'class_ious': class_ious,
            'points_per_class': points_per_class,
        }, save_path)
        print(f'Final model saved to {save_path}')
    
    print(f'Training completed! Best mIoU: {best_miou:.4f}')
    return history, best_miou, class_ious


# ============== Visualization ==============
def visualize_predictions(model, dataloader, points_per_class, num_samples=3, save_path=None):
    model.eval()
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    sample_count = 0
    with torch.no_grad():
        for imgs, masks, tile_ids in dataloader:
            for i in range(min(len(imgs), num_samples - sample_count)):
                img = imgs[i]
                mask = masks[i]
                
                # Generate sparse labels
                sparse_mask = generate_sparse_labels(mask, points_per_class)
                
                # Get prediction
                img_input = img.unsqueeze(0).to(DEVICE)
                output = model(img_input)
                pred = output.argmax(dim=1).squeeze().cpu()
                
                # Convert to numpy for visualization
                img_np = img.permute(1, 2, 0).numpy()
                mask_np = mask.numpy()
                sparse_np = sparse_mask.numpy()
                pred_np = pred.numpy()
                
                # Create colored masks
                mask_colored = COLORS[mask_np]
                pred_colored = COLORS[pred_np]
                
                # Sparse mask visualization (show labeled points)
                sparse_colored = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                sparse_colored[sparse_np >= 0] = COLORS[sparse_np[sparse_np >= 0]]
                
                # Plot
                row = sample_count
                axes[row, 0].imshow(img_np)
                axes[row, 0].set_title(f'Image: {tile_ids[i][:20]}...')
                axes[row, 0].axis('off')
                
                axes[row, 1].imshow(mask_colored)
                axes[row, 1].set_title('Ground Truth')
                axes[row, 1].axis('off')
                
                axes[row, 2].imshow(sparse_colored)
                axes[row, 2].set_title(f'Sparse Labels ({points_per_class} pts/class)')
                axes[row, 2].axis('off')
                
                axes[row, 3].imshow(pred_colored)
                axes[row, 3].set_title('Prediction')
                axes[row, 3].axis('off')
                
                sample_count += 1
                if sample_count >= num_samples:
                    break
            
            if sample_count >= num_samples:
                break
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_experiment_results(results):
    points = list(results.keys())
    mious = [results[p]['miou'] for p in points]
    
    plt.figure(figsize=(10, 5))
    
    # Plot mIoU vs points per class
    plt.subplot(1, 2, 1)
    plt.plot(points, mious, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Points per Class')
    plt.ylabel('Mean IoU')
    plt.title('mIoU vs Number of Point Labels')
    plt.grid(True)
    
    # Plot per-class IoU for each experiment
    plt.subplot(1, 2, 2)
    x = np.arange(NUM_CLASSES)
    width = 0.25
    
    for i, pts in enumerate(points):
        class_ious = results[pts]['class_ious']
        ious = [class_ious.get(c, 0) for c in range(NUM_CLASSES)]
        plt.bar(x + i*width, ious, width, label=f'{pts} points')
    
    plt.xlabel('Class')
    plt.ylabel('IoU')
    plt.title('Per-Class IoU')
    plt.xticks(x + width, CLASS_NAMES, rotation=45)
    plt.legend()
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('experiment_results.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============== Main Experiment ==============
def run_experiments(data_dir, points_list=[15, 40, 70], epochs=60, batch_size=16):
    print(f'Using device: {DEVICE}')
    if DEVICE.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'Running experiments with points per class: {points_list}')
    print(f'Model: DeepLabV3+ with ResNet-50 backbone (pretrained)')
    print(f'Loss: Class-Balanced Partial Cross Entropy')
    print(f'Epochs: {epochs} (with early stopping, patience=10)')
    print('='*60)
    
    # Create datasets (no data augmentation for faster training)
    train_dataset = LandCoverDataset(data_dir, os.path.join(data_dir, 'train.txt'), augment=False)
    val_dataset = LandCoverDataset(data_dir, os.path.join(data_dir, 'val.txt'), augment=False)
    
    # USE FULL TRAINING SET (no subsampling)
    print(f'Training samples: {len(train_dataset)} (FULL DATASET)')
    print(f'Validation samples: {len(val_dataset)}')
    
    # Use multiple workers for faster data loading
    num_workers = 4 if DEVICE.type == 'cuda' else 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    results = {}
    
    for points_per_class in points_list:
        print(f'\n{"="*60}')
        print(f'Training with {points_per_class} points per class')
        print('='*60)
        
        # Initialize DeepLabV3+ with pretrained ResNet-50 backbone
        model = DeepLabV3Plus(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Total parameters: {total_params/1e6:.1f}M')
        print(f'Trainable parameters: {trainable_params/1e6:.1f}M')
        
        # Model save path
        model_save_path = f'models/resnet50_deeplabv3plus_{points_per_class}pts.pth'
        
        # Train with model saving
        history, best_miou, class_ious = train_model(
            model, train_loader, val_loader, 
            points_per_class=points_per_class, 
            epochs=epochs,
            patience=10,
            save_path=model_save_path
        )
        
        results[points_per_class] = {
            'miou': best_miou,
            'class_ious': class_ious,
            'history': history,
            'model_path': model_save_path
        }
        
        # Print class-wise results
        print(f'\nFinal Results ({points_per_class} points/class):')
        print(f'  Mean IoU: {best_miou:.4f}')
        for cls in range(NUM_CLASSES):
            print(f'  {CLASS_NAMES[cls]}: {class_ious.get(cls, 0):.4f}')
        print(f'  Model saved: {model_save_path}')
        
        # Visualize predictions
        visualize_predictions(
            model, val_loader, points_per_class, 
            num_samples=3, 
            save_path=f'predictions_{points_per_class}pts.png'
        )
    
    # Summary
    print('\n' + '='*60)
    print('EXPERIMENT SUMMARY')
    print('='*60)
    print(f'{"Points/Class":<15} {"mIoU":<10} {"Model Path":<50}')
    print('-'*75)
    for pts in points_list:
        print(f'{pts:<15} {results[pts]["miou"]:.4f}     {results[pts]["model_path"]}')
    
    # Save results summary
    with open('experiment_summary.txt', 'w') as f:
        f.write('EXPERIMENT SUMMARY\n')
        f.write('='*75 + '\n')
        f.write(f'Model: DeepLabV3+ with ResNet-50 backbone\n')
        f.write(f'Loss: Class-Balanced Partial Cross Entropy\n')
        f.write(f'Image Size: 512x512\n')
        f.write(f'Training Samples: 7470 (Full Dataset)\n')
        f.write(f'Epochs: 60 (with early stopping)\n')
        f.write(f'Points per class: {points_list}\n')
        f.write('='*75 + '\n\n')
        
        for pts in points_list:
            f.write(f'\n{pts} Points/Class:\n')
            f.write(f'  Mean IoU: {results[pts]["miou"]:.4f}\n')
            for cls in range(NUM_CLASSES):
                f.write(f'  {CLASS_NAMES[cls]}: {results[pts]["class_ious"].get(cls, 0):.4f}\n')
            f.write(f'  Model: {results[pts]["model_path"]}\n')
    
    print('\nResults saved to experiment_summary.txt')
    
    # Plot comparison
    plot_experiment_results(results)
    
    return results


# ============== Entry Point ==============
if __name__ == '__main__':
    # Set data directory (adjust path as needed)
    DATA_DIR = '.'  # Current directory, assumes output/ folder exists
    
    # Check if output folder exists
    if not os.path.exists(os.path.join(DATA_DIR, 'output')):
        print('ERROR: output/ folder not found!')
        print('Please run split.py first to generate tiled images and masks.')
        print('Command: python split.py')
        exit(1)
    
    # Run experiments with different point label counts
    # Using 60 epochs with early stopping (patience=10), batch size 16
    # Full dataset (7470 samples), points [15, 40, 70]
    # Models will be saved to models/ directory
    results = run_experiments(
        data_dir=DATA_DIR,
        points_list=[15, 40, 70],
        epochs=60,
        batch_size=16
    )
