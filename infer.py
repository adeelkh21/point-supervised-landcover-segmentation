import os
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import models
from torchvision.models import ResNet50_Weights

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

IMG_SIZE = 512
NUM_CLASSES = 5
CLASS_NAMES = ['background', 'building', 'woodland', 'water', 'road']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

COLORS = np.array([
    [0, 0, 0],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
], dtype=np.uint8)

class LandCoverDataset(Dataset):
    def __init__(self, data_dir, split_file, img_size=IMG_SIZE):
        self.data_dir = data_dir
        self.img_size = img_size
        with open(split_file, 'r') as f:
            self.tile_ids = [line.strip() for line in f.readlines()]
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def __len__(self):
        return len(self.tile_ids)
    
    def __getitem__(self, idx):
        tile_id = self.tile_ids[idx]
        img_path = os.path.join(self.data_dir, 'output', f'{tile_id}.jpg')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img_original = img.copy()
        img = img.astype(np.float32) / 255.0
        
        mask_path = os.path.join(self.data_dir, 'output', f'{tile_id}_m.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        
        img = (img - self.mean) / self.std
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask.copy()).long()
        
        return img, mask, tile_id, img_original

def generate_sparse_labels(mask, points_per_class):
    H, W = mask.shape
    sparse_mask = torch.full((H, W), -1, dtype=torch.long)
    
    for cls in range(NUM_CLASSES):
        if cls == 0:
            continue
        cls_pixels = (mask == cls).nonzero(as_tuple=False)
        if len(cls_pixels) == 0:
            continue
        num_points = min(points_per_class, len(cls_pixels))
        if num_points > 0:
            indices = torch.randperm(len(cls_pixels))[:num_points]
            selected = cls_pixels[indices]
            sparse_mask[selected[:, 0], selected[:, 1]] = cls
    
    bg_pixels = (mask == 0).nonzero(as_tuple=False)
    if len(bg_pixels) > 0:
        num_bg = min(points_per_class * 2, len(bg_pixels))
        indices = torch.randperm(len(bg_pixels))[:num_bg]
        selected = bg_pixels[indices]
        sparse_mask[selected[:, 0], selected[:, 1]] = 0
    
    return sparse_mask

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

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        if pretrained:
            resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            resnet = models.resnet50(weights=None)
        
        self.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        for name, module in self.layer4.named_modules():
            if isinstance(module, nn.Conv2d):
                if module.stride == (2, 2):
                    module.stride = (1, 1)
                if module.kernel_size == (3, 3):
                    module.dilation = (2, 2)
                    module.padding = (2, 2)
        
        self.aspp = ASPP(2048, 256)
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
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
        x = self.layer0(x)
        low_level = self.layer1(x)
        x = self.layer2(low_level)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.aspp(x)
        x = F.interpolate(x, size=low_level.shape[-2:], mode='bilinear', align_corners=False)
        low_level = self.low_level_proj(low_level)
        x = torch.cat([x, low_level], dim=1)
        
        x = self.decoder(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return x

def compute_iou(pred, target, num_classes=NUM_CLASSES):
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

def load_model(model_path, device):
    model = DeepLabV3Plus(num_classes=NUM_CLASSES, pretrained=False).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint.get('points_per_class', 0), checkpoint.get('best_miou', 0)

def visualize_inference(dataset, model, points_per_class, indices, save_path=None):
    model.eval()
    num_samples = len(indices)
    
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4*num_samples))
    
    with torch.no_grad():
        for row, idx in enumerate(indices):
            img, mask, tile_id, img_original = dataset[idx]
            sparse_mask = generate_sparse_labels(mask, points_per_class)
            
            img_input = img.unsqueeze(0).to(DEVICE)
            output = model(img_input)
            pred = output.argmax(dim=1).squeeze().cpu()
            
            ious = compute_iou(pred, mask)
            miou = np.nanmean(ious)
            
            mask_np = mask.numpy()
            sparse_np = sparse_mask.numpy()
            pred_np = pred.numpy()
            
            mask_colored = COLORS[mask_np]
            pred_colored = COLORS[pred_np]
            sparse_colored = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            sparse_colored[sparse_np >= 0] = COLORS[sparse_np[sparse_np >= 0]]
            
            axes[row, 0].imshow(img_original)
            axes[row, 0].set_title(f'Original\n{tile_id[:15]}...')
            axes[row, 0].axis('off')
            
            axes[row, 1].imshow(mask_colored)
            axes[row, 1].set_title('Ground Truth')
            axes[row, 1].axis('off')
            
            axes[row, 2].imshow(sparse_colored)
            axes[row, 2].set_title(f'Sparse ({points_per_class} pts)')
            axes[row, 2].axis('off')
            
            axes[row, 3].imshow(pred_colored)
            axes[row, 3].set_title('Prediction')
            axes[row, 3].axis('off')
            
            class_text = '\n'.join([f'{CLASS_NAMES[c]}: {ious[c]:.2f}' if not np.isnan(ious[c]) else f'{CLASS_NAMES[c]}: N/A' for c in range(NUM_CLASSES)])
            axes[row, 4].text(0.1, 0.5, f'mIoU: {miou:.4f}\n\n{class_text}', 
                            fontsize=10, verticalalignment='center', family='monospace')
            axes[row, 4].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved to {save_path}')
    plt.close()

def select_diverse_images(dataset, num_samples=5):
    random.seed(42)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    images_by_classes = {2: [], 3: [], 4: [], 5: []}
    
    for idx in indices[:500]:
        _, mask, _, _ = dataset[idx]
        mask_np = mask.numpy()
        total_pixels = mask_np.size
        
        class_ratios = []
        for c in range(NUM_CLASSES):
            ratio = (mask_np == c).sum() / total_pixels
            class_ratios.append(ratio)
        
        num_classes = sum([1 for r in class_ratios if r > 0.05])
        
        if num_classes < 2:
            continue
        
        balance_score = 1.0 - np.std(class_ratios)
        has_rare_classes = (class_ratios[1] > 0.02 or class_ratios[4] > 0.01)
        quality_score = balance_score + (0.5 if has_rare_classes else 0)
        
        if num_classes in images_by_classes:
            images_by_classes[num_classes].append((idx, quality_score))
    
    for num_cls in images_by_classes:
        images_by_classes[num_cls].sort(key=lambda x: x[1], reverse=True)
    
    selected_indices = []
    for num_cls in [2, 3, 4, 5]:
        if images_by_classes[num_cls] and len(selected_indices) < num_samples:
            selected_indices.append(images_by_classes[num_cls][0][0])
    
    if len(selected_indices) < num_samples and images_by_classes[5]:
        for idx, _ in images_by_classes[5][1:]:
            if len(selected_indices) >= num_samples:
                break
            selected_indices.append(idx)
    
    return selected_indices

if __name__ == '__main__':
    print(f'Device: {DEVICE}')
    if DEVICE.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    
    DATA_DIR = '.'
    val_dataset = LandCoverDataset(DATA_DIR, os.path.join(DATA_DIR, 'val.txt'))
    print(f'Validation samples: {len(val_dataset)}')
    
    print('Selecting 5 diverse/complex images for inference...')
    fixed_indices = select_diverse_images(val_dataset, num_samples=5)
    print(f'Selected image indices: {fixed_indices}')
    
    model_configs = [
        ('models/resnet50_deeplabv3plus_15pts.pth', 15),
        ('models/resnet50_deeplabv3plus_40pts.pth', 40),
        ('models/resnet50_deeplabv3plus_70pts.pth', 70),
    ]
    
    for model_path, expected_pts in model_configs:
        if not os.path.exists(model_path):
            print(f'Model not found: {model_path}')
            continue
        
        print(f'\n{"="*60}')
        print(f'Loading model: {model_path}')
        model, points_per_class, best_miou = load_model(model_path, DEVICE)
        print(f'Points per class: {points_per_class}')
        print(f'Best mIoU: {best_miou:.4f}')
        
        save_path = f'inference_results_{points_per_class}pts.png'
        print(f'Running inference on 5 diverse validation images...')
        visualize_inference(val_dataset, model, points_per_class, fixed_indices, save_path=save_path)
    
    print(f'\n{"="*60}')
    print('Inference completed!')
    print('Generated files (5 images each, same scenes for comparison):')
    print('  - inference_results_15pts.png')
    print('  - inference_results_40pts.png')
    print('  - inference_results_70pts.png')
