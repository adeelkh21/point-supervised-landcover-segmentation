# Land Cover Classification Dataset - LandCover.ai v1

## Overview

This repository contains the **LandCover.ai v1** dataset, which is designed for land cover semantic segmentation tasks. The dataset consists of high-resolution aerial imagery paired with corresponding segmentation masks. This data is pre-processed and split into training, validation, and test sets for machine learning model development.

---

## Dataset Structure

```
landcover.ai.v1/
│
├── images/                 # Directory containing aerial images (41 TIFF files)
├── masks/                  # Directory containing segmentation masks (41 TIFF files)
├── split.py                # Python script for data preprocessing and tiling
├── train.txt              # List of training sample IDs (7,471 samples)
├── val.txt                # List of validation sample IDs (1,603 samples)
├── test.txt               # List of test sample IDs (1,603 samples)
└── README.md              # This documentation file
```

---

## 📁 Data Components

### 1. **Images Directory** (`images/`)

Contains **41 large-scale aerial images** in TIFF format. These are high-resolution orthophotos captured from aerial surveys.

**File Format:** `.tif` (TIFF)
**Naming Convention:** Geographic coordinate-based naming (e.g., `M-33-20-D-c-4-2.tif`)

The naming follows a structured geographic tiling system commonly used in topographic mapping:
- **M/N:** Major grid zones
- **Numbers:** Hierarchical subdivision of the geographic area
- **Letters:** Further refinement of the location

**Examples:**
- `M-33-20-D-c-4-2.tif`
- `N-34-140-A-b-3-2.tif`
- `M-34-65-D-c-4-2.tif`

**Total Image Files:** 41

---

### 2. **Masks Directory** (`masks/`)

Contains **41 corresponding segmentation masks** in TIFF format. Each mask has the **same filename** as its corresponding image, ensuring perfect alignment.

**File Format:** `.tif` (TIFF)
**Purpose:** Pixel-wise annotations for land cover classification

Each pixel in a mask represents a specific land cover class (e.g., buildings, woodland, water, roads).

**Mask-Image Pairing:**
- `M-33-20-D-c-4-2.tif` (image) ↔ `M-33-20-D-c-4-2.tif` (mask)
- Every image has an exact matching mask with the same filename
- Masks have the **same spatial dimensions** as their corresponding images

**Total Mask Files:** 41

---

## 🔪 Data Split

The dataset is split into three subsets to enable proper training, validation, and testing of machine learning models:

### Training Set (`train.txt`)
- **Purpose:** Used to train the model
- **Total Samples:** 7,471 tiles
- **Percentage:** ~70% of the dataset
- Contains tile IDs like:
  - `M-33-20-D-c-4-2_0`
  - `M-33-20-D-c-4-2_1`
  - `M-33-20-D-c-4-2_10`

### Validation Set (`val.txt`)
- **Purpose:** Used to tune hyperparameters and monitor training progress
- **Total Samples:** 1,603 tiles
- **Percentage:** ~15% of the dataset
- Contains tile IDs like:
  - `M-33-20-D-c-4-2_101`
  - `M-33-20-D-c-4-2_103`
  - `M-33-20-D-d-3-3_1`

### Test Set (`test.txt`)
- **Purpose:** Used for final model evaluation (kept separate during training)
- **Total Samples:** 1,603 tiles
- **Percentage:** ~15% of the dataset
- Contains tile IDs like:
  - `M-33-20-D-c-4-2_105`
  - `M-33-20-D-c-4-2_110`
  - `M-33-20-D-c-4-2_117`

**Total Dataset Size:** 10,677 tiles

---

## 🔧 Data Processing Script (`split.py`)

This Python script is responsible for preprocessing the raw aerial images and masks by tiling them into smaller, manageable pieces.

### What the Script Does:

1. **Reads Raw Images:** Loads all `.tif` files from the `images/` and `masks/` directories
2. **Validates Pairing:** Ensures each image has a corresponding mask with matching dimensions
3. **Tiling Process:** Divides large images into smaller tiles of **512×512 pixels**
4. **Output Generation:** Saves processed tiles to an `output/` directory

### Key Parameters:

```python
IMGS_DIR = "./images"       # Source directory for images
MASKS_DIR = "./masks"       # Source directory for masks
OUTPUT_DIR = "./output"     # Destination directory for tiles
TARGET_SIZE = 512           # Tile dimensions (512×512 pixels)
```

### Tiling Logic:

- The script uses a **sliding window approach** with no overlap
- Starts from the top-left corner (0, 0) and moves across and down
- Only tiles that are **exactly 512×512 pixels** are saved (edge tiles may be discarded if smaller)

### Naming Convention for Tiles:

**Images:** `{original_name}_{tile_index}.jpg`  
**Masks:** `{original_name}_{tile_index}_m.png`

**Example:**
- Original: `M-33-20-D-c-4-2.tif` → Tiles: `M-33-20-D-c-4-2_0.jpg`, `M-33-20-D-c-4-2_1.jpg`, etc.
- Masks: `M-33-20-D-c-4-2_0_m.png`, `M-33-20-D-c-4-2_1_m.png`, etc.

### How to Run:

```bash
python split.py
```

**Requirements:**
- Python 3.x
- OpenCV (`cv2`)

**Output:**
- Creates an `output/` directory
- Generates thousands of 512×512 image-mask tile pairs
- Prints progress for each processed image

---

## 📊 Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Raw Images** | 41 files |
| **Total Raw Masks** | 41 files |
| **Total Tiles (after preprocessing)** | 10,677 tiles |
| **Training Tiles** | 7,471 (~70%) |
| **Validation Tiles** | 1,603 (~15%) |
| **Test Tiles** | 1,603 (~15%) |
| **Tile Size** | 512×512 pixels |
| **Image Format (Original)** | TIFF (.tif) |
| **Image Format (Tiles)** | JPEG (.jpg) |
| **Mask Format (Tiles)** | PNG (.png) |

---

## 🎯 Use Cases

This dataset is suitable for:

- **Semantic Segmentation:** Training models like U-Net, DeepLab, SegNet, etc.
- **Land Cover Classification:** Identifying different land cover types (buildings, forests, water bodies, agricultural land)
- **Remote Sensing Research:** Analyzing aerial imagery for environmental monitoring
- **Computer Vision Projects:** Experimenting with image segmentation techniques
- **Transfer Learning:** Pre-training models for similar geospatial tasks

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install opencv-python numpy
```

### Step 1: Prepare the Data

Run the preprocessing script to generate tiles:

```bash
python split.py
```

This will create an `output/` directory with all the tiled images and masks.

### Step 2: Load the Data

Use the split files to load your data:

```python
# Read training file IDs
with open('train.txt', 'r') as f:
    train_ids = [line.strip() for line in f.readlines()]

# Read validation file IDs
with open('val.txt', 'r') as f:
    val_ids = [line.strip() for line in f.readlines()]

# Read test file IDs
with open('test.txt', 'r') as f:
    test_ids = [line.strip() for line in f.readlines()]
```

### Step 3: Build Your Data Pipeline

```python
import cv2
import os

def load_image_and_mask(tile_id, output_dir='./output'):
    # Load image
    img_path = os.path.join(output_dir, f"{tile_id}.jpg")
    img = cv2.imread(img_path)
    
    # Load mask
    mask_path = os.path.join(output_dir, f"{tile_id}_m.png")
    mask = cv2.imread(mask_path)
    
    return img, mask

# Example usage
for tile_id in train_ids[:5]:
    image, mask = load_image_and_mask(tile_id)
    print(f"Loaded {tile_id}: Image shape = {image.shape}, Mask shape = {mask.shape}")
```

---

## 📝 File Format Details

### Split Files (train.txt, val.txt, test.txt)

- **Format:** Plain text, one tile ID per line
- **Content:** Base filenames without extensions
- **Example Lines:**
  ```
  M-33-20-D-c-4-2_0
  M-33-20-D-c-4-2_1
  M-33-20-D-d-3-3_106
  ```

### Image Files

- **Original Format:** `.tif` (TIFF) - Lossless, high-quality
- **Processed Format:** `.jpg` (JPEG) - Compressed, smaller file size
- **Color Space:** RGB (3 channels)
- **Dimensions:** Variable for originals; 512×512 for tiles

### Mask Files

- **Original Format:** `.tif` (TIFF)
- **Processed Format:** `.png` (PNG) - Lossless, preserves pixel values
- **Purpose:** Each pixel value represents a land cover class
- **Dimensions:** Match corresponding image dimensions

---

## 🔍 Important Notes

1. **Data Integrity:** The script performs assertion checks to ensure:
   - Each image has a corresponding mask
   - Image and mask filenames match
   - Spatial dimensions are identical

2. **Edge Handling:** Tiles that don't meet the 512×512 size requirement (typically edge tiles) are **excluded** from the output

3. **File Naming:** The original TIFF files use geographic coordinate naming, while processed tiles append an incremental index (`_0`, `_1`, `_2`, etc.)

4. **Mask Suffix:** All mask tiles end with `_m.png` to differentiate them from image tiles

5. **No Overlap:** The tiling process uses non-overlapping windows, so adjacent tiles don't share pixels

---

## 🔬 Dataset Source

This appears to be derived from the **LandCover.ai** project, which provides high-resolution land cover classification datasets based on aerial imagery from Poland. The dataset is commonly used for semantic segmentation research in remote sensing and computer vision.

---

## 📖 Recommended Reading

- **Semantic Segmentation:** Understanding pixel-wise classification
- **U-Net Architecture:** Popular model for image segmentation
- **Data Augmentation:** Techniques to increase training data diversity
- **Loss Functions for Segmentation:** Cross-entropy, Dice loss, IoU loss

---

## ⚠️ Common Issues & Solutions

### Issue 1: Missing `output/` directory
**Solution:** The script automatically creates it. If it fails, create it manually: `mkdir output`

### Issue 2: Dimension mismatch between image and mask
**Solution:** The script includes assertion checks. If this fails, verify the raw data integrity.

### Issue 3: Out of memory during processing
**Solution:** Process images in batches or use a machine with more RAM.

---

## 📧 Support

If you're using this dataset for research or projects and encounter issues:
1. Check the original LandCover.ai documentation
2. Verify all dependencies are installed correctly
3. Ensure the raw TIFF files are not corrupted

---

## 🤖 Point-Supervised Semantic Segmentation

This repository includes a complete implementation for **point-supervised semantic segmentation** using sparse point labels instead of full pixel-wise annotations.

### File: `point_supervised_segmentation.py`

A single-file implementation containing:

#### Components:

| Component | Description |
|-----------|-------------|
| `LandCoverDataset` | PyTorch Dataset class that loads images and masks |
| `generate_sparse_labels()` | Simulates sparse point annotations from full masks |
| `PartialCELoss` | Cross-entropy loss that ignores unlabeled pixels (-1) |
| `UNet` | Encoder-decoder segmentation model |
| `train_model()` | Training loop with point supervision |
| `evaluate()` | Computes per-class IoU and mean IoU |
| `visualize_predictions()` | Generates visual comparison plots |
| `run_experiments()` | Runs experiments with varying point counts |

#### Mask Class Encoding:

| Pixel Value | Class |
|-------------|-------|
| 0 | Background |
| 1 | Building |
| 2 | Woodland |
| 3 | Water |
| 4 | Road |

#### How Point Supervision Works:

1. **Full masks** are loaded but only used to sample sparse points
2. **N points per class** are randomly selected from the mask
3. All other pixels are set to **-1** (ignored during training)
4. The model learns from only the labeled points
5. At inference, the model predicts dense segmentation

#### Running Experiments:

```bash
# First, generate tiles (if not done already)
python split.py

# Run point-supervised training experiments
python point_supervised_segmentation.py
```

#### Experiment Configuration:

- **Points per class:** 5, 20, 50 (configurable)
- **Image size:** 256×256 pixels
- **Epochs:** 15
- **Optimizer:** Adam (lr=1e-3)
- **Batch size:** 8

#### Expected Output:

1. **Console output:** Training loss and validation mIoU per epoch
2. **predictions_Xpts.png:** Visualization of image, ground truth, sparse labels, and predictions
3. **experiment_results.png:** Comparison plots of mIoU across experiments

#### Sample Visualization:

Each visualization shows 4 columns:
- **Image:** Original RGB aerial image
- **Ground Truth:** Full segmentation mask
- **Sparse Labels:** Only the sampled point labels (rest is black)
- **Prediction:** Model's dense prediction

---

## 📄 License

Please refer to the original LandCover.ai dataset license and terms of use.

---

**Last Updated:** January 28, 2026  
**Dataset Version:** v1  
**Total Tiles:** 10,677  
**Ready for Training:** Yes ✅
