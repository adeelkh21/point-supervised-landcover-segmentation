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

## 🔪 Data Split

The dataset is split into three subsets to enable proper training, validation, and testing of machine learning models:

### Training Set (`train.txt`)
- **Purpose:** Used to train the model
- **Total Samples:** 7,471 tiles
- **Percentage:** ~70% of the dataset

### Validation Set (`val.txt`)
- **Purpose:** Used to tune hyperparameters and monitor training progress
- **Total Samples:** 1,603 tiles
- **Percentage:** ~15% of the dataset

### Test Set (`test.txt`)
- **Purpose:** Used for final model evaluation (kept separate during training)
- **Total Samples:** 1,603 tiles
- **Percentage:** ~15% of the dataset

**Total Dataset Size:** 10,677 tiles

---

## 🔧 Data Processing Script (`split.py`)

This Python script is responsible for preprocessing the raw aerial images and masks by tiling them into smaller, manageable pieces.

### What the Script Does:

1. **Reads Raw Images:** Loads all `.tif` files from the `images/` and `masks/` directories
2. **Validates Pairing:** Ensures each image has a corresponding mask with matching dimensions
3. **Tiling Process:** Divides large images into smaller tiles of **512×512 pixels**
4. **Output Generation:** Saves processed tiles to an `output/` directory


### How to Run:

```bash
python split.py
```

**Output:**
- Creates an `output/` directory
- Generates thousands of 512×512 image-mask tile pairs
- Prints progress for each processed image

---

## Getting Started

### Step 1: Prepare the Data

Run the preprocessing script to generate tiles:

```bash
python split.py
```

This will create an `output/` directory with all the tiled images and masks.

---

## 📝 File Format Details

### Split Files (train.txt, val.txt, test.txt)

- **Format:** Plain text, one tile ID per line
- **Content:** Base filenames without extensions

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

## 🔬 Dataset Source

This appears to be derived from the **LandCover.ai** project, which provides high-resolution land cover classification datasets based on aerial imagery from Poland. The dataset is commonly used for semantic segmentation research in remote sensing and computer vision.

---

## Point-Supervised Semantic Segmentation

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

