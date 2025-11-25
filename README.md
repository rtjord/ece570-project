# Cut Once, Measure Twice: A Reproduction of CutMix’s Reported Gains

This repository contains a single Google Colab notebook, `ECE_570_Project.ipynb`, that reproduces the core classification results of **CutMix** versus **Baseline, Mixup, and Cutout** on CIFAR-100 and Tiny ImageNet using ResNet backbones.

All code lives in one `.ipynb` file; **no local installation is required** beyond using Google Colab.

---

## 1. Code Structure

The notebook is organized into the following sections (see the markdown headings inside the notebook):

1. **Runtime Setup**
   - Creates base directories (`data/`, `checkpoints/`, `results/`, `figures/`).
   - Prints device / package versions.

2. **Helpers**
   - Functions to select CUDA / MPS / CPU and to set global random seeds.
   - Key functions:
     - `set_seed`
     - `get_device`

3. **Load Data**
   - Functionality:
     - Batch-level augmentations (MixUp, CutMix, Cutout) via `torchvision.transforms.v2`:
       - `make_batch_aug`
     - Per-dataset transform builders:
       - `make_cifar_transforms`
       - `make_imagenet_transforms`
     - CIFAR-100 loaders (auto-downloads via `torchvision.datasets.CIFAR100`):
       - `build_cifar100_loaders`
     - Tiny ImageNet download and dataset:
       - `ensure_tinyimagenet`
       - `TinyImageNetVal` dataset class
       - `build_tinyimagenet_loaders`

4. **Build Model**
   - Uses `torchvision.models.resnet18` and `resnet50` with different stems:
     - CIFAR-100 (32×32) “CIFAR-style” stem.
     - Tiny ImageNet (64×64) “ImageNet-style” stem.
   - Key functions:
     - `build_cifar100_model`
     - `build_tiny_imagenet_model`

5. **Train/Evaluate Model**
   - Defines loss, training loop, evaluation loop, and plotting utilities:
     - `SoftCE` (soft cross-entropy)
     - `plot_curves`
     - `train_one_epoch`
     - `cpu_state_dict`
     - `train`
     - `evaluate`
     - `run_experiment`
   - `run_experiment` orchestrates:
     - Data loaders (CIFAR-100 or Tiny ImageNet)
     - Model creation
     - Batch augmentations
     - Optimizer + LR scheduler
     - Mixed-precision training (AMP)
     - Saving checkpoints and result dictionaries

6. **Global Configuration and Experiments**
   - Global hyperparameters:
     - `GLOBAL_CONFIG`: seed, batch size, workers, aug probability, LR, weight decay, AMP.
   - Per-experiment `CONFIG` dictionaries that extend `GLOBAL_CONFIG` with:
     - Dataset (`"cifar100"` or `"tinyimagenet"`)
     - Architecture (`"resnet18"` or `"resnet50"`)
     - Augmentation mode (`"baseline"`, `"mixup"`, `"cutout"`, `"cutmix"`)
     - Number of epochs (e.g., 300 for ResNet-50 CIFAR-100, 200 for ResNet-18)
   - Sections:
     - `# ResNet-50 CIFAR-100`  
       - Baseline, CutMix, Cutout, Mixup
     - `# ResNet-18 CIFAR-100`  
       - Baseline, CutMix, Cutout, MixUp
     - `# ResNet-50 Tiny ImageNet`  
       - Baseline, CutMix
   - Each block defines a `CONFIG = GLOBAL_CONFIG | { ... }` and calls:
     - `run_experiment("<name>", CONFIG)`

7. **Create Figures**
   - Result loading and plotting:
     - `load_runs`
     - `plot_metric`
   - Augmentation visualization helpers:
     - `load_image`, `pil_to_tensor`, `tensor_to_pil`
     - Deterministic augmentation helpers:
       - `apply_mixup`
       - `apply_cutout`
       - `rand_bbox`
       - `apply_cutmix`
       - `make_augmentation_row`
   - These functions generate the 1×4 row of Baseline / Mixup / Cutout / CutMix images shown in the report/poster and the training-history plots.

---

## 2. Dependencies & Environment

The notebook is designed for **Google Colab** with the following libraries:

- **Python**: Colab default (Python 3.x)
- **PyTorch**: `torch`
- **Torchvision**: `torchvision`, including `torchvision.datasets` and `torchvision.transforms.v2`
- **NumPy**: `numpy`
- **Matplotlib**: for plotting curves and images
- **Pillow (PIL)**: image loading for augmentation visualizations
- **Standard library**: `pathlib`, `subprocess`, `typing`, `os`, `random`, `math`, `time`, `json`, etc.
- **google.colab**: for saving files from the Colab runtime (`google.colab.files`)

On a standard Colab GPU runtime, these are already installed. If Colab ever changes its base image and you encounter import errors, you can add a small setup cell such as:

```python
!pip install torch torchvision matplotlib pillow
```

---

## 3. Datasets & Model Downloads

All required datasets are downloaded automatically:

### CIFAR-100

- Loaded via `torchvision.datasets.CIFAR100` with `download=True` inside
  `build_cifar100_loaders`
- Dataset root: `./data/cifar100` (relative to the Colab working directory).
- Train / validation / test splits are created in that function.

### Tiny ImageNet

- Downloaded and extracted by `ensure_tinyimagenet`:
  - Downloads `http://cs231n.stanford.edu/tiny-imagenet-200.zip` (if not already present).
  - Extracts into `./data/tiny-imagenet-200`.
- Validation set is handled by the custom `TinyImageNetVal` class, which:
  - Reads `val_annotations.txt`.
  - Maps images to class IDs using the train directory structure.
- Train / validation / test loaders are constructed by `build_tinyimagenet_loaders`.

### Models

- All models are trained **from scratch** (no pre-trained weights).
- `build_cifar100_model` and `build_tiny_imagenet_model` instantiate ResNet-18 / ResNet-50 from `torchvision.models` with the appropriate stem and classifier head.

---

## 4. How to Run (Colab Instructions)

1. **Open the notebook in Google Colab**
   - Upload `ECE_570_Project.ipynb` to Colab.

2. **Enable a GPU runtime**
   - In Colab: `Runtime → Change runtime type`
   - Set:
     - **Hardware accelerator:** `A100 GPU` (preferred; any GPU is acceptable)
     - **High-RAM:** `On`
   - Click **Save**.

3. **Run the notebook top-to-bottom**
   - Option A: `Runtime → Run all`
   - Option B: Execute each cell manually in order from top to bottom.
   - The notebook will:
     - Create directories
     - Download CIFAR-100 and Tiny ImageNet (if needed)
     - Build models and dataloaders
     - Train all configurations
     - Save checkpoints and result files
     - Generate tables and figures

4. **Do not skip cells**
   - Later experiment cells assume that helper functions, global configs, and paths have already been defined.
   - If Colab disconnects, re-open the notebook and re-run from the top to re-create the state before re-running experiments.

---

## 5. Outputs

After a full run, the notebook produces:

- **Checkpoints** (`checkpoints/`)
  - Model weights for each experiment (e.g., `resnet50_cifar_cutmix.pt`).

- **Results** (`results/`)
  - Serialized dictionaries containing:
    - Training history (loss, Top-1, Top-5 per epoch)
    - Final test metrics for each configuration.

- **Figures & Plots**
  - Training and validation curves, plotted inline.
  - Bar charts comparing augmentations across CIFAR-100 and Tiny ImageNet.
  - Data augmentation example rows (Baseline / Mixup / Cutout / CutMix).
  - Figures are also saved under `figures/` or the notebook root (depending on the cell).

---

## 6. Code Provenance (Original vs Adapted vs External)

### 6.1 Code Written for This Project

All code in `ECE_570_Project.py` was written specifically for this CutMix reproduction. LLMs were used to help write and review this notebok.

We rely heavily on **PyTorch** and **torchvision** APIs, but those are used via their public interfaces (e.g., `torchvision.models.resnet18`, `torchvision.datasets.CIFAR100`, `torchvision.transforms.v2`) rather than copying any of their internal implementation code.

### 6.2 Adapted from Prior Code

At present, there is **no separate starter repository or older codebase** being directly edited inside this notebook. The design, function signatures, and training logic were developed for this reproduction.

You can find the original CutMix code at https://github.com/clovaai/CutMix-PyTorch. We were aware of this repository but did not copy or adapt code from it.


### 6.3 Copied from External Repositories

No functions or classes in this notebook are verbatim copies from external GitHub repositories.

External resources used:

- **Datasets**
  - Tiny ImageNet is downloaded from  
    `http://cs231n.stanford.edu/tiny-imagenet-200.zip`, but the download helper and dataset code (`ensure_tinyimagenet`, `TinyImageNetVal`) are written in this notebook.

---

## 7. Reproducibility Notes

- Random seeds are set for Python, NumPy, and PyTorch in `set_seed`.
- Due to GPU nondeterminism (e.g., cuDNN), exact accuracy numbers may vary slightly between runs/hardware even with fixed seeds.
- You can adjust per-experiment `"epochs"` in each `CONFIG`
  to trade off runtime vs final accuracy.
