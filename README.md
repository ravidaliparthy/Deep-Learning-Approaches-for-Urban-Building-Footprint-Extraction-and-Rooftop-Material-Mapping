# Deep-Learning-Approaches-for-Urban-Building-Footprint-Extraction-and-Rooftop-Material-Mapping
# Nacala Rooftop Segmentation & Material Classification

**Deep Learning Approaches for Urban Building Footprint Extraction and Rooftop Material Mapping**

> **Project:** ICT400 Major Project — SASTRA Deemed to be University (2025–26)  
> **Authors:** Daliparthy Ravi Krishna Charan (126014014) · Bilavampalli Dinesh Reddy (126014011)   
> **Base Paper:** Guthula et al. (2025), *Science of Remote Sensing* — [DOI: 10.1016/j.srs.2025.100306](https://doi.org/10.1016/j.srs.2025.100306)

---

## What This Project Does (Plain English)

Malaria is one of the deadliest diseases in sub-Saharan Africa. Research shows that the **type of roof on a house** is a strong predictor of malaria risk:
- **Thatch roofs** → more mosquito entry points, higher risk
- **Metal sheet roofs** → higher indoor temperatures, lower mosquito survival, lower risk

To map housing quality at scale, this project trains deep learning models to:
1. **Find buildings** in high-resolution drone photos
2. **Separate individual buildings** even when they are touching each other
3. **Classify the roof material** of each building into one of 5 categories

The drone imagery comes from Nacala, Mozambique — a city with one of the highest malaria burdens in the world.

---

## Table of Contents

- [Why This Is Hard](#why-this-is-hard)
- [Dataset](#dataset)
- [Unified Output Format](#unified-output-format)
- [Models — All 17 Variants](#models--all-17-variants)
- [Architecture Details](#architecture-details)
  - [U-Net Family](#u-net-family-models-1-4)
  - [DINOv2 Family](#dinov2-family-models-5-8)
  - [YOLO Family](#yolo-family-models-9-14)
  - [Novel Architectures](#novel-architectures-models-15-17)
  - [Ensemble Models](#ensemble-models-18-19)
- [Deep Ordinal Watershed (DOW)](#deep-ordinal-watershed-dow--the-key-innovation)
- [Loss Functions — Explained](#loss-functions--explained)
- [Class Imbalance — How We Handle It](#class-imbalance--how-we-handle-it)
- [Installation](#installation)
- [Dataset Folder Structure](#dataset-folder-structure)
- [Quick Start](#quick-start)
- [Training — Step by Step](#training--step-by-step)
- [The Disk Cache System](#the-disk-cache-system)
- [Evaluation Pipeline](#evaluation-pipeline)
- [Metrics — What They Mean](#metrics--what-they-mean)
- [Results](#results)
- [Configuration Reference](#configuration-reference)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Citations](#citations)

---

## Why This Is Hard

This problem has three separate challenges that make it harder than standard segmentation:

### Challenge 1: Touching Buildings
In dense informal settlements, buildings are packed so tightly that their roofs literally touch. A standard segmentation model sees them as one big blob, not two separate buildings. You cannot count buildings or classify individual roofs if they are merged.

### Challenge 2: Class Imbalance
The dataset has 9,776 metal sheet buildings but only 174 concrete buildings. A naive model just learns to predict "metal sheet" for everything and still gets decent accuracy. The rare classes (concrete, asbestos) are exactly the ones that are most interesting for public health analysis.

### Challenge 3: Two Conflicting Goals
- **High IoU** (segmentation quality) → you want the boundary of the mask to follow the exact edge of the roof
- **High AP50** (instance detection) → you want to detect each building as a separate object

These goals conflict. If two roofs touch, perfect IoU means predicting them merged, but perfect AP50 means predicting them separated. Our DOW approach addresses this conflict by using two output heads.

---

## Dataset

### Overview

The **Nacala-Roof-Material** dataset was captured by a DJI Phantom 4 Pro drone flying at 120m altitude over three informal settlements in Nacala, Mozambique, between October–December 2021.

- **Resolution:** 4.4 cm/pixel (extremely high detail)
- **Total buildings annotated:** 17,954
- **Annotation method:** Local community mappers traced boundaries on-site, then all annotations were verified twice by the paper authors

### Class Distribution

| Class | Train | Validation | Test | External |
|---|---|---|---|---|
| Metal Sheet | 3,696 | 1,093 | 978 | 661 |
| Thatch | 1,093 | — | — | 792 |
| No Roof | 626 | 149 | 149 | 86 |
| Asbestos | 394 | 77 | 95 | 0 |
| Concrete | 121 | 28 | 23 | 2 |
| **Total** | **6,093** | **1,609** | **1,282** | **792** |

**Key observation:** Concrete has only 121 training buildings — roughly 1.3% of the dataset. This is why class imbalance handling is critical.

### Dataset Splits

The dataset is split in a spatially aware way:
- **Train + Val + Test**: from two settlements, split by a 225m grid (stratified by class)
- **External (ext)**: entire third settlement, never seen during training — used to test generalization

A building belongs to a grid cell based on its centroid. If a building straddles two grid cells from different splits, the "other" side is masked to prevent data leakage.

### What the Files Contain

| Folder | Content | Format |
|---|---|---|
| `images/` | RGB drone orthoimage patches | `.tif` (GeoTIFF, 3-band) |
| `int_mask/` | Binary building mask | `.tif` (0=background, 1=building) |
| `labels/` | YOLO polygon labels with roof class | `.txt` (one line per building) |
| `energy_mask/` | Pre-computed DOW interior maps | `.tif` (optional, skips recompute) |
| `weights/` | Pre-computed pixel weight maps | `.tif` (optional) |
| `p_labels/` | Ground truth building polygons | `.geojson` (for vector-level eval) |

### YOLO Label Format

Each `.txt` file has one line per building:
```
class_id  x1 y1 x2 y2 x3 y3 ...  (normalized polygon coordinates 0–1)
```

YOLO class IDs map to dataset class IDs:

| YOLO ID | Material | Dataset Class ID |
|---|---|---|
| 0 | Metal Sheet | 1 |
| 1 | Thatch | 2 |
| 2 | Asbestos | 3 |
| 3 | Concrete | 4 |
| 4 | No Roof | 5 |

> **Why not the same numbers?** YOLO is 0-indexed; our dataset is 1-indexed (0 = background). The pipeline handles this mapping automatically.

---

## Unified Output Format

Every model in this pipeline outputs a tensor of shape `[B, 8, H, W]`:

```
Dimension  Channel  Content
────────────────────────────────────────────
B          —        Batch size
8          ch 0     Binary building logit  (sigmoid → building probability)
           ch 1     DOW interior logit     (sigmoid → interior probability)
           ch 2     Roof class 0 logit     (Background / no building)
           ch 3     Roof class 1 logit     (Metal Sheet)
           ch 4     Roof class 2 logit     (Thatch)
           ch 5     Roof class 3 logit     (Asbestos)
           ch 6     Roof class 4 logit     (Concrete)
           ch 7     Roof class 5 logit     (No Roof)
H, W       —        Spatial height × width
```

**These are raw logits, not probabilities.** Apply `torch.sigmoid` to channels 0–1 and `torch.argmax` to channels 2–7.

**Exception:** Models without a DOW head (e.g., `unet`, `dinov2`) output `[B, 2, H, W]`. The pipeline handles this automatically.

---

## Models — All 17 Variants

### Selection Guide

| Use case | Recommended model |
|---|---|
| Fastest training / limited GPU | `yolov2_fcn` or `unet` |
| Best binary segmentation only | `ensemble_unet_dino` |
| Best multi-class (all metrics) | `mega_ensemble` |
| Best single model (balanced) | `unet_dow_multi` or `dinov2_dow_multi` |
| Understanding the DOW concept | `unet` → `unet_dow` comparison |
| Research into attention mechanisms | `yolov11_seg` (AIFI) or `yolov12_seg` (Area Attention) |

---

## Architecture Details

### U-Net Family (Models 1–4)

All U-Net models use a **ResNet34 encoder** pretrained on ImageNet. ResNet34 has 4 residual stages producing feature maps at strides 2, 4, 8, 16, and 32. The decoder reconstructs spatial detail by upsampling (nearest-neighbor) and concatenating with skip connections from the encoder.

#### Model 1 — `unet` (Baseline)

```
Input (3×H×W)
    ↓
ResNet34 Encoder  →  features at 5 scales
    ↓
UNet Decoder  (nearest-neighbor upsample + skip concat at each scale)
    ↓  [B, 16, H, W]
    ├── head_binary    → [B, 1, H, W]  binary building logit
    └── head_interior  → [B, 1, H, W]  DOW interior logit
Output: [B, 2, H, W]
```

- One shared decoder, two output heads
- Binary head: 3×3 conv → BN → ReLU → 1×1 conv
- Interior head: larger (64 filters mid-channel) for finer detail
- **No roof classification** — use this for binary segmentation benchmarks only

#### Model 2 — `unet_multi`

Same as `unet` but adds a third head for roof classification:
```
    ├── head_binary    → [B, 1, H, W]
    ├── head_interior  → [B, 1, H, W]
    └── head_roof      → [B, 6, H, W]  6 class logits (0=bg, 1–5=materials)
Output: [B, 8, H, W]
```

- All three tasks share the same decoder features
- Simpler but gradients from all three tasks compete in the shared decoder

#### Model 3 — `unet_dow`

Two completely **separate decoders** from the same encoder:

```
ResNet34 Encoder
    ├── dec_spatial  (for precise boundaries)
    │       ├── head_binary    → [B, 1, H, W]
    │       └── head_interior  → [B, 1, H, W]
    └── dec_semantic (for semantic meaning)
            └── head_roof      → [B, 6, H, W]
Output: [B, 8, H, W]
```

- Spatial decoder focuses on sharp boundaries
- Semantic decoder focuses on class regions
- No gradient interference between segmentation and classification tasks

#### Model 4 — `unet_dow_multi` ⭐ Best U-Net

Two separate **UNet++ decoders** (with nested dense skip connections):

```
UnetPlusPlus (spatial decoder)  →  head_binary, head_interior
UnetPlusPlus (semantic decoder) →  head_roof
```

- UNet++ differs from UNet in that each decoder node receives inputs from all previous nodes at that scale, not just the directly corresponding encoder node
- This creates dense "nested" skip connections that preserve fine-grained boundary detail
- Both decoders share the same ResNet34 encoder weights
- **Falls back to standard UNet** if UnetPlusPlus is not available

---

### DINOv2 Family (Models 5–8)

DINOv2 is a **Vision Transformer (ViT)** trained with self-supervised learning on 142M images. It produces patch embeddings that capture rich semantic meaning without any labels.

**Key difference from U-Net:** DINOv2 processes images as a grid of 14×14 pixel patches, not pixels. The encoder is **completely frozen** — only the decoder is trained.

#### How DINOv2 features are extracted

```python
# Input resized to 448×448 (multple of patch size 14)
x_in = F.interpolate(x, (448, 448))

# DINOv2 produces 1024 patch tokens of dimension 768
# (32×32 spatial grid after excluding the CLS token)
tokens = backbone(x_in).last_hidden_state[:, 1:, :]  # [B, 1024, 768]

# Reshape to spatial feature map
features = tokens.reshape(B, 32, 32, 768).permute(0, 3, 1, 2)  # [B, 768, 32, 32]
```

#### Model 5 — `dinov2`

```
Input → DINOv2 (frozen) → [B, 768, 32, 32]
    ↓
4-stage conv decoder:
  768→256 (Conv+GELU+BN) → Upsample×2
  256→128 (Conv+GELU+BN) → Upsample×2
  128→64  (Conv+GELU+BN) → Upsample×2
   64→32  (Conv+GELU+BN)
    ↓ [B, 32, H, W] (bilinear upsample to input size)
    ├── head_binary    → [B, 1, H, W]
    └── head_interior  → [B, 1, H, W]
Output: [B, 2, H, W]
```

Note: GELU (Gaussian Error Linear Unit) is used instead of ReLU — smoother gradient, better suited for transformer-derived features.

#### Model 6 — `dinov2_multi`

Adds a roof head to model 5. Same decoder, three output heads.

#### Model 7 — `dinov2_dow`

Two separate 4-stage decoders (spatial + semantic) from the same frozen DINOv2 features.

#### Model 8 — `dinov2_dow_multi` ⭐ Best DINOv2

Two separate **5-stage decoders** (768→512→256→128→64→32) — one extra stage compared to models 5–7:

```
DINOv2 features [B, 768, 32, 32]
    ├── dec_spatial (5-stage):  768→512→256→128→64→32  → binary + interior
    └── dec_semantic (5-stage): 768→512→256→128→64→32  → roof classes
```

The extra depth helps recover fine spatial detail from DINOv2's purely semantic patch tokens.

---

### YOLO Family (Models 9–14)

All YOLO models here are **pure PyTorch** — they do not use the Ultralytics library for training. They use a ResNet34 backbone + Feature Pyramid Network (FPN) neck.

#### Model 9 — `yolov2_fcn` (Lightweight)

A simple 5-block FCN encoder (no pretrained weights) + 3-stage decoder. Fastest model, no ImageNet pretraining. Good for quick experiments or very limited hardware.

```
Encoder: 3→32→64→128→256→256  (stride=2 downsampling at blocks 2,3,4)
Decoder: 256→128→64→32  (bilinear upsample ×2 at each stage)
```

#### Models 10/11 — `yolov8_seg` / `yolov8_seg_multi`

**Feature Pyramid Network (FPN):** Multi-scale feature fusion that lets the model detect both large and small buildings simultaneously.

```
ResNet34 features at 4 scales:
  f1: [B, 64,  H/4,  W/4]    ← fine detail (buildings, edges)
  f2: [B, 128, H/8,  W/8]
  f3: [B, 256, H/16, W/16]
  f4: [B, 512, H/32, W/32]   ← coarse semantics

Lateral projections → all 128ch (uniform to allow element-wise addition):
  lat4: 512→128,  lat3: 256→128,  lat2: 128→128,  lat1: 64→128

Top-down FPN (each level adds upsampled features from level above):
  p4 = fpn4(lat4)
  p3 = fpn3(lat3 + upsample(p4))
  p2 = fpn2(lat2 + upsample(p3))
  p1 = fpn1(lat1 + upsample(p2))

Merge: concat all 4 levels at 1/4 resolution → 512ch → 128ch → 64ch
Final upsample to full resolution.
```

#### Models 12/13 — `yolov11_seg` / `yolov11_seg_multi`

Adds **two novel components** on top of the FPN:

**C3k2 Block (Cross-Stage Partial with dual kernels):**
- Splits features into two branches
- Branch 1: identity skip
- Branch 2: two sequential depthwise-separable convolutions
- More efficient than standard convolutions with same accuracy

**AIFI (Area-Invariant Feature Interaction):**
- Lightweight multi-head self-attention applied to the deepest FPN scale
- Downsamples to 16×16 for attention (saves compute)
- Captures global context: helps distinguish a thatch building surrounded by metal buildings
- Upsample back to original scale, added as residual

#### Models 14/15 — `yolov12_seg` / `yolov12_seg_multi`

Two additional novel components:

**Area Attention (A²):**
- Partitions feature map into local windows (8×8 pixels each)
- Computes self-attention within each window independently
- Much more efficient than global attention: O(A × n²) vs O(HW)²
- Applied to deepest two FPN scales (1/32 and 1/16)
- Captures local roof cluster context — perfect for dense settlements

**R-ELAN (Residual Efficient Layer Aggregation Network):**
- Chain of depthwise-separable conv blocks with dense residual connections
- Each branch adds to accumulated output: `out = br_n(br_{n-1}(...br_1(x)...)) + x`
- Provides multi-scale gradient paths for faster convergence

---

### Novel Architectures (Models 15–17)

#### Model 15 — `deeplab_multi`

**DeepLabV3+** with ResNet50 backbone:

```
ResNet50 (dilated: layer3 dilation=2, layer4 dilation=4 → stride-8 output)
    ↓
ASPP (Atrous Spatial Pyramid Pooling):
  5 parallel branches at different scales:
    - 1×1 conv (rate=1)
    - 3×3 dilated conv (rate=6)
    - 3×3 dilated conv (rate=12)
    - 3×3 dilated conv (rate=18)
    - Global average pooling + 1×1 conv
  → Concatenated → 256ch projection

Low-level skip from layer1 (stride-4, 256ch) → projected to 48ch

Decoder: concat(ASPP upsampled to 1/4, low-level 48ch)
         → 2× 3×3 conv → full resolution
```

**Why dilation?** Normal stride-2 convolutions reduce spatial resolution and lose detail. Dilated convolutions maintain resolution while expanding the receptive field — like looking through a sparse sampling grid.

**Known limitation:** Lower AP50 (0.5998) despite good IoU (0.8346). ASPP captures region context well but the DeepLabV3+ design doesn't naturally separate instances — it tends to merge touching buildings.

#### Model 16 — `pspnet_multi`

**Pyramid Scene Parsing Network** with ResNet34:

```
ResNet34 encoder → deepest features [B, 512, H/32, W/32]

PPM (Pyramid Pooling Module):
  4 parallel pooling branches: 1×1, 2×2, 3×3, 6×6 adaptive avg pool
  Each branch: pool → 1×1 conv (512//4 = 128ch) → upsample back
  Concat with original features → 512 + 4×128 = 1024ch
  Bottleneck: 1024 → 256ch

Auxiliary deep supervision head: predicts binary at stride-8 (gradient boost)

Decoder: PPM → upsample ×4 → skip from stride-8 → upsample ×2 → full res
```

**PPM insight:** The 1×1 pool captures global scene context (what settlement type is this?). The 6×6 pool captures local context. Together they help the model know: "this building is in a thatch-heavy area" which can improve classification of ambiguous cases.

#### Model 17 — `segnet_multi`

**SegNet** encoder-decoder with SE (Squeeze-and-Excitation) attention:

```
Encoder (5 stages, VGG-like):
  Each stage: Conv-BN-ReLU × 2 or 3  →  MaxPool2d (return indices!)

Bottleneck:
  Dilated convs (rates 2, 4) for extra context without losing resolution

Decoder (5 stages, mirror of encoder):
  Each stage: MaxUnpool2d (using saved indices) → Conv-BN-ReLU × n
                                                 → SEBlock

SEBlock (Squeeze-and-Excitation):
  Global AvgPool → flatten → FC(c→c/r) → ReLU → FC(c/r→c) → Sigmoid
  Output multiplied channel-wise with input features
  → Recalibrates which feature channels are important
```

**Key idea:** Max-pooling indices remember the *exact pixel locations* of maximum activations. The unpooling step places features back at exactly those locations — preserving spatial precision without learning any upsampling parameters.

**Performance note:** SegNet was the weakest model (IoU 0.6717, AP50 0.4129 at 25 epochs). It needs more epochs to converge — its architecture was designed for lower-resolution inputs and doesn't handle the scale of drone imagery as well.

---

### Ensemble Models (18–19)

#### Model 18 — `ensemble_unet_dino`

Combines `unet_dow_multi` and `dinov2_dow_multi` with **learnable per-channel fusion weights**:

```python
# fusion_logits: shape [2, 8]  (2 branches, 8 output channels)
# Initialized to 0 → softmax gives 0.5 each (equal weight at start)
w = torch.softmax(fusion_logits, dim=0)  # [2, 8]

output = w[0].view(1,8,1,1) * unet_output + w[1].view(1,8,1,1) * dino_output
```

Each output channel can learn a different branch weight. For example:
- Channel 0 (binary) might trust UNet more (precise skip connections)
- Channels 2–7 (roof classes) might trust DINOv2 more (richer semantic features)

#### Model 19 — `mega_ensemble` ⭐ Best Overall

Combines all 8 models:
1. `unet_dow_multi` — precise skip connections
2. `dinov2_dow_multi` — rich semantic patch features
3. `deeplab_multi` — multi-scale dilated context (ASPP)
4. `pspnet_multi` — global scene understanding (PPM)
5. `segnet_multi` — index-based precise unpooling
6. `yolov8_seg` — FPN multi-scale
7. `yolov11_seg` — C3k2 + AIFI global attention
8. `yolov12_seg` — R-ELAN + local window attention

**Adaptive spatial gating** — a lightweight 3-layer CNN reads the input image and predicts per-pixel per-channel weights for each model:

```
gate_net: 3ch → 32ch (stride=4) → 64ch (stride=4) → n_branches×8 ch
→ bilinear upsample to full resolution
→ softmax over branches dimension → spatial weights [B, 8, 8, H, W]

# Also maintains global learnable weights (per-channel, not per-pixel)
# Final blend: 0.3 × global_weights + 0.7 × spatial_weights
```

This means different parts of the image can be handled by different models. Dense urban areas with touching buildings might weight UNet more; sparse areas with rare roof types might weight DINOv2 more.

---

## Deep Ordinal Watershed (DOW) — The Key Innovation

### The Problem

Standard segmentation predicts one mask. When two roofs touch, the model merges them:

```
Ground truth:   [building A] [gap] [building B]
U-Net output:   [     building AB (merged)     ]
```

Result: you count 1 building instead of 2, and if A is metal sheet and B is thatch, you misclassify both.

### The DOW Solution

Predict **two masks** simultaneously:

```
Level 1 (Area):     [  building A  ] [gap] [  building B  ]
Level 2 (Interior): [ interior A ]         [ interior B ]
```

The interior mask is the building eroded by `DOW_NPIX=10` pixels. Even if two buildings touch, their interiors will still be separated (as long as each building is >10px from its neighbor).

**Post-processing (watershed algorithm):**
1. Compute Euclidean distance transform of the area mask
2. Use the interior mask as seeds (markers)
3. Watershed floods the distance map outward from each interior seed
4. Where two seeds' flood regions meet = the boundary between buildings

```python
dist    = ndi.distance_transform_edt(binary_mask)       # height map
markers = ndi.label(interior_mask)[0]                   # one marker per building
inst    = watershed(-dist, markers, mask=binary_mask)   # separated instances
```

### Training Targets

Building the interior target is simple:
```python
dist     = ndi.distance_transform_edt(binary_mask)
interior = (dist > DOW_NPIX).astype(np.uint8)   # pixels > 10px from edge
```

Small buildings (<10px radius) produce empty interior targets — the watershed then falls back to using the binary mask directly.

### DOW Loss Weight

The interior head gets 2× the loss weight because it converges more slowly (it's a harder task):
```python
loss += 2.0 * BCE(interior_pred, interior_target)
```

---

## Loss Functions — Explained

### Binary Segmentation Loss

```
L_binary = BCE + 0.5 × Dice + 0.5 × Tversky
```

**BCE (Binary Cross-Entropy with logits):**
Standard pixel-wise log loss. Includes label smoothing: target 1.0→0.9, target 0.0→0.05. This prevents the model from becoming overconfident on backgrounds.

**Dice Loss:**
```
Dice = 1 - (2 × TP) / (2×TP + FP + FN)
```
Focuses on overlap, not per-pixel accuracy. Helps when buildings are small relative to image size.

**Tversky Loss:**
```
Tversky = 1 - TP / (TP + 0.3×FP + 0.7×FN)
```
Like Dice but penalizes false negatives (missed buildings) more than false positives. Setting β=0.7 (higher penalty for FN) improves recall — important for counting all buildings.

**Pixel weight map:** During training, background pixels between closely spaced buildings are up-weighted:
```
w(x) = w0 × exp( -(d1(x) + d2(x))² / (2 × σ²) )
```
Where d1 and d2 are distances to the nearest and second-nearest building borders. `w0=10`, `σ=5`. This is the formulation from the original U-Net paper (Ronneberger et al. 2015).

### Multi-Class Roof Loss

```
L_multi = 3.0 × FocalCE(roof_logits, roof_target)
```

**Focal Cross-Entropy:**
Standard CE modified to down-weight easy examples:
```
FocalCE = (1 - p_correct)^γ × CE,    γ = 2.0
```
When the model correctly predicts a class with high probability, `(1-p)^γ` ≈ 0 and the loss contribution is small. Hard examples (where the model is wrong or uncertain) contribute much more. This forces the model to focus on rare classes like Concrete.

**MULTI_LOSS_MASK_ONLY=True:** The CE loss is computed only on foreground (building) pixels. Background pixels are excluded because: (a) they are not labeled with a roof class, and (b) they dominate the pixel count and would drown out the signal from actual roof pixels.

**MULTI_LOSS_WEIGHT=3.0:** The multi-class loss is multiplied by 3 to give it enough gradient signal relative to the binary loss. Without this, the model would underfit rare classes.

### DOW Interior Loss

```
L_interior = 2.0 × BCE(interior_pred, interior_target)
```

The 2× weight is because the interior head converges more slowly — without it, the watershed post-processing would receive poor interior maps and fail.

### Total Loss

```
L_total = L_binary + L_interior + L_multi
        = (BCE + 0.5×Dice + 0.5×Tversky) + 2.0×BCE_interior + 3.0×FocalCE_roof
```

---

## Class Imbalance — How We Handle It

### Problem
Concrete has 121 buildings vs Metal Sheet's 3,696. Without intervention, the model ignores Concrete entirely.

### Solution 1: Median-frequency class weights

At the start of training, the full training set is scanned once to compute class pixel frequencies:

```python
freq_c = (count of pixels with class c) / (total pixels)
weights_c = median(freq) / freq_c
```

Result (approximate for this dataset):
```
Class 0 (background):  w = 0.05  (suppressed — handled by binary head)
Class 1 (Metal Sheet): w = 1.0   (baseline)
Class 2 (Thatch):      w = 1.5
Class 3 (Asbestos):    w = 8.0
Class 4 (Concrete):    w = 25.0
Class 5 (No Roof):     w = 10.0
```

Absent classes get weight 0. Weights are capped at 50× minimum to prevent gradient explosion.

### Solution 2: Focal loss (γ=2.0)

Even with class weights, easy examples dominate. Focal loss additionally down-weights easy predictions, forcing the model to focus on Concrete and Asbestos even when Metal Sheet predictions are already very confident.

### Solution 3: Mask-only CE

Computing CE only inside building pixels removes the massive imbalance between building pixels (minority) and background pixels (majority). Without this, the model would overfit to predicting background everywhere.

---

## Installation

### Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.9 | 3.11 |
| PyTorch | 1.13 | 2.1+ (for `torch.compile`) |
| GPU VRAM | 8 GB (batch_size=4) | 16 GB (batch_size=16) |
| Disk space | 5 GB (dataset) + 2 GB (models) | 20 GB |
| RAM | 16 GB | 32 GB |

### Step-by-step installation

```bash
# 1. Create a virtual environment
python -m venv nacala_env
source nacala_env/bin/activate        # Linux/Mac
# nacala_env\Scripts\activate.bat     # Windows

# 2. Install PyTorch (adjust CUDA version for your GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# For CPU only: pip install torch torchvision

# 3. Install segmentation and vision libraries
pip install segmentation-models-pytorch
pip install transformers huggingface-hub   # for DINOv2

# 4. Install geospatial libraries
pip install rasterio geopandas rasterstats shapely

# 5. Install core scientific stack
pip install numpy pandas matplotlib seaborn scikit-image scipy tqdm

# 6. Install CV and metric libraries
pip install opencv-python Pillow torchmetrics

# 7. (Optional) For ultralytics-based inference utilities only
pip install ultralytics
```

### Verify installation

```python
import torch
import rasterio
import segmentation_models_pytorch as smp
from transformers import Dinov2Model

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

---

## Dataset Folder Structure

```
/path/to/dataset/
├── train/
│   ├── images/               ← .tif RGB orthoimage patches
│   │   ├── 00000001.tif
│   │   ├── 00000002.tif
│   │   └── ...
│   ├── int_mask/             ← binary mask (0=bg, 1=building)
│   │   ├── 00000001.tif
│   │   └── ...
│   ├── labels/               ← YOLO polygon labels (multi-class roof)
│   │   ├── 00000001.txt
│   │   └── ...
│   ├── energy_mask/          ← (optional) pre-built DOW interior maps
│   ├── weights/              ← (optional) pre-built pixel weight maps
│   └── p_labels/             ← (optional) polygon GeoJSON for vector eval
│
└── valid/
    ├── images/
    ├── int_mask/
    ├── labels/
    └── p_labels/
```

### The cache directory

The first time you run, the pipeline builds a disk cache at `.nacala_cache/`:
```
.nacala_cache/
├── train_offset0_precomputed.npz   ← binary, interior, weight, multi arrays for train
└── val_offset0_precomputed.npz     ← same for validation
```

These `.npz` files store precomputed numpy arrays for all training samples. After the first run, loading the cache takes under 5 seconds regardless of dataset size.

---

## Quick Start

### 1. Edit paths (required)

Open `nacala_pipeline.py` and set:

```python
TRAIN_ROOT = "/path/to/your/dataset/train"
VALID_ROOT = "/path/to/your/dataset/valid"
```

### 2. Choose a model (required)

```python
MODELS_TO_RUN = ["unet_dow_multi"]
```

### 3. Adjust for your GPU (if needed)

```python
# 8 GB GPU:
PARAMS["batch_size"] = 8
TRAIN_IMG_SIZE = 256

# 16 GB GPU:
PARAMS["batch_size"] = 16
TRAIN_IMG_SIZE = 384   # default

# 40+ GB GPU:
PARAMS["batch_size"] = 32
TRAIN_IMG_SIZE = 512
```

### 4. Run

```bash
python nacala_pipeline.py
```

You'll see:
1. Dataset diagnostic (class values in first 5 masks)
2. Cache build progress (first run only, ~5 min for small datasets)
3. Class weight computation scan
4. Training epochs with loss, IoU, and timing per epoch
5. Final evaluation with full metrics table
6. Saved visualizations in `results/` and `checkpoints/`

---

## Training — Step by Step

### What happens during one epoch

```
for batch in train_loader:
    1. Load image, binary mask, interior mask, weight map, multi-class mask
    2. Apply augmentation (flips, rotations, brightness jitter)
    3. Forward pass → pred [B, 8, H, W]
    4. Compute L_binary (BCE + Dice + Tversky, weighted)
    5. Compute L_interior (BCE × 2.0)  [if DOW model]
    6. Compute L_multi (FocalCE × 3.0, mask-only)  [if *_multi model]
    7. total_loss = L_binary + L_interior + L_multi
    8. If loss is NaN/Inf → skip this batch (NaN guard)
    9. loss.backward() → clip gradients (max_norm=5.0) → optimizer step
```

### Optimizer and LR schedule

**Optimizer:** AdamW with `lr=3e-4`, `weight_decay=1e-4`

**Default schedule:** Cosine annealing with 10-epoch linear warmup
```
Epoch 1–10:  lr = (epoch/10) × base_lr   (linear warmup)
Epoch 11–N:  lr = 0.5 × base_lr × (1 + cos(π × progress))
```
Warmup prevents large early updates from overwhelming rare class gradient signals before the model has seen enough examples.

**Paper-faithful schedule** (`USE_PAPER_STEPLR=True`):
```
StepLR: decay by 0.1 every 50 epochs (best for 300 epochs)
```

### AMP (Automatic Mixed Precision)

Training uses FP16 where safe and FP32 where needed:
```python
with torch.amp.autocast("cuda"):
    pred = model(imgs)
    loss = compute_loss(pred, batch)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

This typically gives 1.5–2× speedup with minimal accuracy impact on Ampere+ GPUs.

### torch.compile

For PyTorch 2.0+, models are compiled for ~20–30% additional speedup:
```python
model = torch.compile(model, mode="default", fullgraph=False, dynamic=False)
```
`dynamic=False` disables symbolic shape tracing, which eliminates spurious warnings.

### Validation frequency

Validation runs every 5 epochs (not every epoch) to save time:
```
run_val = (epoch == 0) or ((epoch+1) % 5 == 0) or ((epoch+1) == total_epochs)
```
Mid-training validation uses no TTA, no morphological post-processing, and no watershed — this is ~3× faster than full evaluation.

### Checkpointing

The best model (by validation IoU) is saved after every validation:
```
checkpoints/{model_name}_best.pt
```

---

## The Disk Cache System

### Why it exists

Building the training targets per-sample is expensive:
- **Border relabeling** (`USE_BORDER_RELABEL=True`): erodes each building instance by 3–4 pixels using `ndi.binary_erosion` → O(N_buildings) per image
- **Weight map** (`USE_WEIGHT_MAP=True`): computes distance to two nearest building borders per pixel → very slow, ~1 second per image

For 6,093 training images, this would be 2+ hours per epoch without caching.

**With caching:** computed once, stored as compressed `.npz`, loaded in microseconds via numpy array indexing.

### Cache invalidation

The cache filename encodes the `MASK_CLASS_OFFSET` setting:
```
train_offset0_precomputed.npz
```

If you change `MASK_CLASS_OFFSET`, the old cache has wrong class IDs. **Always delete the cache after changing this setting:**

```bash
rm -rf .nacala_cache/
```

The pipeline warns you about stale caches with different offsets at startup.

### Cache contents

Each `.npz` stores 4 arrays:
| Array | Shape | dtype | Content |
|---|---|---|---|
| `binary` | [N, 512, 512] | uint8 | Binary building mask (0/1) |
| `interior` | [N, 512, 512] | uint8 | DOW interior mask (0/1) |
| `weight` | [N, 512, 512] | float32 | Pixel weight map (≥1.0) |
| `multi` | [N, 512, 512] | uint8 | Multi-class roof mask (0–5) |

All stored at 512×512. During training, they are resized on-the-fly to `TRAIN_IMG_SIZE` using nearest-neighbor interpolation.

---

## Evaluation Pipeline

After training, the following pipeline runs automatically:

```
1. Restore best checkpoint (by val IoU)

2. Threshold search (single forward pass over validation set):
   - Collect all logits without TTA
   - Sweep thresholds [0.30, 0.35, ..., 0.60]
   - Pick threshold maximizing pixel-level IoU
   → best_thr (used for all final evaluation)

3. Final evaluation (TTA + morph + watershed):
   for each batch:
     a. TTA: run model 3× (original, H-flip, V-flip) → average logits
     b. sigmoid(logits[:,0]) > best_thr → binary mask
     c. Morphological: remove connected components < 16 pixels
     d. For DOW models: watershed instance separation
     e. Compute all metrics

4. Generate outputs:
   - Per-class AP bar chart
   - Normalized confusion matrix
   - Summary metrics heatmap
   - N_VIZ_SAMPLES sample visualizations
```

### Test-Time Augmentation (TTA)

```python
pred_orig = model(images)
pred_hflip = flip(model(flip(images, horizontal)))
pred_vflip = flip(model(flip(images, vertical)))
pred_final = (pred_orig + pred_hflip + pred_vflip) / 3
```

Averaging predictions across augmentations reduces variance and typically improves IoU by 0.5–1%.

---

## Metrics — What They Mean

### Pixel-Level Metrics

**IoU (Intersection over Union)** — "How much do the predicted and true masks overlap?"
```
IoU = |Pred ∩ GT| / |Pred ∪ GT|
```
- Range: 0 (no overlap) to 1 (perfect)
- IoU = 0.85 means 85% of pixels are correct

**mIoU3 / mIoU5** — Mean IoU across roof classes
```
mIoU3 = average(IoU_metal, IoU_thatch, IoU_asbestos)
mIoU5 = average(IoU_metal, IoU_thatch, IoU_asbestos, IoU_concrete, IoU_noroof)
```

**Boundary F1** — Are the building outlines accurate?
```
Recall:    what fraction of true boundary pixels are within 2px of a predicted boundary?
Precision: what fraction of predicted boundary pixels are within 2px of a true boundary?
BF1 = 2 × Precision × Recall / (Precision + Recall)
```

### Object-Level Metrics

**AP50 (Average Precision at IoU=0.5)** — "Do we detect individual buildings correctly?"

A predicted instance is a True Positive if it overlaps with a ground truth instance by ≥50% IoU. For each predicted building, the pipeline:
1. Builds an IoU matrix [Predictions × GT buildings]
2. Greedily matches predictions to GT (best IoU first)
3. Unmatched predictions → False Positives; unmatched GT → False Negatives

```
Precision = TP / (TP + FP)
AP50 = area under Precision-Recall curve
```

**AP50-95** — COCO-style, average AP over IoU thresholds 0.5, 0.55, ..., 0.95
```
AP50-95 = mean(AP@0.50, AP@0.55, ..., AP@0.95)
```
Stricter than AP50 — rewards precise localization.

**TPs (True Positives count)** — How many buildings were correctly detected?
Out of 2,527 reference buildings in the test set, TPs tells you how many were found.

**mAP3@50 / mAP5@50** — Mean AP50 over roof material classes (only correct if roof type is also correct)

---

## Results

### Summary Table (25 training epochs, validation set)

| Model | IoU | AP50 | Notes |
|---|---|---|---|
| **U-Net** | 0.8586 | 0.8393 | Best binary-only IoU |
| unet_dow | 0.8563 | 0.8384 | DOW slightly reduces IoU, improves separation |
| unet_multi | 0.8173 | 0.7461 | Multi-task hurts when sharing decoder |
| unet_dow_multi | 0.8368 | 0.7965 | Separate decoders recover most of the loss |
| dinov2 | 0.8416 | 0.8295 | Strong semantic features |
| dinov2_multi | 0.8279 | 0.8130 | — |
| dinov2_dow | 0.8413 | 0.8277 | — |
| dinov2_dow_multi | 0.8431 | 0.8289 | Best DINOv2 |
| yolov8_seg | 0.8531 | 0.8410 | Fast + accurate |
| yolov8_seg_multi | 0.8316 | 0.7826 | Multi-task cost |
| yolov12_seg | 0.8556 | 0.8316 | Best YOLO binary |
| yolov12_seg_multi | 0.8310 | 0.7713 | — |
| deeplab_multi | 0.8346 | 0.5998 | Good IoU, poor instance separation |
| pspnet_multi | 0.8266 | 0.7582 | Balanced |
| segnet_multi | 0.6717 | 0.4129 | Underperforms at 25 epochs |
| **ensemble_unet_dino** | **0.8671** | **0.8405** | Best overall |
| **mega_ensemble** | **0.8671** | **0.8405** | Best overall |

### Paper baseline (300 epochs, test set — from Guthula et al. 2025)

| Model | IoU | AP50 |
|---|---|---|
| U-Net | 0.895 | 0.910 |
| U-NetDOW | 0.895 | 0.935 |
| DINOv2 | 0.880 | 0.881 |
| DINOv2DOW | 0.881 | 0.931 |
| YOLOv8 | 0.866 | 0.941 |

Note: Paper results use 300 epochs, full pixel weight maps, 5-trial averaging. Our 25-epoch results are solid baselines for the models added beyond the paper's scope.

---

## Configuration Reference

All settings are global variables at the top of `nacala_pipeline.py`. Change them before running.

### Data

```python
TRAIN_ROOT        = "/path/to/train"       # Training dataset root
VALID_ROOT        = "/path/to/valid"       # Validation dataset root
MASK_CLASS_OFFSET = 0                      # 0 = masks already 1-indexed
                                           # If masks are 0-indexed YOLO-style, set to 1
                                           # ALWAYS delete cache when changing this!
```

### Training

```python
PARAMS = {
    "lr":           3e-4,   # Learning rate (AdamW)
    "batch_size":   16,     # Reduce to 8 or 4 if low VRAM
    "epochs":       25,     # 150–300 for paper-faithful results
    "weight_decay": 1e-4,   # L2 regularization
}
TRAIN_IMG_SIZE = 384        # Resize images to this during training
EVAL_IMG_SIZE  = 512        # Final evaluation resolution
```

### Models

```python
MODELS_TO_RUN = ["unet_dow_multi"]   # List of models to train/evaluate sequentially
```

### Loss and class weights

```python
AUTO_COMPUTE_CLASS_WEIGHTS  = True    # Scan train set once to compute weights
USE_CLASS_WEIGHTS_FOR_MULTI = False   # Use manual weights (ignored if AUTO=True)
CLASS_WEIGHTS_MULTI         = None    # Manual weights list [w0, w1, ..., w5]
MULTI_LOSS_WEIGHT           = 3.0     # Multiplier for multi-class CE loss
MULTI_LOSS_MASK_ONLY        = True    # CE only inside building pixels
USE_FOCAL_IN_LOSS           = False   # Add extra focal term to binary loss
FOCAL_GAMMA                 = 2.0
FOCAL_ALPHA                 = 0.75
```

### DOW settings

```python
DOW_NPIX           = 10    # Interior erosion margin (pixels)
USE_BORDER_RELABEL = True  # Erode building borders in training masks
BORDER_GAP_PX      = 7     # Minimum background gap between buildings
USE_WEIGHT_MAP     = False # Pixel weight map (VERY slow, ~2h cache build)
WEIGHT_MAP_W0      = 10.0  # Max weight for between-building pixels
WEIGHT_MAP_SIGMA   = 5.0   # Gaussian falloff
```

### Inference

```python
USE_TTA              = True    # 3-augmentation test-time averaging
USE_THRESHOLD_SEARCH = True    # Search optimal binarization threshold
THRESHOLD_CANDIDATES = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
USE_MORPH_POST       = True    # Remove tiny predicted segments
MORPH_MIN_SIZE       = 16      # Minimum segment size in pixels
MORPH_CLOSE_ITERS    = 0       # Morphological closing iterations
```

### Visualization

```python
SHOW_VIS      = True   # Display plots interactively (set False on HPC/headless)
N_VIZ_SAMPLES = 2      # Number of validation samples to visualize
CKPT_DIR      = "checkpoints"  # Where to save checkpoints and metric plots
```

### LR schedule

```python
USE_PAPER_STEPLR = False  # True = StepLR(step=50, gamma=0.1)
                           # False = CosineAnnealing with 10-epoch warmup
USE_WARMUP       = False   # (reserved, warmup is always on with cosine)
ENCODER_LR_SCALE = 1.0    # <1.0 = lower LR for encoder (differential LR)
```

### Hardware

```python
_MAX_WORKERS = 32   # DataLoader worker cap (too high causes file-descriptor exhaustion)
USE_SWA      = False  # Stochastic Weight Averaging (useful for long runs)
SWA_START_EPOCH = 200
```

---

## Project Structure

```
.
├── nacala_pipeline.py                  ← Everything: models, dataset, training, eval
│
├── checkpoints/                        ← Auto-created on first run
│   ├── {model}_best.pt                 ← Best model weights (by val IoU)
│   ├── {model}_training_curves.png     ← Loss/IoU/F1/LR over epochs
│   ├── {model}_per_class_ap.png        ← Per-class AP@50 vs F1
│   ├── {model}_confusion_matrix.png    ← Normalized confusion matrix
│   ├── {model}_metrics_heatmap.png     ← All 10 metrics in one view
│   └── {model}_final_overlay.png       ← Building overlay visualization
│
├── results/                            ← Per-sample visualizations
│   └── {model}/
│       ├── fig*_rgb_gt_pred.png        ← RGB / GT mask / predicted mask
│       ├── fig*_overlay.png            ← Prediction overlaid on RGB
│       ├── fig*_error.png              ← TP(green)/FP(red)/FN(blue) error map
│       ├── fig*_dow.png                ← DOW area + interior maps (DOW models)
│       ├── fig*_instances.png          ← Instance-colored segmentation
│       ├── fig*_sizes.png              ← Instance size histogram
│       ├── fig*_contours.png           ← GT(green) vs Pred(red) contours
│       ├── fig*_roof.png               ← 6-panel roof material figure
│       └── fig*_roof_comparison.png    ← GT vs pred overlay comparison
│
└── .nacala_cache/                      ← Auto-created on first run
    ├── train_offset0_precomputed.npz
    └── val_offset0_precomputed.npz
```

---

## Troubleshooting

### "All masks appear to be all-zeros"
The `int_mask/` folder path is wrong or the files are misnamed. Check:
```python
import rasterio, numpy as np
with rasterio.open("path/to/int_mask/00000001.tif") as src:
    print(np.unique(src.read(1)))   # should print [0, 1]
```

### "No roof classes found in YOLO labels"
The `labels/` folder is empty or the `.txt` files don't match image filenames. Check:
```bash
ls labels/ | head -5
ls images/ | head -5
# Filenames should match (e.g., 00000001.tif ↔ 00000001.txt)
```

### "Metal Sheet has 0 GT pixels" (in evaluation output)
`MASK_CLASS_OFFSET` is wrong. The multi-class mask has shifted all class IDs. Fix:
1. Check what values your `int_mask/` files actually contain
2. If they're already `[0, 1, 2, 3, 4, 5]`, set `MASK_CLASS_OFFSET = 0`
3. If they're `[0, 1]` (binary only), the multi-class labels come from `labels/` (this is correct for Nacala dataset)
4. Delete cache: `rm -rf .nacala_cache/`

### "CUDA out of memory"
Reduce batch size and/or image size:
```python
PARAMS["batch_size"] = 4   # or 8
TRAIN_IMG_SIZE = 256        # or 320
```

### DINOv2 download fails
First-time use downloads `facebook/dinov2-base` from HuggingFace (~330 MB). Set cache dir if needed:
```bash
export HF_HOME="/path/to/fast/storage"
```
Or pre-download:
```python
from transformers import Dinov2Model
Dinov2Model.from_pretrained("facebook/dinov2-base")
```

### Stale cache warning
If you see: `Found stale caches with different offsets`, delete them:
```bash
rm -rf .nacala_cache/
```

### torch.compile errors (PyTorch < 2.0)
The pipeline silently skips compilation if it fails. No action needed.

### SegNet_multi has very poor results at 25 epochs
This is expected. SegNet's architecture requires more epochs (~100+) to converge on drone imagery. Increase `PARAMS["epochs"]` or use a different model.

---

## Citations

### Base paper (dataset + DOW method)

```bibtex
@article{guthula2025nacala,
  title     = {Drone imagery for roof detection, classification, and segmentation
               to support mosquito-borne disease risk assessment:
               The Nacala-Roof-Material dataset},
  author    = {Guthula, Venkanna Babu and Oehmcke, Stefan and Chilaule, Remigio
               and Zhang, Hui and Lang, Nico and Kariryaa, Ankit
               and Mottelson, Johan and Igel, Christian},
  journal   = {Science of Remote Sensing},
  volume    = {12},
  pages     = {100306},
  year      = {2025},
  publisher = {Elsevier},
  doi       = {10.1016/j.srs.2025.100306}
}
```

### U-Net

```bibtex
@inproceedings{ronneberger2015unet,
  title     = {{U-Net}: Convolutional networks for biomedical image segmentation},
  author    = {Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle = {MICCAI},
  year      = {2015}
}
```

### DINOv2

```bibtex
@article{oquab2024dinov2,
  title   = {{DINOv2}: Learning robust visual features without supervision},
  author  = {Oquab, Mathilde and others},
  journal = {Transactions on Machine Learning Research},
  year    = {2024}
}
```

### This project

Daliparthy Ravi Krishna Charan and Bilavampalli Dinesh Reddy.  
*Deep Learning Approaches for Urban Building Footprint Extraction and Rooftop Material Mapping.*  
ICT400 Major Project, SASTRA Deemed to be University, May 2026.  
Supervisor: Dr. Manjula K R.

---

## Acknowledgements

- Original dataset and DOW method: [Guthula et al. (2025)](https://doi.org/10.1016/j.srs.2025.100306)
- Dataset exploration: [https://mosquito-risk.github.io/Nacala/demo](https://mosquito-risk.github.io/Nacala/demo)
- DOW concept from: Cheng et al. (2024), *Nature Communications* — scattered tree segmentation
- Segmentation models backbone: [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- DINOv2 weights: [facebook/dinov2-base](https://huggingface.co/facebook/dinov2-base) via HuggingFace
- Funding (original paper): Novo Nordisk Foundation (NNF21OC0069116), Pioneer Centre for AI (DNRF P1)
