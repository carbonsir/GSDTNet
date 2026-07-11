# GSDTNet: Geometry-Aware Semantic-Detail Transformer for Lightweight Pseudo-RGB-D Camouflaged Object Detection

[![Paper](https://img.shields.io/badge/Paper-PRCV%202026%20Accepted-blue)](#paper-and-code-relationship)
[![Presentation](https://img.shields.io/badge/Presentation-Poster-green)](#paper-and-code-relationship)
[![License](https://img.shields.io/badge/License-Academic%20Research-lightgrey.svg)](#license)

## Citation

If you use this code, pretrained weights, prediction maps, pseudo-depth maps, or experimental results in your research, please cite our paper.

```bibtex
@inproceedings{gsdtnet2026,
  title     = {GSDTNet: Geometry-Aware Semantic-Detail Transformer for Lightweight Pseudo-RGB-D Camouflaged Object Detection},
  author    = {Song, Tan and Li, Jinbao},
  booktitle = {Proceedings of the 9th Chinese Conference on Pattern Recognition and Computer Vision, PRCV 2026},
  year      = {2026},
  note      = {Accepted for poster presentation}
}

## Abstract

<p align="justify">
Camouflaged object detection (COD) remains challenging due to weak foreground-background contrast, ambiguous boundaries, and complex object structures. Existing lightweight COD models often lose fine object details, while dense depth branches may introduce noisy geometric responses and additional computational cost. In this paper, we propose GSDTNet, a lightweight pseudo-RGB-D COD network built upon a Geometry-Aware Semantic-Detail Transformer (GSDT). GSDT recalibrates shallow detail features under high-level semantic guidance and uses pseudo-depth-derived geometric cues as lightweight structural guidance to improve boundary and shape localization. The proposed COD network contains only 4.13M parameters and requires 1.52G FLOPs, where the reported complexity refers to the detection network itself and excludes offline pseudo-depth generation. Experiments on CAMO, COD10K, and NC4K show that GSDTNet achieves superior performance over representative lightweight COD methods, while ablation studies and visualizations verify the effectiveness of GSDT.
</p>

---

## Paper and Code Relationship

This repository contains the official implementation of the paper:

**GSDTNet: Geometry-Aware Semantic-Detail Transformer for Lightweight Pseudo-RGB-D Camouflaged Object Detection**

The paper has been accepted by **The 9th Chinese Conference on Pattern Recognition and Computer Vision, PRCV 2026** for **poster presentation**.

The reported parameters and FLOPs correspond to the proposed COD detection network itself. Pseudo-depth maps are generated offline by Depth Anything V2 and used as fixed auxiliary inputs during training and testing. Therefore, the cost of offline pseudo-depth generation is not included in the reported detection-network complexity.

---

## Download Resources

To improve reproducibility, this repository provides or points to source code, trained model weights, prediction maps, pseudo-depth preparation, and COD benchmark datasets.

### Model weights and prediction maps

Google Drive mirrors:

| Resource | Link |
|---|---|
| GSDTNet pre-trained weights | [Google Drive](https://drive.google.com/file/d/1WZlSN59lzaSjlXx1Pw2p_27j2FRi7pXo/view?usp=drive_link) |
| GSDTNet prediction maps / experimental results | [Google Drive](https://drive.google.com/file/d/1mIBnBQHwlc4nUCdYeS9R6bVkZr55EOMi/view?usp=drive_link) |

### Pseudo-depth maps

GSDTNet uses fixed pseudo-depth maps as auxiliary inputs. The manuscript experiments generate pseudo-depth maps offline using Depth Anything V2, and the generated maps should be placed in the `Depth/` folders shown in the dataset structure below.

| Resource | Link |
|---|---|
| Depth Anything V2 code and model checkpoints | [Official GitHub](https://github.com/DepthAnything/Depth-Anything-V2) |
| Pre-generated pseudo-depth maps for this repository |[Google Drive](https://drive.google.com/file/d/15jOTGaGACdK68Eu91GN94793Z07ZSHBo/view?usp=drive_link) |

### COD benchmark datasets

Raw COD images are not redistributed in this repository because of dataset license restrictions. Please download the original datasets from the official or commonly used benchmark sources and organize them as described below.

| Dataset Resource | Google Drive / Project Page |
|---|---|
| CAMO dataset | [CAMO project page](https://sites.google.com/view/ltnghia/research/camo) |
| COD10K training set | [Google Drive](https://drive.google.com/file/d/1D9bf1KeeCJsxxri6d2qAC7z6O1X_fxpt/view?usp=sharing) |
| COD10K-test + CAMO-test + CHAMELEON test set | [Google Drive](https://drive.google.com/file/d/1QEGnP9O7HbN_2tH999O3HRIsErIVYalx/view?usp=sharing) |
| COD10K full package | [Google Drive](https://drive.google.com/file/d/1vRYAie0JcNStcSwagmCq55eirGyMYGm5/view?usp=sharing) |
| NC4K test set | [Google Drive](https://drive.google.com/file/d/1kzpX_U3gbgO9MuwZIWTuRVpiB7V6yrAQ/view?usp=sharing) |

---

## Overall Architecture

```text
RGB image + pseudo-depth map
    -> PseudoRGBDAdapter
    -> EfficientNet-B0 encoder
    -> Geometry-aware Semantic-Detail Transformer
    -> CGGDecoder
    -> mask predictions + auxiliary edge prediction
```

Main module correspondence:

| Manuscript term | Code implementation |
|---|---|
| GSDTNet | `Model/GSDTNet.py::GSDTNet` |
| Pseudo-RGB-D Adapter | `PseudoRGBDAdapter` |
| EfficientNet-B0 Encoder | `GSDTNet.encoder` |
| Geometry-aware Semantic-Detail Transformer | `GSDT` |
| Depth-guided Geometry Gate | `GSDT.depth_guided_geometry_gate` |
| Semantic Region Residual Compensation | `GSDT.semantic_region_gate` and zero-initialized `region_scale` |
| CGG Decoder | `CGGDecoder` |

---

## Project Structure

```text
GSDTNet/
├── Model/
│   ├── GSDTNet.py
│   ├── EfficientNet.py
│   └── modules.py
├── utils/
│   ├── config.py
│   ├── edge_dataloader.py
│   ├── metrics.py
│   └── utils.py
├── train.py
├── inference.py
├── evaluate.py
└── README.md
```

Only the final GSDTNet model code is included in this clean repository.

---

## Requirements

The experiments were conducted with PyTorch 2.5.1 on a single NVIDIA RTX 3090 GPU.

Recommended environment:

```text
Ubuntu 20.04
Python 3.8+
PyTorch 2.5.1
CUDA 12.1
torchvision
numpy
opencv-python
Pillow
tqdm
scipy
matplotlib
```

Install example:

```bash
conda create -n gsdtnet python=3.8
conda activate gsdtnet

pip install torch torchvision
pip install numpy opencv-python Pillow tqdm scipy matplotlib
```

Please install a PyTorch version compatible with your CUDA environment.

---

## Dataset Preparation

GSDTNet is trained and evaluated on public COD benchmark datasets including CAMO, COD10K, and NC4K.

Default dataset root in `utils/config.py`:

```text
/root/autodl-tmp/data
```

Expected dataset structure:

```text
data/
├── TrainDataset/
│   ├── Imgs/
│   ├── GT/
│   ├── Depth/
│   └── Edge/
└── TestDataset/
    ├── CAMO/
    │   ├── Imgs/
    │   ├── GT/
    │   ├── Depth/
    │   └── Edge/
    ├── COD10K/
    │   ├── Imgs/
    │   ├── GT/
    │   ├── Depth/
    │   └── Edge/
    ├── NC4K/
    │   ├── Imgs/
    │   ├── GT/
    │   ├── Depth/
    │   └── Edge/
    └── CHAMELEON/
        ├── Imgs/
        ├── GT/
        ├── Depth/
        └── Edge/
```

Notes:

- `Depth/` contains pseudo-depth maps generated offline.
- `Edge/` contains boundary maps generated from ground-truth masks for auxiliary edge supervision.
- File names in `Imgs/`, `GT/`, `Depth/`, and `Edge/` should have the same stem.

---

## Training

Example training command:

```bash
python train.py \
  --dataset_dir /path/to/data \
  --epochs 200 \
  --batch_size 32 \
  --trainsize 384 \
  --seed 42 \
  --save_dir ./results/GSDTNet
```

Training outputs:

```text
results/GSDTNet/
├── best.pth
├── latest.pth
├── train.log
├── train_epoch_metrics.csv
└── config.json
```

The training objective uses structure loss for mask prediction and binary cross-entropy loss for the auxiliary edge prediction.

---

## Inference

Run inference with a trained checkpoint:

```bash
python inference.py \
  --ckpt ./results/GSDTNet/best.pth \
  --dataset_dir /path/to/data \
  --datasets CAMO COD10K NC4K \
  --test_size 384 \
  --save_dir ./results/GSDTNet/prediction_maps
```

By default, inference uses EMA weights if they are available in the checkpoint. Add `--use_raw` to evaluate raw model weights.

Expected prediction output:

```text
results/GSDTNet/prediction_maps/
├── CAMO/
├── COD10K/
└── NC4K/
```

---

## Evaluation

Evaluate saved prediction maps:

```bash
python evaluate.py \
  --pred_root ./results/GSDTNet/prediction_maps \
  --dataset_dir /path/to/data \
  --datasets CAMO COD10K NC4K \
  --save_json
```

You can also evaluate directly from a checkpoint. If prediction maps are missing, the script will first run inference:

```bash
python evaluate.py \
  --ckpt ./results/GSDTNet/best.pth \
  --dataset_dir /path/to/data \
  --datasets CAMO COD10K NC4K \
  --save_json
```

Metrics:

- S-measure
- adaptive E-measure
- weighted F-measure
- MAE

---

## Quantitative Results

The following table reports the GSDTNet results from the submitted manuscript.

| Dataset | S-measure ↑ | E-measure ↑ | Weighted F-measure ↑ | MAE ↓ |
|---|---:|---:|---:|---:|
| CAMO | 0.871 | 0.921 | 0.819 | 0.047 |
| COD10K | 0.854 | 0.911 | 0.752 | 0.026 |
| NC4K | 0.875 | 0.924 | 0.816 | 0.036 |

Efficiency:

| Model | Params ↓ | FLOPs ↓ | Input size |
|---|---:|---:|---|
| GSDTNet | 4.13M | 1.52G | 384 × 384 |

Params and FLOPs exclude offline pseudo-depth generation.

---

## Reproducibility Notes

- The repository contains only the final GSDTNet implementation.
- Pseudo-depth maps are fixed auxiliary inputs during training and inference.
- The main prediction used for evaluation is `m1`.
- The `region_scale` parameter in GSDT is zero-initialized so that the semantic region residual branch is learned progressively during training.
- Experimental results may vary slightly because of GPU environment, random seed, dataset preprocessing, and pseudo-depth generation details.

---

## License

This repository is released for academic research purposes only.

Commercial use is not permitted without permission from the authors.

---

## Acknowledgements

We sincerely thank the authors of CAMO, COD10K, NC4K, SINet, Depth Anything V2, and other public COD resources.

---
## Authors

Tan Song and Jinbao Li

Corresponding author: Jinbao Li
