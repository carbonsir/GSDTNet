# GSDTNet: Geometry-aware Semantic-Detail Transformer for Lightweight Pseudo-RGB-D Camouflaged Object Detection

[![Manuscript](https://img.shields.io/badge/Manuscript-Submitted-blue)](#paper-and-code-relationship)
[![License](https://img.shields.io/badge/License-Academic%20Research-lightgrey.svg)](#license)

## Recommended Citation

If you use this code, trained weights, prediction maps, pseudo-depth maps, or experimental results in your research, please cite the related manuscript.

The manuscript is currently under submission. The official citation information will be updated after the paper is accepted or published.

```bibtex
@misc{gsdtnet2026,
  title        = {GSDTNet: Geometry-aware Semantic-Detail Transformer for Lightweight Pseudo-RGB-D Camouflaged Object Detection},
  author       = {Authors},
  year         = {2026},
  note         = {Manuscript submitted to The 9th Chinese Conference on Pattern Recognition and Computer Vision, PRCV 2026}
}
---

## Abstract

<p align="justify">
Camouflaged object detection (COD) remains challenging because camouflaged targets usually have weak foreground-background contrast, ambiguous boundaries, and complex object structures. This repository provides the official implementation of <b>GSDTNet</b>, a lightweight pseudo-RGB-D COD network based on a <b>Geometry-aware Semantic-Detail Transformer</b> (GSDT). GSDT recalibrates shallow detail features under high-level semantic guidance and uses pseudo-depth-derived geometric cues as lightweight structural guidance. With a single-stream pseudo-RGB-D encoder and an efficient decoder, GSDTNet achieves a favorable accuracy-efficiency trade-off with 4.13M parameters and 1.52G FLOPs, excluding the offline pseudo-depth generation process.
</p>

---

## Paper and Code Relationship

This repository contains the implementation corresponding to the submitted manuscript:

**GSDTNet: Geometry-aware Semantic-Detail Transformer for Lightweight Pseudo-RGB-D Camouflaged Object Detection**

The paper is currently in the manuscript submission stage. If you use this repository, please cite the submitted manuscript. The BibTeX entry will be updated after formal acceptance or publication.

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
| Pre-generated pseudo-depth maps for this repository | To be updated by the authors |

### COD benchmark datasets

Raw COD images are not redistributed in this repository because of dataset license restrictions. Please download the original datasets from the official or commonly used benchmark sources and organize them as described below.

| Dataset Resource | Google Drive / Project Page | Baidu Netdisk / Code |
|---|---|---|
| CAMO dataset | [CAMO project page](https://sites.google.com/view/ltnghia/research/camo) | - |
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

The manuscript experiments were conducted with PyTorch 2.5.1 on a single NVIDIA RTX 3090 GPU.

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

## Contact

For questions, please contact:

```text
carbonsir@126.com
```
