# GSDTNet

Official PyTorch implementation of **GSDTNet: Geometry-aware Semantic-Detail Transformer for Lightweight Pseudo-RGB-D Camouflaged Object Detection**.

GSDTNet is a lightweight pseudo-RGB-D camouflaged object detection network. It uses an RGB image and an offline pseudo-depth map as input, projects them into a compact pseudo-RGB-D representation, extracts multi-level features with EfficientNet-B0, recalibrates shallow detail features with the Geometry-aware Semantic-Detail Transformer (GSDT), and predicts camouflaged object masks with a lightweight CGG decoder.

## Highlights

- **Pseudo-RGB-D single-stream design**: RGB and pseudo-depth are fused by a lightweight 1×1 adapter before feature extraction.
- **GSDT module**: high-level semantics guide shallow detail recalibration, while pseudo-depth-derived geometry improves structure localization.
- **Semantic region residual compensation**: a zero-initialized region branch stabilizes training and gradually provides region-level compensation.
- **Lightweight decoder**: a CGG-based top-down decoder produces multi-scale mask predictions and an auxiliary edge prediction.
- **Efficient COD model**: designed for a favorable accuracy-efficiency trade-off on CAMO, COD10K, and NC4K.

## Network Overview

```text
RGB image + pseudo-depth map
        |
PseudoRGBDAdapter
        |
EfficientNet-B0 encoder
        |
F1, F2, F3, F4, F5
        |
GSDT semantic-detail-geometric recalibration
        |
Enhanced F2 + F3/F4/F5
        |
CGGDecoder
        |
Mask predictions + auxiliary edge prediction
```

## Repository Structure

```text
GSDTNet/
├── Model/
│   ├── EfficientNet.py
│   ├── GSDTNet.py
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

## Environment

The code is implemented with PyTorch. A typical environment is:

```bash
conda create -n gsdt python=3.10 -y
conda activate gsdt

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python pillow numpy tqdm scipy
```

Install the PyTorch build that matches your CUDA version.

## Dataset Preparation

Prepare the dataset directory as follows:

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
    └── NC4K/
        ├── Imgs/
        ├── GT/
        ├── Depth/
        └── Edge/
```

Notes:

- `Imgs/` contains RGB images.
- `GT/` contains binary ground-truth masks.
- `Depth/` contains offline pseudo-depth maps.
- `Edge/` contains edge supervision maps generated from the object masks.

Pseudo-depth maps can be generated offline with Depth Anything V2 or another monocular depth estimator. The COD network itself only loads the generated depth maps during training and inference.

## Training

Run training with:

```bash
python train.py \
  --dataset_dir /path/to/data \
  --save_dir ./results/GSDTNet \
  --epochs 180 \
  --batch_size 16 \
  --trainsize 384 \
  --amp
```

Useful options:

- `--dataset_dir`: root directory of the prepared datasets.
- `--save_dir`: directory for checkpoints and logs.
- `--resume`: resume training from a checkpoint.
- `--amp`: enable automatic mixed precision.
- `--no_ema`: disable EMA checkpoint tracking.

The script saves:

```text
results/GSDTNet/
├── best.pth
├── latest.pth
├── epoch_*.pth
├── config.json
├── train.log
└── train_epoch_metrics.csv
```

## Inference

Generate prediction maps from a checkpoint:

```bash
python inference.py \
  --ckpt ./results/GSDTNet/best.pth \
  --dataset_dir /path/to/data \
  --datasets CAMO COD10K NC4K \
  --save_dir ./results/GSDTNet/prediction_maps
```

Prediction maps are saved under:

```text
results/GSDTNet/prediction_maps/
├── CAMO/
├── COD10K/
└── NC4K/
```

To save auxiliary visualization panels, add:

```bash
--save_aux
```

## Evaluation

Evaluate prediction maps:

```bash
python evaluate.py \
  --pred_root ./results/GSDTNet/prediction_maps \
  --dataset_dir /path/to/data \
  --datasets CAMO COD10K NC4K \
  --save_json
```

You can also run inference and evaluation together by passing a checkpoint:

```bash
python evaluate.py \
  --ckpt ./results/GSDTNet/best.pth \
  --dataset_dir /path/to/data \
  --datasets CAMO COD10K NC4K \
  --save_json
```

The evaluation script reports:

- `sm`: Structure-measure
- `emAdp`: adaptive E-measure
- `wfm`: weighted F-measure
- `mae`: Mean Absolute Error

## Model Components

The main model is implemented in `Model/GSDTNet.py`.

- `PseudoRGBDAdapter`: projects RGB and pseudo-depth into a 3-channel pseudo-RGB-D input.
- `GSDT`: performs semantic-detail cross attention, depth-guided geometry gating, and semantic region residual compensation.
- `CGGDecoder`: fuses top-down encoder features and produces mask and edge predictions.
- `GSDTNet`: integrates the adapter, EfficientNet-B0 encoder, GSDT module, and decoder.

## Citation

If this repository is useful for your research, please cite the paper:

```bibtex
@inproceedings{gsdtnet,
  title     = {GSDTNet: Geometry-aware Semantic-Detail Transformer for Lightweight Pseudo-RGB-D Camouflaged Object Detection},
  author    = {First Author and Second Author},
  booktitle = {Proceedings},
  year      = {2026}
}
```

## Acknowledgement

This implementation uses EfficientNet-B0 as the lightweight backbone and follows common COD training and evaluation protocols. We thank the authors of the public COD benchmarks and related open-source projects.
