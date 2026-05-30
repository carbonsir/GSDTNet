# GSDTNet

GSDTNet 是基于最终 `v4b_res_region` 分支整理出的干净版本。该版本删除了消融实验开关和辅助验证脚本，只保留论文中的完整模型路径：

```text
RGB + pseudo-depth
    -> PseudoRGBDAdapter
    -> EfficientNet-B0 encoder
    -> GSDT (Geometry-aware Semantic-Detail Transformer)
    -> CGG-style top-down decoder
    -> mask predictions + auxiliary edge prediction
```

## 保留的最终模型逻辑

本代码保留原 `v4b_res_region` 分支中的有效路径，不再提供 `clean_early`、`v2_stride_bridge`、`naive_detail`、`no_geo`、`full` 等消融选项。

对应关系如下：

| 论文名称 | 代码位置 |
| --- | --- |
| GSDTNet | `Model/GSDTNet.py::GSDTNet` |
| Pseudo-RGB-D Adapter / `A([I,D])` | `PseudoRGBDAdapter` |
| EfficientNet-B0 Encoder | `GSDTNet.encoder` |
| Geometry-aware Semantic-Detail Transformer | `GSDT` |
| Depth-guided Geometry Gate (DGG) | `GSDT.depth_guided_geometry_gate` |
| Semantic Region Residual Compensation (RRC) | `GSDT.semantic_region_gate` + zero-init `region_scale` |
| CGG-style Decoder | `CGGDecoder` |

## 目录结构

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

## 数据目录

默认数据根目录在 `utils/config.py` 中设置：

```text
/root/autodl-tmp/data
```

期望的数据组织形式为：

```text
TrainDataset/
  Imgs/
  GT/
  Depth/
  Edge/
TestDataset/
  CAMO/{Imgs,GT,Depth,Edge}
  COD10K/{Imgs,GT,Depth,Edge}
  NC4K/{Imgs,GT,Depth,Edge}
  CHAMELEON/{Imgs,GT,Depth,Edge}
```

其中 `Depth` 为离线生成的 pseudo-depth map，`Edge` 为由 mask 边界生成的辅助边缘监督。

## 训练

```bash
python train.py \
  --dataset_dir /path/to/data \
  --epochs 200 \
  --batch_size 32 \
  --trainsize 384 \
  --seed 42 \
  --save_dir ./results/GSDTNet
```

训练输出包含：

- `best.pth`：按最低训练 loss 保存的最佳模型
- `latest.pth`：最新模型
- `train.log`
- `train_epoch_metrics.csv`
- `config.json`

## 推理

```bash
python inference.py \
  --ckpt ./results/GSDTNet/best.pth \
  --dataset_dir /path/to/data \
  --datasets CAMO COD10K NC4K \
  --test_size 384 \
  --save_dir ./results/GSDTNet/prediction_maps
```

默认使用 checkpoint 中的 EMA 权重；如需使用原始模型权重，可添加 `--use_raw`。

## 评估

```bash
python evaluate.py \
  --pred_root ./results/GSDTNet/prediction_maps \
  --dataset_dir /path/to/data \
  --datasets CAMO COD10K NC4K \
  --save_json
```

也可以直接指定 checkpoint，若预测图不存在会先自动执行推理：

```bash
python evaluate.py \
  --ckpt ./results/GSDTNet/best.pth \
  --dataset_dir /path/to/data \
  --datasets CAMO COD10K NC4K \
  --save_json
```

## 说明

- 本版本只保留最终 GSDTNet 代码，不包含消融实验代码和历史 `.md` 文档。
- `GSDT.region_scale` 仍保持零初始化，这是最终 `v4b_res_region` 分支的关键设计，用于保证训练初期不破坏原几何门控路径。
- `inference.py` 中提供旧 `v4b_res_region` checkpoint 的键名映射，便于在模型重命名后加载旧最终分支权重。
