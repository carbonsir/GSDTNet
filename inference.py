import argparse
import csv
import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from Model.GSDTNet import GSDTNet
from utils.config import Config
from utils.edge_dataloader import TestEdgeDataset


DEFAULT_DATASETS = ['CAMO', 'COD10K', 'NC4K']
VALID_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def _find_gt_path(mask_root, pred_name):
    stem = os.path.splitext(pred_name)[0]
    for ext in VALID_EXTS:
        p = os.path.join(mask_root, stem + ext)
        if os.path.exists(p):
            return p
    return None


def _read_original_gt(mask_root, pred_name):
    gt_path = _find_gt_path(mask_root, pred_name)
    if gt_path is None:
        raise FileNotFoundError(f'Cannot find original GT for {pred_name} in {mask_root}')
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(f'Cannot read original GT: {gt_path}')
    return gt


def parse_args():
    p = argparse.ArgumentParser(description='Run inference with GSDTNet.')
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS)
    p.add_argument('--dataset_dir', type=str, default='')
    p.add_argument('--test_size', type=int, default=None)
    p.add_argument('--save_dir', type=str, default='')
    p.add_argument('--use_raw', action='store_true', default=False)
    p.add_argument('--save_aux', action='store_true', default=False,
                   help='Save auxiliary panels: prediction, GT and pseudo-depth cue.')
    return p.parse_args()


def _neighbor_config_path(ckpt_path):
    return os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), 'config.json')


def _safe_load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_checkpoint(ckpt_path):
    return torch.load(ckpt_path, map_location='cpu')


def pick_state_dict(checkpoint, use_raw=False):
    if isinstance(checkpoint, dict):
        if not use_raw and 'ema_model' in checkpoint:
            return checkpoint['ema_model']
        if 'model' in checkpoint:
            return checkpoint['model']
    return checkpoint


def resolve_model_kwargs(ckpt_path, checkpoint):
    # The package contains only GSDTNet. Checkpoint config is read only
    # for metadata compatibility and does not select model variants.
    return {'pretrained': False}


def _remap_checkpoint_keys(state_dict):
    """Map renamed checkpoint keys to the current GSDTNet module names.

    The mapping keeps compatible checkpoints usable after module-name cleanup.
    It does not change tensor values or model logic.
    """
    remapped = {}
    replacements = [
        ('early_rgbd_adapter.', 'pseudo_rgbd_adapter.proj.'),
        ('rgb_backbone.', 'encoder.'),
        ('rgb_projectors.', 'feature_projectors.'),
        ('gsdt.detail_r2.', 'gsdt.detail_f1.'),
        ('gsdt.detail_r3.', 'gsdt.detail_f2.'),
        ('gsdt.semantic_r6.', 'gsdt.semantic_f4.'),
        ('gsdt.semantic_r8.', 'gsdt.semantic_f5.'),
        ('gsdt.geo_gate.', 'gsdt.depth_guided_geometry_gate.'),
        ('gsdt.region_gate.', 'gsdt.semantic_region_gate.'),
    ]
    for key, value in state_dict.items():
        new_key = key
        for old, new in replacements:
            if new_key.startswith(old):
                new_key = new + new_key[len(old):]
                break
        remapped[new_key] = value
    return remapped

def load_model_state(model, state_dict):
    state_dict = _remap_checkpoint_keys(state_dict)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            raise RuntimeError(f'Missing required GSDTNet keys: {result.missing_keys}') from exc
        if result.unexpected_keys:
            print(f'Ignored {len(result.unexpected_keys)} unexpected keys from the checkpoint.')

def _to_uint8_map(x):
    x = x.detach().float().squeeze().cpu().numpy()
    x = np.nan_to_num(x)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn > 1e-8:
        x = (x - mn) / (mx - mn)
    else:
        x = np.zeros_like(x)
    return (x * 255.0).clip(0, 255).astype(np.uint8)


def _tensor_to_uint8_resized(tensor, target_size):
    arr = _to_uint8_map(tensor)
    if arr.shape[:2] != target_size:
        arr = cv2.resize(arr, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    return arr


def _colorize_gray(gray_u8):
    return cv2.applyColorMap(gray_u8, cv2.COLORMAP_JET)


def _panel_cell(title, img_u8, cell_h=220, cell_w=220):
    img = _colorize_gray(img_u8) if img_u8.ndim == 2 else img_u8
    img = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((cell_h + 28, cell_w, 3), 255, dtype=np.uint8)
    canvas[28:, :, :] = img
    cv2.putText(canvas, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def _save_aux_panel(root, dataset, name, pred_u8, gt_u8, depth_t, target_size):
    out_dir = os.path.join(root, dataset + '_aux', 'panel')
    os.makedirs(out_dir, exist_ok=True)
    cells = [
        _panel_cell('Pred', pred_u8),
        _panel_cell('GT', gt_u8),
        _panel_cell('DepthCue', _tensor_to_uint8_resized(depth_t, target_size)),
    ]
    while len(cells) < 4:
        cells.append(np.full_like(cells[0], 255))
    panel = np.concatenate(cells, axis=1)
    out_name = os.path.splitext(name)[0] + '_aux_panel.jpg'
    cv2.imwrite(os.path.join(out_dir, out_name), panel, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def _safe_mean(arr, mask):
    return float(arr[mask].mean()) if mask.any() else 0.0


def _tensor_to_resized_np(tensor, target_hw):
    arr = tensor.detach().float().squeeze().cpu().numpy()
    if arr.shape[:2] != target_hw:
        arr = cv2.resize(arr, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)
    return arr


def _aux_stats(name, pred_u8, gt_u8, dep_t, edge_pred_u8):
    gt_mask = gt_u8 > 127
    bg_mask = ~gt_mask
    target_hw = gt_u8.shape[:2]
    pred_f = pred_u8.astype(np.float32) / 255.0
    gt_f = gt_mask.astype(np.float32)
    dep = _tensor_to_resized_np(dep_t, target_hw)
    dep_fg = _safe_mean(dep, gt_mask)
    dep_bg = _safe_mean(dep, bg_mask)
    return {
        'image': name,
        'pred_mae': float(np.mean(np.abs(pred_f - gt_f))),
        'depth_fg_mean': dep_fg,
        'depth_bg_mean': dep_bg,
        'depth_fg_bg_contrast': abs(dep_fg - dep_bg),
        'edge_pred_mean': float(edge_pred_u8.mean() / 255.0),
    }


@torch.no_grad()
def inference_one(model, cfg, dataset, test_size, pred_root, save_aux=False):
    save_path = os.path.join(pred_root, dataset)
    edge_path = os.path.join(pred_root, dataset + '_edge')
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(edge_path, exist_ok=True)
    aux_rows = []

    ds = TestEdgeDataset(
        getattr(cfg.dp, f'test_{dataset}_imgs'),
        getattr(cfg.dp, f'test_{dataset}_masks'),
        getattr(cfg.dp, f'test_{dataset}_depth'),
        getattr(cfg.dp, f'test_{dataset}_edges'),
        test_size,
    )
    model.eval()
    mask_root = getattr(cfg.dp, f'test_{dataset}_masks')

    for _, img, _, dep, _, _, name in tqdm(ds, desc=f'Infer {dataset}'):
        img = img.unsqueeze(0).to(cfg.device)
        dep = dep.unsqueeze(0).to(cfg.device)
        pred, _, _, _, edge_pred = model(img, dep)

        gt_u8_original = _read_original_gt(mask_root, name)
        target_size = gt_u8_original.shape[:2]
        pred = F.interpolate(pred, size=target_size, mode='bilinear', align_corners=False)
        edge_pred = F.interpolate(edge_pred, size=target_size, mode='bilinear', align_corners=False)

        pred_u8 = (torch.sigmoid(pred).squeeze().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        edge_u8 = (torch.sigmoid(edge_pred).squeeze().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        cv2.imwrite(os.path.join(save_path, name), pred_u8)
        cv2.imwrite(os.path.join(edge_path, name), edge_u8)

        if save_aux:
            _save_aux_panel(pred_root, dataset, name, pred_u8, gt_u8_original, dep, target_size)
            aux_rows.append(_aux_stats(name, pred_u8, gt_u8_original, dep, edge_u8))

    if save_aux and aux_rows:
        csv_path = os.path.join(pred_root, dataset + '_aux', 'aux_metrics.csv')
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(aux_rows[0].keys()))
            writer.writeheader()
            writer.writerows(aux_rows)


def main():
    args = parse_args()
    cfg = Config()
    if args.dataset_dir:
        cfg.dp.dataset_dir = args.dataset_dir
        cfg.dp.refresh()

    checkpoint = load_checkpoint(args.ckpt)
    state_dict = pick_state_dict(checkpoint, use_raw=args.use_raw)
    model_kwargs = resolve_model_kwargs(args.ckpt, checkpoint)
    model_info = {
        'model': 'GSDTNet',
        'state_dict_source': 'model' if args.use_raw else ('ema_model' if isinstance(checkpoint, dict) and 'ema_model' in checkpoint else 'model'),
    }
    print('==== Inference Model Config ====')
    print(json.dumps(model_info, indent=2, ensure_ascii=False))

    model = GSDTNet(**model_kwargs).to(cfg.device)
    load_model_state(model, state_dict)

    pred_root = args.save_dir or os.path.join(os.path.dirname(args.ckpt), 'prediction_maps')
    test_size = args.test_size or cfg.trainsize
    for dataset in args.datasets:
        inference_one(model, cfg, dataset, test_size, pred_root, save_aux=args.save_aux)

    print(f'Prediction maps saved to: {pred_root}')


if __name__ == '__main__':
    main()
