import argparse
import copy
import csv
import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from Model.GSDTNet import GSDTNet
from utils.config import Config
from utils.edge_dataloader import TrainEdgeDataset
from utils.utils import CosineDecay


EDGE_LOSS_WEIGHT = 0.2
LOSS_POLICY = {
    'segmentation': 'structure_loss on m1 plus weighted side losses on m2/m3/m4',
    'edge_gt': f'BCEWithLogits(edge_pred, edge_gt) * {EDGE_LOSS_WEIGHT}',
}


def parse_args():
    p = argparse.ArgumentParser(description='Train GSDTNet with the final GSDT module.')
    for k in ['epochs', 'batch_size', 'trainsize', 'num_workers']:
        p.add_argument(f'--{k}', type=int, default=None)
    p.add_argument('--dataset_dir', type=str, default=None)
    p.add_argument('--save_dir', type=str, default='')
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--amp', action='store_true', default=False)
    p.add_argument('--no_ema', action='store_true', default=False)
    return p.parse_args()

def make_save_dir(cfg, args):
    if args.save_dir:
        return args.save_dir
    return os.path.join(cfg.result_path, cfg.dir_name)

def structure_loss(logits, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, 31, 1, 15) - mask)
    wbce = F.binary_cross_entropy_with_logits(logits, mask, reduction='none')
    wbce = (weit * wbce).sum((2, 3)) / weit.sum((2, 3))
    pred = torch.sigmoid(logits)
    inter = ((pred * mask) * weit).sum((2, 3))
    union = ((pred + mask) * weit).sum((2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def edge_gt_loss(edge_logits, edge_gt):
    return F.binary_cross_entropy_with_logits(edge_logits, edge_gt)


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(msd[k])

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, sd):
        self.ema.load_state_dict(sd)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger(save_dir)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fh = logging.FileHandler(os.path.join(save_dir, 'train.log'), mode='a', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler())
    return logger


def atomic_save(obj, save_path):
    tmp = save_path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, save_path)


def main():
    args = parse_args()
    cfg = Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.trainsize is not None:
        cfg.trainsize = args.trainsize
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.dataset_dir:
        cfg.dp.dataset_dir = args.dataset_dir
        cfg.dp.refresh()

    save_dir = make_save_dir(cfg, args)
    logger = build_logger(save_dir)
    set_seed(args.seed)
    device = cfg.device

    logger.info('Model: GSDTNet')
    logger.info('Seed: %d', args.seed)
    logger.info('Loss policy: %s', json.dumps(LOSS_POLICY, ensure_ascii=False))

    model = GSDTNet(pretrained=True).to(device)
    bp, op = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (bp if 'encoder' in n else op).append(p)

    optimizer = torch.optim.AdamW(
        [
            {'params': bp, 'lr': cfg.learning_rate * cfg.backbone_lr_mult},
            {'params': op, 'lr': cfg.learning_rate},
        ],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineDecay(optimizer=optimizer, max_lr=cfg.learning_rate, min_lr=cfg.min_lr, max_epoch=cfg.epochs)
    loader = DataLoader(
        TrainEdgeDataset(
            cfg.dp.train_imgs,
            cfg.dp.train_masks,
            cfg.dp.train_depth,
            cfg.dp.train_edges,
            cfg.trainsize,
            True,
            True,
            True,
            True,
        ),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and torch.cuda.is_available())
    ema = None if args.no_ema else ModelEMA(model, cfg.ema_decay)
    start_epoch = 1

    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'], strict=True)
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'ema_model' in ckpt and ema is not None:
            ema.load_state_dict(ckpt['ema_model'])
        if 'scheduler_epoch' in ckpt:
            for _ in range(int(ckpt['scheduler_epoch'])):
                scheduler.step()
        start_epoch = int(ckpt.get('epoch', 0)) + 1

    config_snapshot = vars(args).copy()
    config_snapshot.update({
        'model': 'GSDTNet',
        'dataset_dir': cfg.dp.dataset_dir,
        'epochs': cfg.epochs,
        'batch_size': cfg.batch_size,
        'trainsize': cfg.trainsize,
        'loss_policy': LOSS_POLICY,
        'edge_loss_weight': EDGE_LOSS_WEIGHT,
        'checkpoint_selection': 'minimum_training_loss',
    })
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config_snapshot, f, indent=2, ensure_ascii=False)

    best_loss = float('inf')
    metrics_csv = os.path.join(save_dir, 'train_epoch_metrics.csv')
    if start_epoch == 1 and not os.path.exists(metrics_csv):
        with open(metrics_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'total', 'seg', 'edge_gt', 'lr'])

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        running = {'total': 0.0, 'seg': 0.0, 'egt': 0.0}
        pbar = tqdm(loader, desc=f'Epoch {epoch}/{cfg.epochs}')
        for _, img, gt, dep, edge_gt in pbar:
            img = img.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            dep = dep.to(device, non_blocking=True)
            edge_gt = edge_gt.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=args.amp and torch.cuda.is_available()):
                m1, m2, m3, m4, edge_pred = model(img, dep)
                loss_seg = structure_loss(m1, gt)
                for side_w, logit in zip(cfg.side_loss_weights, (m2, m3, m4)):
                    loss_seg = loss_seg + side_w * structure_loss(logit, gt)
                l_egt = edge_gt_loss(edge_pred, edge_gt)
                loss = loss_seg + EDGE_LOSS_WEIGHT * l_egt

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)

            running['total'] += float(loss.item())
            running['seg'] += float(loss_seg.item())
            running['egt'] += float(l_egt.item())

            denom = max(1, len(loader))
            pbar.set_postfix(
                total=f"{running['total'] / denom:.4f}",
                seg=f"{running['seg'] / denom:.4f}",
                egt=f"{running['egt'] / denom:.4f}",
                lr=f"{scheduler.get_lr():.2e}",
            )

        logger.info(
            'Epoch %d: total=%.4f, seg=%.4f, edge_gt=%.4f, lr=%.3e',
            epoch,
            running['total'] / len(loader),
            running['seg'] / len(loader),
            running['egt'] / len(loader),
            scheduler.get_lr(),
        )

        avg_total_loss = running['total'] / len(loader)
        with open(metrics_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                avg_total_loss,
                running['seg'] / len(loader),
                running['egt'] / len(loader),
                scheduler.get_lr(),
            ])

        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler_epoch': epoch - 1,
            'config': config_snapshot,
        }
        if ema is not None:
            ckpt['ema_model'] = ema.state_dict()
        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            ckpt['best_loss'] = best_loss
            ckpt['best_epoch'] = epoch
            atomic_save(ckpt, os.path.join(save_dir, 'best.pth'))
            logger.info('New best.pth by minimum training loss: epoch=%d, loss=%.6f', epoch, best_loss)
        if epoch % cfg.save_interval == 0 or epoch in (175, 180, 200) or epoch == cfg.epochs:
            atomic_save(ckpt, os.path.join(save_dir, f'epoch_{epoch}.pth'))
        atomic_save(ckpt, os.path.join(save_dir, 'latest.pth'))
        scheduler.step()


if __name__ == '__main__':
    main()
