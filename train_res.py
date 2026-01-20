import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from pytorch_msssim import ssim

from utils import AverageMeter
from datasets.loader import PairLoader
from models import *

# ============================
# ARGUMENTS
# ============================
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='dehazeformer-t', type=str)
parser.add_argument('--num_workers', default=4, type=int)          # Windows-safe
parser.add_argument('--batch_size', default=4, type=int)           # USE YOUR VRAM
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--save_dir', default='./saved_models/', type=str)
parser.add_argument('--data_dir', default='./data/', type=str)
parser.add_argument('--log_dir', default='./logs/', type=str)
parser.add_argument('--dataset', default='RESIDE-IN', type=str)
parser.add_argument('--exp', default='finetune_small', type=str)
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--pretrained', required=True, type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.backends.cudnn.benchmark = True

# ============================
# LOSS
# ============================
def criterion(pred, target):
    l1 = F.l1_loss(pred, target)
    ssim_loss = 1 - ssim(
        pred * 0.5 + 0.5,
        target * 0.5 + 0.5,
        data_range=1.0,
        size_average=True
    )
    return l1 + 0.2 * ssim_loss

# ============================
# TRAIN (ITERATION TQDM)
# ============================
def train_epoch(loader, model, optimizer, scaler, epoch):
    model.train()
    running_loss = 0.0

    pbar = tqdm(
        loader,
        total=len(loader),
        desc=f"Epoch {epoch}",
        ncols=120,
        leave=False
    )

    for i, batch in enumerate(pbar):
        src = batch['source'].cuda(non_blocking=True)
        tgt = batch['target'].cuda(non_blocking=True)

        with autocast('cuda'):
            out = model(src)
            loss = criterion(out, tgt)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        avg_loss = running_loss / (i + 1)

        pbar.set_postfix(
            iter=i,
            loss=f"{loss.item():.4f}",
            avg=f"{avg_loss:.4f}"
        )

    return avg_loss

# ============================
# VALIDATION
# ============================
def validate(loader, model):
    model.eval()
    psnr_meter = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            src = batch['source'].cuda()
            tgt = batch['target'].cuda()

            out = torch.clamp(model(src), -1, 1)

            mse = F.mse_loss(
                out * 0.5 + 0.5,
                tgt * 0.5 + 0.5,
                reduction='none'
            ).mean((1, 2, 3))

            psnr = 10 * torch.log10(1.0 / mse)
            psnr_meter.update(psnr.mean().item(), src.size(0))

    return psnr_meter.avg

# ============================
# MAIN
# ============================
if __name__ == '__main__':

    # ---- Load config
    cfg = os.path.join('configs', args.exp, args.model + '.json')
    if not os.path.exists(cfg):
        cfg = os.path.join('configs', args.exp, 'default.json')

    with open(cfg, 'r') as f:
        setting = json.load(f)

    # ---- Build model
    model = eval(args.model.replace('-', '_'))()

    ckpt = torch.load(args.pretrained, map_location='cpu')
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']

    model.load_state_dict(ckpt, strict=False)
    print("==> Loaded pretrained weights")

    model = nn.DataParallel(model).cuda()

    # ---- Freeze early layers
    for name, p in model.named_parameters():
        if any(x in name for x in ['patch_embed', 'layer1', 'layer2']):
            p.requires_grad = False

    print("==> Frozen early layers")

    # ---- Optimizer
    finetune_lr = setting['lr'] * 0.1
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=finetune_lr
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=finetune_lr * 0.1
    )

    scaler = GradScaler('cuda')

    # ---- Dataset
    root = os.path.join(args.data_dir, args.dataset)

    train_ds = PairLoader(
        root, 'train', 'train',
        setting['patch_size'],
        setting['edge_decay'],
        setting['only_h_flip']
    )

    val_ds = PairLoader(
        root, 'train', setting['valid_mode'],
        setting['patch_size']
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # ---- Logging
    save_dir = os.path.join(args.save_dir, args.exp)
    os.makedirs(save_dir, exist_ok=True)

    writer = SummaryWriter(os.path.join(args.log_dir, args.exp, args.model))
    best_psnr = 0.0

    # ---- Training loop (CLEAN CTRL+C)
    try:
        epoch_pbar = tqdm(range(args.epochs), desc="Training", ncols=120)

        for epoch in epoch_pbar:
            loss = train_epoch(train_loader, model, optimizer, scaler, epoch)
            writer.add_scalar("train/loss", loss, epoch)

            scheduler.step()

            if epoch % setting['eval_freq'] == 0:
                psnr = validate(val_loader, model)
                writer.add_scalar("val/psnr", psnr, epoch)

                if psnr > best_psnr:
                    best_psnr = psnr
                    torch.save(
                        {'state_dict': model.state_dict()},
                        os.path.join(save_dir, args.model + '.pth')
                    )

            epoch_pbar.set_postfix(
                loss=f"{loss:.4f}",
                best_psnr=f"{best_psnr:.2f}"
            )

    except KeyboardInterrupt:
        print("\n⛔ Training interrupted. Saving checkpoint...")
        torch.save(
            {'state_dict': model.state_dict()},
            os.path.join(save_dir, args.model + '_interrupted.pth')
        )

    print(f"==> Training done. Best PSNR: {best_psnr:.2f}")
