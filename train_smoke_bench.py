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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from tqdm import tqdm
from PIL import Image

from pytorch_msssim import ssim
from models import *

# ============================
# ARGUMENTS
# ============================
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='dehazeformer-t', type=str)
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--save_dir', default='./saved_models_Smoked/', type=str)
parser.add_argument('--log_dir', default='./logs/', type=str)
parser.add_argument('--data_root', required=True, type=str)
parser.add_argument('--exp', default='finetune_small', type=str)
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--pretrained', required=True, type=str)
parser.add_argument('--out_dir', default='./results/', type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.backends.cudnn.benchmark = True

# ============================
# LOSS (UNCHANGED)
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
# DATASET
# ============================
class SmokeBenchTrain(Dataset):
    def __init__(self, root, patch_size=256):
        self.hazy_dir = os.path.join(root, 'Train', 'hazy')
        self.gt_dir = os.path.join(root, 'Train', 'GT')
        self.files = sorted(os.listdir(self.hazy_dir))
        self.patch = patch_size

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.files)

    def _random_crop(self, x, y):
        _, h, w = x.shape
        ps = self.patch

        if h < ps or w < ps:
            return x, y  # let it crash loudly if images are too small

        i = torch.randint(0, h - ps + 1, (1,)).item()
        j = torch.randint(0, w - ps + 1, (1,)).item()

        return x[:, i:i+ps, j:j+ps], y[:, i:i+ps, j:j+ps]

    def __getitem__(self, idx):
        name = self.files[idx]

        hazy = Image.open(os.path.join(self.hazy_dir, name)).convert('RGB')
        gt   = Image.open(os.path.join(self.gt_dir, name)).convert('RGB')

        hazy = self.to_tensor(hazy)
        gt   = self.to_tensor(gt)

        hazy, gt = self._random_crop(hazy, gt)
        return hazy, gt


class SmokeBenchTest(Dataset):
    def __init__(self, root):
        self.hazy_dir = os.path.join(root, 'test', 'hazy')
        self.files = sorted(os.listdir(self.hazy_dir))
        self.t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img = Image.open(os.path.join(self.hazy_dir, name)).convert('RGB')
        return self.t(img), name

# ============================
# TRAIN
# ============================
def train_epoch(loader, model, optimizer, scaler, epoch):
    model.train()
    pbar = tqdm(enumerate(loader, 1), total=len(loader), ncols=120, desc=f"Epoch {epoch}")
    run = 0.0

    for i, (src, tgt) in pbar:
        src = src.cuda(non_blocking=True)
        tgt = tgt.cuda(non_blocking=True)

        with autocast('cuda'):
            out = model(src)
            loss = criterion(out, tgt)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        run += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{run/i:.4f}")

# ============================
# INFERENCE
# ============================
@torch.no_grad()
def inference(loader, model, out_dir):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)

    for img, name in tqdm(loader, ncols=120, desc="Final Inference"):
        img = img.cuda()
        out = torch.clamp(model(img), -1, 1)
        out = out * 0.5 + 0.5
        save_image(out, os.path.join(out_dir, name[0]))

# ============================
# MAIN
# ============================
if __name__ == '__main__':

    # ---- Load JSON config (UNCHANGED STYLE)
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
    model = nn.DataParallel(model).cuda()

    # ---- Optimizer & scheduler
    lr = setting['lr'] * 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=lr * 0.1
    )
    scaler = GradScaler('cuda')

    # ---- Data
    train_ds = SmokeBenchTrain(args.data_root, patch_size=setting['patch_size'])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    os.makedirs(args.save_dir, exist_ok=True)

    # ---- Training (NO VALIDATION)
    for epoch in range(1, args.epochs + 1):
        train_epoch(train_loader, model, optimizer, scaler, epoch)
        scheduler.step()

        torch.save(
            {'state_dict': model.state_dict()},
            os.path.join(args.save_dir, f'{args.model}_epoch_{epoch}.pth')
        )

    # ---- FINAL TEST INFERENCE
    test_ds = SmokeBenchTest(args.data_root)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    inference(test_loader, model, args.out_dir)
