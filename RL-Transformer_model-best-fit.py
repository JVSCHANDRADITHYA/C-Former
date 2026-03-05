"""
batch_rl_inference.py
=====================
Pipeline: DehazeFormer  →  10-step RL post-processor  →  results + CSV

RL policy is trained ONCE on the first test image, then applied to all images.
Denoising uses bilateralFilter (fast). BM3D is commented out below if you prefer it.

Directory assumed:
    DATA_ROOT/
        test/
            hazy/   ← input images
            GT/     ← (optional) ground-truth for PSNR/SSIM

Edit the CONFIG block at the top before running.
"""

# ============================================================
# CONFIG  ← edit these
# ============================================================
DATA_ROOT      = "F:\DehazeFormer\data\SmokeBench"          # root of SmokeBench dataset
PRETRAINED     = "F:\DehazeFormer\save_models\indoor\dehazeformer-t.pth"  # path to your .pth checkpoint
MODEL_NAME     = "dehazeformer-t"        # must match a class in models/
OUT_DIR        = "./results_RL_batch"    # all outputs go here

BRIGHTNESS_OFFSETS = [-0.09, -0.06, -0.03, 0.0, +0.03, +0.06, +0.09]   # added to [0,1] image
CONTRAST_SCALES    = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]         # multiplier around 0.5
DENOISE_OPTIONS    = [False, True]   # bilateral filter on/off

SAVE_ALL_COMBOS    = False   # True → saves all 98 combo images per image (slow disk)
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import os, csv, time, itertools
import torch
import torch.nn as nn
import numpy as np
import cv2
from datetime import datetime
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from pytorch_msssim import ssim as ssim_func

from models import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {DEVICE}")


# ============================================================
# METRICS
# ============================================================
def calc_psnr(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return min(float(20 * np.log10(255.0 / (np.sqrt(mse) + 1e-8))), 60.0)


def calc_ssim(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    def t(x):
        return torch.from_numpy(
            cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        ).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return ssim_func(t(a), t(b), data_range=1.0, size_average=True).item()


# ============================================================
# DEHAZEFORMER
# ============================================================
_tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def load_transformer(name: str, ckpt_path: str) -> nn.Module:
    model = eval(name.replace('-', '_'))()
    ckpt  = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)
    return nn.DataParallel(model).to(DEVICE).eval()


@torch.no_grad()
def run_transformer(model: nn.Module, bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    inp = _tfm(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
    out = torch.clamp(model(inp), -1, 1) * 0.5 + 0.5
    out = (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


# ============================================================
# IMAGE TRANSFORMS
# ============================================================
def apply_combo(img_f32: np.ndarray, brightness: float, contrast: float, denoise: bool) -> np.ndarray:
    """
    img_f32: float32 [0,1] BGR
    Returns: uint8 BGR
    """
    img = img_f32.copy()

    # contrast around midpoint
    if contrast != 1.0:
        img = np.clip(contrast * (img - 0.5) + 0.5, 0, 1)

    # brightness shift
    if brightness != 0.0:
        img = np.clip(img + brightness, 0, 1)

    img_u8 = (img * 255).astype(np.uint8)

    # bilateral denoise (fast, ~5ms)
    if denoise:
        img_u8 = cv2.bilateralFilter(img_u8, d=9, sigmaColor=75, sigmaSpace=75)

    return img_u8


# ============================================================
# PER-IMAGE GRID SEARCH
# ============================================================
def grid_search_image(trans_u8: np.ndarray, gt_u8: np.ndarray, img_out_dir: str):
    """
    Searches over all (brightness, contrast, denoise) combos.
    Returns: (best_img_u8, best_params, all_results_list)
    all_results_list: list of dicts with params + psnr/ssim
    """
    trans_f32 = trans_u8.astype(np.float32) / 255.0

    best_psnr   = -1.0
    best_img    = trans_u8.copy()
    best_params = {"brightness": 0.0, "contrast": 1.0, "denoise": False}
    all_results = []

    combos = list(itertools.product(BRIGHTNESS_OFFSETS, CONTRAST_SCALES, DENOISE_OPTIONS))

    for b, c, d in combos:
        candidate = apply_combo(trans_f32, b, c, d)
        p = calc_psnr(candidate, gt_u8)
        s = calc_ssim(candidate, gt_u8)

        all_results.append({
            "brightness": b, "contrast": c, "denoise": d,
            "psnr": round(p, 4), "ssim": round(s, 4)
        })

        if p > best_psnr:
            best_psnr   = p
            best_img    = candidate.copy()
            best_params = {"brightness": b, "contrast": c, "denoise": d}

        if SAVE_ALL_COMBOS:
            tag = f"b{b:+.2f}_c{c:.2f}_d{int(d)}"
            cv2.imwrite(os.path.join(img_out_dir, f"combo_{tag}.png"), candidate)

    return best_img, best_params, all_results


# ============================================================
# BATCH INFERENCE
# ============================================================
def run_batch(model: nn.Module, data_root: str, out_dir: str):
    hazy_dir = os.path.join(data_root, 'test', 'hazy')
    gt_dir   = os.path.join(data_root, 'test', 'GT')
    has_gt   = os.path.isdir(gt_dir)

    if not has_gt:
        raise RuntimeError(
            "Grid search requires GT images (test/GT/ not found).\n"
            "Without GT you can't score combos — use RL script instead."
        )

    files = sorted(os.listdir(hazy_dir))
    n_combos = len(BRIGHTNESS_OFFSETS) * len(CONTRAST_SCALES) * len(DENOISE_OPTIONS)
    print(f"[GRID] {len(files)} images | {n_combos} combos/image | SAVE_ALL_COMBOS={SAVE_ALL_COMBOS}")

    all_rows = []

    for fname in tqdm(files, desc="Grid search"):
        stem        = os.path.splitext(fname)[0]
        img_out_dir = os.path.join(out_dir, stem)
        os.makedirs(img_out_dir, exist_ok=True)

        t0   = time.time()
        hazy = cv2.imread(os.path.join(hazy_dir, fname))
        gt   = cv2.imread(os.path.join(gt_dir,   fname))

        # --- transformer ---
        trans = run_transformer(model, hazy)
        cv2.imwrite(os.path.join(img_out_dir, "00_hazy.png"),        hazy)
        cv2.imwrite(os.path.join(img_out_dir, "01_transformer.png"), trans)

        trans_psnr = calc_psnr(trans, gt)
        trans_ssim = calc_ssim(trans, gt)

        # --- grid search ---
        best_img, best_params, combo_results = grid_search_image(trans, gt, img_out_dir)
        cv2.imwrite(os.path.join(img_out_dir, "02_best_grid.png"), best_img)

        best_psnr = calc_psnr(best_img, gt)
        best_ssim = calc_ssim(best_img, gt)
        elapsed   = time.time() - t0

        delta = best_psnr - trans_psnr

        # --- also save a side-by-side for quick visual check ---
        h = max(hazy.shape[0], trans.shape[0], best_img.shape[0])
        def pad(img, target_h):
            if img.shape[0] < target_h:
                img = cv2.copyMakeBorder(img, 0, target_h - img.shape[0], 0, 0, cv2.BORDER_CONSTANT)
            return img
        sbs = np.concatenate([pad(hazy, h), pad(trans, h), pad(best_img, h)], axis=1)
        cv2.imwrite(os.path.join(img_out_dir, "03_comparison_hazy_trans_grid.png"), sbs)

        row = {
            "file":             fname,
            "time_s":           round(elapsed, 3),
            "transformer_psnr": round(trans_psnr, 4),
            "transformer_ssim": round(trans_ssim, 4),
            "best_psnr":        round(best_psnr,  4),
            "best_ssim":        round(best_ssim,  4),
            "delta_psnr":       round(delta,       4),
            "best_brightness":  best_params["brightness"],
            "best_contrast":    best_params["contrast"],
            "best_denoise":     best_params["denoise"],
        }
        all_rows.append(row)

        tqdm.write(
            f"  {fname:<25}  {elapsed:.2f}s  | "
            f"Trans={trans_psnr:.2f}  Best={best_psnr:.2f}  ({delta:+.2f} dB)  "
            f"[b={best_params['brightness']:+.2f} c={best_params['contrast']:.2f} d={int(best_params['denoise'])}]"
        )

    # --- CSV ---
    csv_path = os.path.join(out_dir, "metrics.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n[CSV] → {csv_path}")

    return all_rows


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUT_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)

    print("[1/2] Loading DehazeFormer...")
    model = load_transformer(MODEL_NAME, PRETRAINED)

    print("\n[2/2] Running per-image grid search...")
    rows = run_batch(model, DATA_ROOT, out_dir)

    # ---- summary ----
    def avg(key):
        return np.mean([r[key] for r in rows])

    improved  = sum(1 for r in rows if r["delta_psnr"] > 0)
    unchanged = sum(1 for r in rows if r["delta_psnr"] == 0)
    hurt      = sum(1 for r in rows if r["delta_psnr"] < 0)

    # best-param frequency (useful to know what the grid finds most often)
    from collections import Counter
    b_counts = Counter(r["best_brightness"] for r in rows)
    c_counts = Counter(r["best_contrast"]   for r in rows)
    d_counts = Counter(r["best_denoise"]    for r in rows)

    print(f"\n{'='*65}")
    print(f"  Images processed : {len(rows)}")
    print(f"  Improved         : {improved}  |  Unchanged: {unchanged}  |  Hurt: {hurt}")
    print(f"")
    print(f"  Transformer  avg PSNR : {avg('transformer_psnr'):.4f}   SSIM: {avg('transformer_ssim'):.4f}")
    print(f"  Grid-best    avg PSNR : {avg('best_psnr'):.4f}   SSIM: {avg('best_ssim'):.4f}")
    print(f"  Avg Δ PSNR           : {avg('delta_psnr'):+.4f} dB")
    print(f"")
    print(f"  Most-chosen brightness : {b_counts.most_common(3)}")
    print(f"  Most-chosen contrast   : {c_counts.most_common(3)}")
    print(f"  Denoise chosen         : {d_counts}")
    print(f"{'='*65}")
    print(f"  Results → {out_dir}")