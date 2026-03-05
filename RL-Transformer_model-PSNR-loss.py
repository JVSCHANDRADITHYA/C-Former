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

RL_TRAIN_STEPS   = 1500   # PPO timesteps total across training images
RL_TRAIN_N_IMGS  = 5      # how many images to train on (first N from test set)
MAX_RL_STEPS     = 10     # fixed rollout length at inference
OBS_SIZE         = 64
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import os, csv, time
import torch
import torch.nn as nn
import numpy as np
import cv2
from datetime import datetime
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from pytorch_msssim import ssim as ssim_func

from models import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {DEVICE}")


# ============================================================
# METRICS
# ============================================================
def calc_psnr(img_a: np.ndarray, img_b: np.ndarray) -> float:
    if img_a.shape != img_b.shape:
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))
    mse = np.mean((img_a.astype(np.float64) - img_b.astype(np.float64)) ** 2)
    return min(float(20 * np.log10(255.0 / (np.sqrt(mse) + 1e-8))), 60.0)


def calc_ssim(img_a: np.ndarray, img_b: np.ndarray) -> float:
    if img_a.shape != img_b.shape:
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))
    def to_t(x):
        return torch.from_numpy(
            cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        ).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return ssim_func(to_t(img_a), to_t(img_b), data_range=1.0, size_average=True).item()


# ============================================================
# NO-REF SCORE  (fallback only when GT unavailable)
# ============================================================
def nr_score(img_u8: np.ndarray) -> float:
    gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)
    lap  = cv2.Laplacian(gray, cv2.CV_64F)
    bp   = abs(np.mean(gray) - 128) / 128
    return float(0.5 * lap.var() - 1.0 * np.std(lap) - bp)


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
# RL ENVIRONMENT
# ============================================================
class EnhancementEnv(gym.Env):
    """
    Actions:
        0  brightness +0.03
        1  brightness -0.03
        2  contrast   x1.05  (+5%)
        3  contrast   x0.95  (-5%)
        4  bilateral denoise
        5  STOP (no-op)

    Reward: ΔPSNR vs GT (normalised), or ΔNR-score if no GT.
    """

    def __init__(self, transformer_out_bgr: np.ndarray, gt_bgr=None):
        super().__init__()
        self.orig    = transformer_out_bgr.astype(np.float32) / 255.0
        self.gt_u8   = gt_bgr if gt_bgr is not None else None
        self.current = self.orig.copy()

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(OBS_SIZE * OBS_SIZE * 3,),
            dtype=np.float32
        )
        self.steps = 0

    def _u8(self, arr=None):
        return ((self.current if arr is None else arr) * 255).clip(0, 255).astype(np.uint8)

    def _apply(self, action: int):
        img, stop = self.current.copy(), False
        if action == 0:
            img = np.clip(img + 0.03, 0, 1)
        elif action == 1:
            img = np.clip(img - 0.03, 0, 1)
        elif action == 2:
            img = np.clip(1.05 * (img - 0.5) + 0.5, 0, 1)
        elif action == 3:
            img = np.clip(0.95 * (img - 0.5) + 0.5, 0, 1)
        elif action == 4:
            u8  = (img * 255).astype(np.uint8)
            img = cv2.bilateralFilter(u8, d=9, sigmaColor=75, sigmaSpace=75).astype(np.float32) / 255.0
        elif action == 5:
            stop = True
        return img, stop

    def _reward(self, prev: np.ndarray, new: np.ndarray) -> float:
        if self.gt_u8 is not None:
            r_new  = calc_psnr(self._u8(new),  self.gt_u8)
            r_prev = calc_psnr(self._u8(prev), self.gt_u8)
            return (r_new - r_prev) / 10.0   # 1 dB gain → +0.1
        else:
            return nr_score(self._u8(new)) - nr_score(self._u8(prev))

    def _obs(self) -> np.ndarray:
        small = cv2.resize(self._u8(), (OBS_SIZE, OBS_SIZE))
        return small.astype(np.float32).reshape(-1) / 255.0

    def reset(self, seed=None, options=None):
        self.steps   = 0
        self.current = self.orig.copy()
        return self._obs(), {}

    def step(self, action: int):
        prev = self.current.copy()
        self.current, stop = self._apply(action)
        reward     = self._reward(prev, self.current)
        self.steps += 1
        terminated = stop or (self.steps >= MAX_RL_STEPS)
        return self._obs(), reward, terminated, False, {}


# ============================================================
# TRAIN RL
# ============================================================
def train_rl(model: nn.Module, hazy_dir: str, gt_dir: str) -> PPO:
    files   = sorted(os.listdir(hazy_dir))
    n_train = min(RL_TRAIN_N_IMGS, len(files))
    has_gt  = os.path.isdir(gt_dir)

    print(f"[RL] Training on {n_train} images | GT reward: {has_gt}")

    def make_env(fname):
        hazy  = cv2.imread(os.path.join(hazy_dir, fname))
        trans = run_transformer(model, hazy)
        gt    = cv2.imread(os.path.join(gt_dir, fname)) if has_gt else None
        return lambda: EnhancementEnv(trans, gt)

    envs = DummyVecEnv([make_env(f) for f in files[:n_train]])

    t0    = time.time()
    agent = PPO(
        "MlpPolicy", envs,
        verbose=0,
        device=DEVICE,
        policy_kwargs=dict(net_arch=[128, 128]),
        n_steps=256,
        batch_size=64,
        learning_rate=3e-4,
    )
    agent.learn(total_timesteps=RL_TRAIN_STEPS)
    print(f"[RL] Training done in {time.time()-t0:.1f}s")
    return agent


# ============================================================
# BATCH INFERENCE
# ============================================================
def run_batch(model: nn.Module, agent: PPO, data_root: str, out_dir: str):
    hazy_dir = os.path.join(data_root, 'test', 'hazy')
    gt_dir   = os.path.join(data_root, 'test', 'GT')
    has_gt   = os.path.isdir(gt_dir)

    files = sorted(os.listdir(hazy_dir))
    print(f"[BATCH] {len(files)} images | GT: {has_gt}")

    all_rows = []

    for fname in tqdm(files, desc="Inference"):
        stem        = os.path.splitext(fname)[0]
        img_out_dir = os.path.join(out_dir, stem)
        os.makedirs(img_out_dir, exist_ok=True)

        t0   = time.time()
        hazy = cv2.imread(os.path.join(hazy_dir, fname))
        gt   = cv2.imread(os.path.join(gt_dir, fname)) if has_gt else None

        # transformer
        trans = run_transformer(model, hazy)
        cv2.imwrite(os.path.join(img_out_dir, "00_hazy.png"),        hazy)
        cv2.imwrite(os.path.join(img_out_dir, "01_transformer.png"), trans)

        # RL rollout — always exactly MAX_RL_STEPS saves
        env      = EnhancementEnv(trans, gt)
        obs, _   = env.reset()
        done     = False
        step_imgs = []

        for step_i in range(MAX_RL_STEPS):
            if not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(int(action))
                done = terminated or truncated
            step_img = (env.current * 255).clip(0, 255).astype(np.uint8)
            step_imgs.append(step_img.copy())
            cv2.imwrite(os.path.join(img_out_dir, f"rl_step_{step_i+1:02d}.png"), step_img)

        final_img = step_imgs[-1]
        cv2.imwrite(os.path.join(img_out_dir, "rl_final.png"), final_img)
        elapsed = time.time() - t0

        # metrics
        def m(img):
            if gt is None:
                return None, None
            return round(calc_psnr(img, gt), 4), round(calc_ssim(img, gt), 4)

        row = {"file": fname, "time_s": round(elapsed, 2), "has_gt": has_gt}
        tp, ts = m(trans)
        row["transformer_psnr"] = tp
        row["transformer_ssim"] = ts

        for i, si in enumerate(step_imgs, 1):
            p, s = m(si)
            row[f"rl_{i:02d}_psnr"] = p
            row[f"rl_{i:02d}_ssim"] = s

        fp, fs = m(final_img)
        row["rl_final_psnr"] = fp
        row["rl_final_ssim"] = fs
        all_rows.append(row)

        delta = f"{fp - tp:+.2f} dB" if (fp and tp) else "n/a"
        tqdm.write(f"  {fname:<30}  {elapsed:.1f}s  | Trans={tp}  RL-final={fp}  ({delta})")

    # CSV
    csv_path = os.path.join(out_dir, "metrics.csv")
    if all_rows:
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

    print("[1/3] Loading DehazeFormer...")
    model = load_transformer(MODEL_NAME, PRETRAINED)

    print("\n[2/3] Training RL policy...")
    hazy_dir = os.path.join(DATA_ROOT, 'test', 'hazy')
    gt_dir   = os.path.join(DATA_ROOT, 'test', 'GT')
    agent    = train_rl(model, hazy_dir, gt_dir)
    agent.save(os.path.join(out_dir, "rl_policy.zip"))

    print("\n[3/3] Batch inference...")
    rows = run_batch(model, agent, DATA_ROOT, out_dir)

    # summary
    if rows and rows[0]["has_gt"]:
        def avg(key):
            vals = [r[key] for r in rows if r.get(key) is not None]
            return np.mean(vals) if vals else float('nan')

        print(f"\n{'='*60}")
        print(f"  Transformer  PSNR: {avg('transformer_psnr'):.2f}   SSIM: {avg('transformer_ssim'):.4f}")
        for i in range(1, MAX_RL_STEPS + 1):
            print(f"  RL step {i:02d}   PSNR: {avg(f'rl_{i:02d}_psnr'):.2f}   SSIM: {avg(f'rl_{i:02d}_ssim'):.4f}")
        print(f"  RL final     PSNR: {avg('rl_final_psnr'):.2f}   SSIM: {avg('rl_final_ssim'):.4f}")
        print(f"{'='*60}")
        print(f"Results: {out_dir}")