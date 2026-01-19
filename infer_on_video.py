import cv2
import torch
import numpy as np
import os
from infer_utils import load_model

# ================= CONFIG =================
MODEL_NAME = "dehazeformer-t"
CKPT_PATH = "save_models/outdoor/dehazeformer-t.pth"
INPUT_VIDEO = r"H:\DRDO-Real Data\roorkee_video\roorkee_video\CCD\2026-01-04 07-29-33.mkv"
OUTPUT_VIDEO = "output_dehazed.mp4"
DEVICE = "cuda"
# =========================================

@torch.no_grad()
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img * 2.0 - 1.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img.to(DEVICE)

@torch.no_grad()
def postprocess(tensor):
    tensor = tensor.clamp_(-1, 1)
    tensor = (tensor * 0.5 + 0.5).cpu().squeeze(0)
    img = tensor.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def main():
    model = load_model(MODEL_NAME, CKPT_PATH, DEVICE)
    cap = cv2.VideoCapture(INPUT_VIDEO)

    if not cap.isOpened():
        print("❌ Cannot open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        inp = preprocess(frame)
        out_frame = model(inp)
        out_img = postprocess(out_frame)
        out_img = cv2.resize(out_img, (w, h))

        out.write(out_img)
        frame_id += 1
        print(f"Processed frame {frame_id}", end="\r")

    cap.release()
    out.release()
    print("\n✅ Video saved:", OUTPUT_VIDEO)

if __name__ == "__main__":
    main()
