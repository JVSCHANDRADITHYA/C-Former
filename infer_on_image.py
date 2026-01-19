import cv2
import torch
import numpy as np
from collections import OrderedDict
from models import *

# ================= CONFIG =================
MODEL_NAME = "dehazeformer-b"   # must match model class
CKPT_PATH = "save_models/outdoor/dehazeformer-b.pth"
INPUT_IMAGE = "roorkee_dataset\\frame_00046.png"
OUTPUT_IMAGE = "output_dehazed.jpg"
DEVICE = "cuda"
# =========================================


def load_model():
    model = eval(MODEL_NAME.replace("-", "_"))()
    model.to(DEVICE)
    model.eval()

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    state_dict = ckpt["state_dict"]

    new_state = OrderedDict()
    for k, v in state_dict.items():
        new_state[k[7:]] = v  # remove 'module.'

    model.load_state_dict(new_state)
    return model


@torch.no_grad()
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img * 2.0 - 1.0            # [0,1] → [-1,1]
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img.to(DEVICE)


@torch.no_grad()
def postprocess(tensor):
    tensor = tensor.clamp_(-1, 1)
    tensor = tensor * 0.5 + 0.5      # [-1,1] → [0,1]
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def main():
    model = load_model()

    img = cv2.imread(INPUT_IMAGE)
    if img is None:
        print("❌ Image not found:", INPUT_IMAGE)
        return

    inp = preprocess(img)
    out = model(inp)
    out_img = postprocess(out)

    cv2.imwrite(OUTPUT_IMAGE, out_img)
    print("✅ Saved:", OUTPUT_IMAGE)


if __name__ == "__main__":
    main()
