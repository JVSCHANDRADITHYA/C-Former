import cv2
import torch
import numpy as np
from infer_utils import load_model

# ================= CONFIG =================
MODEL_NAME = "dehazeformer-t"
CKPT_PATH = "save_models/indoor/dehazeformer-t.pth"
DEVICE = "cuda"
CAMERA_ID = 0
# =========================================

def estimate_brightness(bgr_img):
    """Return mean luminance [0–255]."""
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 2].mean()


def gamma_correction(img, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255
                      for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


def clahe_luminance(bgr_img):
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def enhance_low_light(img):
    """
    Adaptive low-light enhancement.
    Only boosts when scene is dark.
    """
    brightness = estimate_brightness(img)

    # threshold tuned for webcams
    if brightness < 90:
        # gamma < 1 brightens
        img = gamma_correction(img, gamma=0.6)
        img = clahe_luminance(img)

    return img


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
    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        print("❌ Camera not found")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        inp = preprocess(frame)
        out = model(inp)

        out_img = postprocess(out)
        out_img = enhance_low_light(out_img)

        h, w = frame.shape[:2]
        out_img = cv2.resize(out_img, (w, h))

        combined = np.hstack((frame, out_img))
        cv2.imshow("Input | Dehazed + Low-Light Enhanced", combined)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
