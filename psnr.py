import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import torchvision.transforms as transforms
from PIL import Image

# 1. Load and prepare images (assuming range [0, 1] for this example)
transform = transforms.ToTensor()
# Load images from paths, e.g., 'path/to/im1.png' and 'path/to/im2.png'
im1 = Image.open(r"C:\Users\adith\Downloads\rl_step_5(1).png")
im2 = Image.open(r"C:\Users\adith\Downloads\46_GT.png")

# Create dummy tensors for demonstration (batch, channels, height, width)
# Replace with your actual image tensors
x = transform(im1).unsqueeze(0)
y = transform(im2).unsqueeze(0)

# 2. Initialize metrics (specify data_range, often 1.0 or 255.0)
psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)

# 3. Compute the metrics
psnr_value = psnr_metric(x, y)
ssim_value = ssim_metric(x, y)

print(f"PSNR: {psnr_value.item()}")
print(f"SSIM: {ssim_value.item()}")