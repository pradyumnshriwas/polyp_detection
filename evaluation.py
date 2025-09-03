import os
import torch
import torch.nn.functional as F
import numpy as np
import random
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score
import segmentation_models_pytorch as smp

# ✅ Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Paths
IMG_DIR = "/Users/aniketsingh/Anik8base/Academia/Polyp_Detection/input/PNG/Original"
MASK_DIR = "/Users/aniketsingh/Anik8base/Academia/Polyp_Detection/input/PNG/Ground"

# ✅ UNet Model Definition
class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 1, kernel_size=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ✅ Load Models
unet_model = UNet().to(device)
unet_model.load_state_dict(torch.load('unet_polyp.pth'))
unet_model.eval()

unetpp_model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1, activation="sigmoid").to(device)
unetpp_model.load_state_dict(torch.load('unet_plus_plus_polyp.pth'))
unetpp_model.eval()

# ✅ Data Preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# ✅ Metrics Calculation
def calculate_metrics(pred, true_mask):
    pred = (pred > 0.5).astype(np.uint8)
    true_mask = (true_mask > 0.5).astype(np.uint8)
    pred = np.resize(pred, true_mask.shape)

    iou = np.sum((pred & true_mask)) / np.sum((pred | true_mask) + 1e-6)
    acc = accuracy_score(true_mask.flatten(), pred.flatten())
    f1 = f1_score(true_mask.flatten(), pred.flatten())
    recall = recall_score(true_mask.flatten(), pred.flatten())
    
    return acc, iou, f1, recall

# ✅ Evaluate Models
image_files = [f for f in os.listdir(IMG_DIR) if os.path.isfile(os.path.join(IMG_DIR, f))]
image_files = random.sample(image_files, min(10, len(image_files)))
metrics_unet, metrics_unetpp = [], []

for img_file in image_files:
    img_path = os.path.join(IMG_DIR, img_file)
    mask_path = os.path.join(MASK_DIR, img_file)

    if not os.path.exists(mask_path):
        print(f"Mask not found for {img_file}, skipping...")
        continue

    image = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    image = transform(image).unsqueeze(0).to(device)
    mask = transforms.ToTensor()(mask).numpy().squeeze()

    with torch.no_grad():
        unet_pred = unet_model(image).cpu().numpy().squeeze()
        unetpp_pred = unetpp_model(image).cpu().numpy().squeeze()

    metrics_unet.append(calculate_metrics(unet_pred, mask))
    metrics_unetpp.append(calculate_metrics(unetpp_pred, mask))

# ✅ Generate Combined Graphs
def plot_combined_metrics(metrics_unet, metrics_unetpp):
    metric_names = ['Accuracy', 'IoU', 'F1 Score', 'Recall']
    plt.figure(figsize=(10, 8))

    for i, name in enumerate(metric_names):
        plt.subplot(2, 2, i+1)
        unet_values = [m[i] for m in metrics_unet]
        unetpp_values = [m[i] for m in metrics_unetpp]
        plt.plot(unet_values, label='UNet', marker='o')
        plt.plot(unetpp_values, label='UNet++', marker='s')
        plt.xlabel('Image Index')
        plt.ylabel(name)
        plt.title(f'{name} Comparison')
        plt.legend()
        plt.grid()

    plt.tight_layout()
    plt.show()

if metrics_unet and metrics_unetpp:
    plot_combined_metrics(metrics_unet, metrics_unetpp)
else:
    print('No valid comparisons available.')

print('✅ Evaluation Complete!')
