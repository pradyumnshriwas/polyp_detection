import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
from source.network import UNetPP  # Ensure this matches your model definition

# Define transformations
val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(),
    ToTensorV2()
])

# Function to load and preprocess an image
def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Error: Image file not found -> {image_path}")

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Error: OpenCV failed to read image -> {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    transformed = val_transform(image=image)["image"]
    return transformed.unsqueeze(0)  # Add batch dimension

# Function to load a mask
def load_mask(mask_path):
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Error: Mask file not found -> {mask_path}")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Error: OpenCV failed to read mask -> {mask_path}")

    mask = cv2.resize(mask, (256, 256))  # Resize to match model input
    return mask / 255.0  # Normalize to [0,1]

# Function to calculate accuracy metrics
def dice_coefficient(y_true, y_pred, epsilon=1e-6):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + epsilon) / (np.sum(y_true) + np.sum(y_pred) + epsilon)

def iou(y_true, y_pred, epsilon=1e-6):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return (intersection + epsilon) / (union + epsilon)

def pixel_accuracy(y_true, y_pred):
    y_pred = (y_pred > 0.5).astype(np.float32)  # Thresholding
    return np.sum(y_pred == y_true) / y_true.size

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
parser.add_argument("--mask_path", type=str, required=True, help="Path to ground truth mask")
parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
args = parser.parse_args()

# Load and preprocess image
image = load_image(args.image_path)
mask = load_mask(args.mask_path)

# Initialize model
num_classes = 1  # Adjust to match training configuration
model = UNetPP(num_classes=num_classes)

# Load model weights with strict=False to avoid mismatched keys
model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')), strict=False)
model.eval()  # Set to evaluation mode

# Perform inference
with torch.no_grad():
    output = model(image)
    output = torch.sigmoid(output).squeeze().numpy()  # Convert to numpy

# Compute accuracy metrics
dice = dice_coefficient(mask, output)
iou_score = iou(mask, output)
acc = pixel_accuracy(mask, output)

# Display input, ground truth, prediction, and metrics
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.imshow(cv2.imread(args.image_path)[..., ::-1])  # Convert BGR to RGB
plt.title("Input Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap="gray")
plt.title("Ground Truth Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(output, cmap="gray")
plt.title(f"Predicted Mask\nDice: {dice:.4f} | IoU: {iou_score:.4f} | Acc: {acc:.4f}")
plt.axis("off")

plt.tight_layout()
plt.show()
