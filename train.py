import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp  # ✅ UNet++ library

# ✅ Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ Dataset Paths
TRAIN_IMG_DIR = "/Users/aniketsingh/Anik8base/Academia/Polyp_Detection/input/PNG/Original"
TRAIN_MASK_DIR = "/Users/aniketsingh/Anik8base/Academia/Polyp_Detection/input/PNG/Ground Truth"

# ✅ Data Augmentation
transform = transforms.Compose([
    transforms.Resize((256, 256)),  
    transforms.ToTensor(),           
    transforms.Normalize(mean=[0.5], std=[0.5])  
])

# ✅ Custom Dataset Class
class PolypDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_list = sorted(os.listdir(img_dir))
        self.mask_list = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.img_dir, self.img_list[idx])
            mask_path = os.path.join(self.mask_dir, self.mask_list[idx])

            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            if self.transform:
                image = self.transform(image)

            mask = mask.resize((256, 256), Image.NEAREST)
            mask = transforms.ToTensor()(mask)

            return image, mask
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None  

# ✅ Data Loaders
train_dataset = PolypDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)  

# ✅ UNet Model (Baseline)
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ✅ Initialize Models
unet_model = UNet().to(device)
unetpp_model = smp.UnetPlusPlus(
    encoder_name="resnet34",  
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation="sigmoid"
).to(device)

# ✅ Loss and Optimizer
criterion = nn.BCELoss()
unet_optimizer = optim.AdamW(unet_model.parameters(), lr=0.001)
unetpp_optimizer = optim.AdamW(unetpp_model.parameters(), lr=0.001)

# ✅ Training Function
def train_model(model, optimizer, model_name, num_epochs=5):
    model.train()
    loss_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"{model_name} - Epoch {epoch+1}/{num_epochs}", leave=False)

        for images, masks in progress_bar:
            if images is None or masks is None:
                continue 

            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"{model_name} - Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    return loss_history

# ✅ Train both models
print("\nTraining UNet...")
unet_loss_history = train_model(unet_model, unet_optimizer, "UNet")

print("\nTraining UNet++...")
unetpp_loss_history = train_model(unetpp_model, unetpp_optimizer, "UNet++")

# ✅ Save Models
torch.save(unet_model.state_dict(), "unet_polyp.pth")
torch.save(unetpp_model.state_dict(), "unet_plus_plus_polyp.pth")

print("✅ Models saved!")

# ✅ Compare Results
plt.figure(figsize=(8, 5))
plt.plot(range(1, 6), unet_loss_history, label="UNet Loss", marker="o")
plt.plot(range(1, 6), unetpp_loss_history, label="UNet++ Loss", marker="s")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Comparison: UNet vs UNet++")
plt.legend()
plt.grid()
plt.show()
