import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# =====================================================
# 🔧 PATHS (CHANGE ONLY THIS)
# =====================================================
DATASET_DIR = r"E:\pancreas\dataset1"
MODEL_PATH = r"E:\Panc\curmodels\resnet18_unet_pancreas.pth"

# =====================================================
IMG_SIZE = 256
BATCH_SIZE = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- DATASET ----------------
class SegmentationDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]

        img = img / 255.0
        mask = mask / 255.0

        img = torch.tensor(img).permute(2, 0, 1).float()
        mask = torch.tensor(mask).unsqueeze(0).float()

        return img, mask, self.img_paths[idx], self.mask_paths[idx]


def load_pairs(dataset_dir, split, cls):
    base_dir = os.path.join(dataset_dir, split)
    img_dir = os.path.join(base_dir, "images", cls)
    mask_dir = os.path.join(base_dir, "masks", cls)

    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    imgs = sorted(os.listdir(img_dir))
    img_paths, mask_paths = [], []

    for img in imgs:
        idx = img.split("_")[1].split(".")[0]
        img_paths.append(os.path.join(img_dir, img))
        mask_paths.append(os.path.join(mask_dir, f"mask_{idx}.png"))

    return img_paths, mask_paths


# ---------------- RISK LEVEL ----------------
def get_risk_level(pred_mask):
    tumor_pixels = pred_mask.sum()
    total_pixels = pred_mask.shape[0] * pred_mask.shape[1]
    tumor_ratio = tumor_pixels / total_pixels

    if tumor_pixels == 0:
        return "No Tumor"
    elif tumor_ratio < 0.002:      # < 0.2%
        return "Low Risk"
    elif tumor_ratio < 0.01:       # < 1%
        return "Medium Risk"
    else:                          # >= 1%
        return "High Risk"


# ---------------- MAIN ----------------
def main():
    val_pos_i, val_pos_m = load_pairs(DATASET_DIR, "Test", "positive")
    val_neg_i, val_neg_m = load_pairs(DATASET_DIR, "Test", "negative")

    val_imgs = val_pos_i + val_neg_i
    val_masks = val_pos_m + val_neg_m

    val_tfms = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE)])

    val_loader = DataLoader(
        SegmentationDataset(val_imgs, val_masks, val_tfms),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    ).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    results = []

    print("\n🔍 Running Testing & Preparing Results...\n")

    with torch.no_grad():
        for img, mask, img_path, mask_path in tqdm(val_loader):
            img = img.to(DEVICE)
            pred = model(img)

            pred_np = (pred[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
            risk = get_risk_level(pred_np)

            results.append({
                "img_path": img_path[0],
                "pred_mask": pred_np,
                "risk": risk
            })

    print("\n✅ Testing completed!")
    print(f"Total Testing  samples: {len(results)}\n")

    # ---------------- PRINT ALL SERIALS + RISKS ----------------
    print("📋 All Testing Samples (Serial No & Risk Level):\n")
    for i, item in enumerate(results):
        print(f"| CT Scan (Serial No: {i}) | Predicted Mask (Risk: {item['risk']}) |")

    # ---------------- INTERACTIVE VIEWER ----------------
    while True:
        user_input = input(f"\nEnter serial number to view (0 to {len(results)-1}) or type 'exit': ")

        if user_input.lower() == "exit":
            print("👋 Exiting viewer...")
            break

        if not user_input.isdigit():
            print("❌ Please enter a valid number.")
            continue

        idx = int(user_input)
        if idx < 0 or idx >= len(results):
            print("❌ Invalid serial number.")
            continue

        item = results[idx]

        ct_img = cv2.imread(item["img_path"], cv2.IMREAD_GRAYSCALE)
        pred_mask = (item["pred_mask"] * 255).astype(np.uint8)

        ct_img = cv2.resize(ct_img, (IMG_SIZE, IMG_SIZE))
        pred_mask = cv2.resize(pred_mask, (IMG_SIZE, IMG_SIZE))

        print(f"\n| CT Scan (Serial No: {idx}) | Predicted Mask (Risk: {item['risk']}) |")

        plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(ct_img, cmap="gray")
        plt.title(f"CT Scan\nSerial No: {idx}")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(pred_mask, cmap="gray")
        plt.title(f"Predicted Mask\nRisk: {item['risk']}")
        plt.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
