import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
import segmentation_models_pytorch as smp
import time
import argparse
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# =====================================================
# 🔧 DEFAULT LOCATIONS
# =====================================================
DEFAULT_DATASET_DIR = r"default dataset path"
DEFAULT_MODEL_DIR = r"default model path"
DEFAULT_MODEL_NAME = r"default model name"

# =====================================================
# TRAINING CONFIG
# =====================================================
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 15
LR = 1e-4

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

        return img, mask


def load_pairs(dataset_dir, split, cls):
    img_dir = os.path.join(dataset_dir, split, "images", cls)
    mask_dir = os.path.join(dataset_dir, split, "masks", cls)

    imgs = sorted(os.listdir(img_dir))
    img_paths, mask_paths = [], []

    for img in imgs:
        idx = img.split("_")[1].split(".")[0]
        img_paths.append(os.path.join(img_dir, img))
        mask_paths.append(os.path.join(mask_dir, f"mask_{idx}.png"))

    return img_paths, mask_paths


# ---------------- MAIN ----------------
def main(dataset_dir, save_path):

    start_time = time.time()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    train_pos_i, train_pos_m = load_pairs(dataset_dir, "train", "positive")
    train_neg_i, train_neg_m = load_pairs(dataset_dir, "train", "negative")

    val_pos_i, val_pos_m = load_pairs(dataset_dir, "val", "positive")
    val_neg_i, val_neg_m = load_pairs(dataset_dir, "val", "negative")

    train_imgs = train_pos_i + train_neg_i
    train_masks = train_pos_m + train_neg_m

    val_imgs = val_pos_i + val_neg_i
    val_masks = val_pos_m + val_neg_m

    print("Train Positive:", len(train_pos_i))
    print("Train Negative:", len(train_neg_i))
    print("Val Positive:", len(val_pos_i))
    print("Val Negative:", len(val_neg_i))

    train_tfms = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2)
    ])

    val_tfms = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE)
    ])

    train_loader = DataLoader(
        SegmentationDataset(train_imgs, train_masks, train_tfms),
        batch_size=BATCH_SIZE, shuffle=True
    )

    val_loader = DataLoader(
        SegmentationDataset(val_imgs, val_masks, val_tfms),
        batch_size=BATCH_SIZE
    )

    # ✅ Removed activation
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=2,
        activation=None
    ).to(DEVICE)

    # ✅ Stable Loss
    dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
    bce = torch.nn.BCEWithLogitsLoss()

    def loss_fn(preds, targets):
        return 0.5 * dice(preds, targets) + 0.5 * bce(preds, targets)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("\n🚀 Training Started\n")

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):

            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            preds = model(imgs)
            loss = loss_fn(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {epoch_loss/len(train_loader):.4f} "
              f"| Time: {time.time() - epoch_start:.2f}s")

    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Model saved at: {save_path}")

# =====================================================
# VALIDATION
# =====================================================
    model.eval()

    image_probs = []
    image_labels = []

    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc="Validation"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            preds = torch.sigmoid(model(imgs))

            for i in range(preds.shape[0]):
                prob = preds[i].mean().item()
                label = 1 if masks[i].sum() > 0 else 0

                image_probs.append(prob)
                image_labels.append(label)

    # --- ROC Curve ---
    fpr, tpr, thresholds = roc_curve(image_labels, image_probs)
    roc_auc = auc(fpr, tpr)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    print("Optimal Threshold:", optimal_threshold)

    # --- Predictions based on threshold ---
    image_preds = [1 if p > optimal_threshold else 0 for p in image_probs]

    # --- Confusion Matrix ---
    TP = sum((p == 1 and l == 1) for p, l in zip(image_preds, image_labels))
    FP = sum((p == 1 and l == 0) for p, l in zip(image_preds, image_labels))
    TN = sum((p == 0 and l == 0) for p, l in zip(image_preds, image_labels))
    FN = sum((p == 0 and l == 1) for p, l in zip(image_preds, image_labels))

    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)

    print("\n🧮 Confusion Matrix (Image-level)")
    print(f"TP: {TP} | FP: {FP}")
    print(f"FN: {FN} | TN: {TN}")
    print(f"Accuracy: {accuracy:.4f}")

    sensitivity = TP / (TP + FN + 1e-8)
    specificity = TN / (TN + FP + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    f1 = 2 * precision * sensitivity / (precision + sensitivity + 1e-8)

    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")

    cm_matrix = np.array([[TN, FP],
                        [FN, TP]])

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_matrix,
                                display_labels=["No Tumor", "Tumor"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix (Image-level)")
    plt.show()

    # --- ROC Plot ---
    print(f"\n📈 Image-Level ROC AUC Score: {roc_auc:.4f}")
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Image-level)")
    plt.legend(loc="lower right")
    plt.show()

    end_time = time.time()
    print("\n⏱️ Total Time:", (end_time - start_time)/60, "minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--save_path", type=str,
                        default=os.path.join(DEFAULT_MODEL_DIR, DEFAULT_MODEL_NAME))

    args = parser.parse_args()
    main(args.dataset_dir, args.save_path)
