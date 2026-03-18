import os
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm

PROJECT_ROOT = Path.cwd()

DATA_ROOT = PROJECT_ROOT / "Deeta"
SAVE_DIR = PROJECT_ROOT / "models"

BATCH_SIZE = 32
IMAGE_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 2.5e-5


class AlbumentationsFolder(Dataset):
    def __init__(self, root, transform):
        self.inner = datasets.ImageFolder(root=str(root))
        self.samples = self.inner.samples
        self.targets = [s[1] for s in self.samples]
        self.class_to_idx = self.inner.class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image_np = np.array(image)
        aug = self.transform(image=image_np)
        tensor = aug["image"]
        return tensor, label


def build_dataloaders(data_root):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_tfms = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02, p=0.75),
        A.Affine(translate_percent=0.02, scale=0.1, rotate=10, p=0.5),
        A.Normalize(mean=imagenet_mean, std=imagenet_std),
        ToTensorV2()
    ])

    valid_tfms = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=imagenet_mean, std=imagenet_std),
        ToTensorV2()
    ])

    if (data_root / "Train").exists():
        train_dir = data_root / "Train"
        valid_dir = data_root / "Validation"
    else:
        subfolders = [d for d in data_root.iterdir() if d.is_dir()]
        found = False
        for sub in subfolders:
            if (sub / "Train").exists():
                train_dir = sub / "Train"
                valid_dir = sub / "Validation"
                found = True
                break

        if not found:
            raise FileNotFoundError(f"Could not find 'Train' folder inside {data_root.absolute()}")

    print(f"Loading data from: {train_dir}")

    train_ds = AlbumentationsFolder(train_dir, transform=train_tfms)
    valid_ds = AlbumentationsFolder(valid_dir, transform=valid_tfms)

    workers = 2

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=workers,
        persistent_workers=(workers > 0)
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=workers,
        persistent_workers=(workers > 0)
    )

    return train_loader, valid_loader


def compute_pos_weight(dataset):
    labels = np.array(dataset.targets)
    num_pos = max(1, int((labels == 1).sum()))
    num_neg = max(1, int((labels == 0).sum()))
    pos_weight_value = float(num_neg) / float(num_pos)
    return torch.tensor([pos_weight_value], dtype=torch.float32)


def forward_batch(model, images, targets, loss_fn, device, use_amp):
    images = images.to(device, non_blocking=True)
    targets = targets.float().unsqueeze(1).to(device, non_blocking=True)
    device_type = "cuda" #if device.type == "cuda" else "cpu"
    with torch.amp.autocast(device_type=device_type, enabled=use_amp):
        logits = model(images)
        loss = loss_fn(logits, targets)

    probs = torch.sigmoid(logits.detach())
    return loss, probs, targets.detach()


def compute_metrics_from_confusion(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(np.int64)
    y_prob = np.asarray(y_prob).astype(np.float32)
    preds = (y_prob >= threshold).astype(np.int64)

    tp = int(np.sum((preds == 1) & (y_true == 1)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))
    tn = int(np.sum((preds == 0) & (y_true == 0)))

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / max(1, total)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {"accuracy": float(accuracy), "f1": float(f1)}


def train_one_epoch(model, loader, optimizer, scaler, loss_fn, device, use_amp):
    model.train()
    running_loss = 0.0
    all_probs = []
    all_targets = []

    bar = tqdm(loader, desc="Train", leave=False, file=sys.stdout)

    for images, labels in bar:
        images.to(device)
        labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, probs, targets = forward_batch(model, images, labels, loss_fn, device, use_amp)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.item()) * images.size(0)
        all_probs.append(probs.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())
        bar.set_postfix(loss=f"{loss.item():.4f}")

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    metrics = compute_metrics_from_confusion(y_true, y_prob)
    return running_loss / len(loader.dataset), metrics


def evaluate(model, loader, loss_fn, device, use_amp):
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_targets = []

    bar = tqdm(loader, desc="Valid", leave=False, file=sys.stdout)

    with torch.no_grad():
        for images, labels in bar:
            loss, probs, targets = forward_batch(model, images, labels, loss_fn, device, use_amp)
            running_loss += float(loss.item()) * images.size(0)
            all_probs.append(probs.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    metrics = compute_metrics_from_confusion(y_true, y_prob)
    return running_loss / len(loader.dataset), metrics


if __name__ == "__main__":
    if not SAVE_DIR.exists():
        print(f"Creating save directory: {SAVE_DIR.absolute()}")
        SAVE_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU found! Aborting.")
    device = torch.device("cuda")
    #device = torch.device("cuda")# if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print(f"Using Device: {device}")
    print(f"Project Root: {PROJECT_ROOT.absolute()}")
    print(f"Looking for data in: {DATA_ROOT.absolute()}")

    try:
        train_loader, valid_loader = build_dataloaders(DATA_ROOT)

        # Initialize model
        print("Initializing model...")
        model = timm.create_model("efficientvit_b0", pretrained=True, num_classes=1)
        model = model.to(device)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=LEARNING_RATE, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        pos_weight = compute_pos_weight(train_loader.dataset).to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        scaler = torch.amp.GradScaler(enabled=use_amp)

        print(f"Starting training for {EPOCHS} epochs...")
        for epoch in range(1, EPOCHS + 1):
            print(f"\nEpoch {epoch}/{EPOCHS}")

            train_loss, train_metrics = train_one_epoch(
                model, train_loader, optimizer, scaler, loss_fn, device, use_amp
            )
            print(
                f"Train | Loss: {train_loss:.4f} | Acc: {train_metrics['accuracy']:.4f} | F1: {train_metrics['f1']:.4f}")

            valid_loss, valid_metrics = evaluate(
                model, valid_loader, loss_fn, device, use_amp
            )
            print(
                f"Valid | Loss: {valid_loss:.4f} | Acc: {valid_metrics['accuracy']:.4f} | F1: {valid_metrics['f1']:.4f}")

            scheduler.step()

        final_path = SAVE_DIR / "deepfake_model_final.pt"
        torch.save(model.state_dict(), final_path)
        print(f"\nTraining Complete! Model saved to {final_path.absolute()}")

    except FileNotFoundError as fnf_error:
        print(f"\n[ERROR] Path Issue: {fnf_error}")
        print("Please check that your 'Deeta' folder is inside your PyCharm project directory.")
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"\n[ERROR] An unexpected error occurred: {e}")







