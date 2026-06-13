from pathlib import Path
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision import models

from dataset import BreakHisDataset, get_train_transforms, get_val_test_transforms

# ── Device Setup ──────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Hyperparameters ───────────────────────────────────────────────────────────
NUM_CLASSES         = 8
BATCH_SIZE          = 16
EPOCHS              = 80
WARMUP_EPOCHS       = 5
EARLY_STOP_PATIENCE = 20
LEARNING_RATE = 0.01
MOMENTUM      = 0.9
WEIGHT_DECAY  = 1e-4
CUTMIX_ALPHA = 1.0
CUTMIX_PROB  = 0.50   
LABEL_SMOOTHING = 0.05

CLASS_NAMES = [
    "adenosis",
    "ductal_carcinoma",
    "fibroadenoma",
    "lobular_carcinoma",
    "mucinous_carcinoma",
    "papillary_carcinoma",
    "phyllodes_tumor",
    "tubular_adenoma"
]


# ── CutMix ────────────────────────────────────────────────────────────────────

def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_ratio), int(H * cut_ratio)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def cutmix_data(images, labels, alpha=1.0):
    lam   = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0)).to(images.device)
    x1, y1, x2, y2 = rand_bbox(images.size(), lam)
    mixed = images.clone()
    mixed[:, :, x1:x2, y1:y2] = images[index, :, x1:x2, y1:y2]
    lam   = 1 - ((x2 - x1) * (y2 - y1) / (images.size(-1) * images.size(-2)))
    return mixed, labels, labels[index], lam


# ── Training & Validation ─────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()

        if np.random.rand() < CUTMIX_PROB:
            mixed, la, lb, lam = cutmix_data(images, labels, CUTMIX_ALPHA)
            outputs = model(mixed)
            loss    = lam * criterion(outputs, la) + (1 - lam) * criterion(outputs, lb)
            preds   = torch.argmax(outputs, dim=1)
            correct += (lam * (preds == la).sum().item()
                        + (1 - lam) * (preds == lb).sum().item())
        else:
            outputs = model(images)
            loss    = criterion(outputs, labels)
            preds   = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()

        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

    return running_loss / total, correct / total


def validate_one_epoch(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds   = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    return running_loss / total, correct / total


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU Name : {torch.cuda.get_device_name(0)}")
        print(f"GPU Mem  : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("CUDA not available - Falling back to CPU")

    # ── Datasets & Loaders ────────────────────────────────────────────────────
    train_dataset = BreakHisDataset("data/train_all.csv", transform=get_train_transforms())
    val_dataset   = BreakHisDataset("data/val_all.csv",   transform=get_val_test_transforms())

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True if DEVICE.type == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True if DEVICE.type == "cuda" else False
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel        : DenseNet-121")
    print(f"Params       : {total_params:,}")
    print(f"Optimizer    : SGD (lr={LEARNING_RATE}, momentum={MOMENTUM}, wd={WEIGHT_DECAY})")
    print(f"Scheduler    : LinearWarmup({WARMUP_EPOCHS} ep) → CosineAnnealing")
    print(f"Aug (batch)  : CutMix only (p={CUTMIX_PROB})")
    print(f"Label Smooth : {LABEL_SMOOTHING}")

    # ── Class Weights ─────────────────────────────────────────────────────────
    class_counts  = train_dataset.df["class_idx"].value_counts().sort_index()
    total_samples = len(train_dataset.df)
    num_classes   = len(class_counts)
    weights       = total_samples / (num_classes * class_counts)
    class_weights = torch.tensor(weights.values, dtype=torch.float).to(DEVICE)

    print("\nClass counts:")
    print(class_counts)

    # ── Loss + Optimizer + Scheduler ──────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=True          
    )

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_EPOCHS]
    )

    # ── Training Loop ─────────────────────────────────────────────────────────
    best_val_loss  = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    print(f"\n{'='*60}")
    print(f"Training — {EPOCHS} max epochs | early stop patience={EARLY_STOP_PATIENCE}")
    print(f"{'='*60}\n")

    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss,   val_acc   = validate_one_epoch(model, val_loader, criterion)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        print(f"Epoch [{epoch+1:>2}/{EPOCHS}]  ({epoch_time:.1f}s)  lr={current_lr:.2e}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"  ✓ New best val_loss = {val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{EARLY_STOP_PATIENCE} epochs"
                  f"  (best val_loss = {best_val_loss:.4f})")

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping triggered after epoch {epoch + 1}")
            break

    # ── Save ──────────────────────────────────────────────────────────────────
    total_time = time.time() - start_time
    Path("models").mkdir(exist_ok=True)
    save_path = "models/best_densenet121_cutmix_all_mag.pth"
    torch.save(best_model_wts, save_path)

    print(f"\n{'='*60}")
    print(f"Training finished in {total_time / 60:.1f} minutes")
    print(f"Best validation loss : {best_val_loss:.4f}")
    print(f"Model saved to       : {save_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
