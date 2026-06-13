import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

from dataset import BreakHisDataset, get_val_test_transforms, get_tta_transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

NUM_CLASSES = 8
BATCH_SIZE = 8
USE_TTA = True   # Set to False to disable test-time augmentation


def predict_with_tta(model, dataset_csv, tta_transforms, batch_size, device):
    """Run TTA: average softmax probabilities over multiple augmented views."""
    all_preds = []
    all_labels = []

    import pandas as pd
    from PIL import Image
    df = pd.read_csv(dataset_csv)

    model.eval()
    with torch.no_grad():
        for idx in range(len(df)):
            row = df.iloc[idx]
            image_path = row["image_path"]
            label = int(row["class_idx"])

            image = Image.open(image_path).convert("RGB")

            avg_probs = None
            for tfm in tta_transforms:
                inp = tfm(image).unsqueeze(0).to(device)
                out = model(inp)
                probs = torch.softmax(out, dim=1).squeeze(0).cpu().numpy()
                if avg_probs is None:
                    avg_probs = probs
                else:
                    avg_probs += probs

            avg_probs /= len(tta_transforms)
            pred = int(np.argmax(avg_probs))

            all_preds.append(pred)
            all_labels.append(label)

            if (idx + 1) % 500 == 0:
                print(f"  TTA progress: {idx + 1}/{len(df)}")

    return all_labels, all_preds


def predict_standard(model, loader, device):
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return all_labels, all_preds


def main():
    print(f"Using device: {DEVICE}")

    # --- Build DenseNet121 model ---
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    
    try:
        model.load_state_dict(torch.load("models/best_densenet121_cutmix_all_mag.pth", map_location=DEVICE))
    except FileNotFoundError:
        print("Warning: models/best_densenet121_cutmix_all_mag.pth not found. Tolong pastikan model DenseNet sudah dilatih.")
        return

    model = model.to(DEVICE)
    model.eval()

    if USE_TTA:
        print("\nRunning evaluation with Test-Time Augmentation (TTA)...")
        tta_transforms = get_tta_transforms()
        print(f"  Number of TTA views: {len(tta_transforms)}")
        all_labels, all_preds = predict_with_tta(
            model,
            dataset_csv="data/test_macenko.csv",
            tta_transforms=tta_transforms,
            batch_size=BATCH_SIZE,
            device=DEVICE
        )
    else:
        print("\nRunning standard evaluation (no TTA)...")
        test_dataset = BreakHisDataset(
            csv_file="data/test_macenko.csv",
            transform=get_val_test_transforms()
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        all_labels, all_preds = predict_standard(model, test_loader, DEVICE)

    acc = accuracy_score(all_labels, all_preds)

    print(f"\nTest Accuracy: {acc:.4f}  ({acc*100:.2f}%)\n")

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()
