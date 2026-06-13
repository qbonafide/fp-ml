import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class BreakHisDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        label = int(row["class_idx"])

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

class MacenkoNormalize:
    def __init__(self, target_path=None):
        try:
            import torchstain
        except ImportError:
            raise ImportError("Please install torchstain: pip install torchstain")
            
        import numpy as np
        import pandas as pd
        from PIL import Image
        
        if target_path is None:
            # Grab a default target from train_all.csv for consistency
            try:
                from pathlib import Path
                base_dir = Path(__file__).parent.parent
                df = pd.read_csv(base_dir / "data" / "train_all.csv")
                target_path = str(base_dir / df.iloc[0]["image_path"])
            except Exception as e:
                print(f"Warning: Could not automatically resolve target path: {e}")
                pass
                
        self.normalizer = torchstain.normalizers.MacenkoNormalizer(backend='numpy')
        if target_path:
            try:
                target_img = np.array(Image.open(target_path).convert('RGB'))
                self.normalizer.fit(target_img)
                self.is_fit = True
            except Exception as e:
                print(f"Failed to fit Macenko normalizer: {e}")
                self.is_fit = False
        else:
            self.is_fit = False

    def __call__(self, img):
        if not self.is_fit:
            return img
            
        import numpy as np
        from PIL import Image
        
        try:
            img_np = np.array(img)
            norm, _, _ = self.normalizer.normalize(I=img_np, stains=False)
            return Image.fromarray(norm)
        except Exception:
            return img


# EfficientNet-B5 native resolution: 456x456
def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((456, 456)),
        transforms.RandomRotation(degrees=45),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])


def get_val_test_transforms():
    return transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(456),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_tta_transforms():
    """Returns a list of 6 deterministic transforms for test-time augmentation."""
    base_resize = transforms.Resize(512)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()

    tta_list = []

    crops = [
        transforms.CenterCrop(456),
        transforms.RandomCrop(456),  # will be seeded to corners effectively via Resize+Pad
    ]
    flips = [
        lambda x: x,
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomVerticalFlip(p=1.0),
    ]

    for crop in crops:
        for flip in flips:
            tta_list.append(
                transforms.Compose([base_resize, crop, flip, to_tensor, normalize])
            )

    return tta_list