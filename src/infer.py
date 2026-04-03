import torch
import torch.nn as nn
from torchvision import models
from PIL import Image

from dataset import get_val_test_transforms

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


def load_model(model_path="models\\best_resnet18_all_mag.pth"):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    return model


def predict_image(image_path, model):
    transform = get_val_test_transforms()

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probabilities).item()

    result = {
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": float(probabilities[pred_idx].item()),
        "all_probabilities": {
            CLASS_NAMES[i]: float(probabilities[i].item())
            for i in range(len(CLASS_NAMES))
        }
    }

    return result


def get_top_k_predictions(probabilities_dict, k=3):
    sorted_preds = sorted(
        probabilities_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )
    return sorted_preds[:k]


def main():
    image_path = input("Enter image path: ").strip()

    model = load_model()
    result = predict_image(image_path, model)

    print("\nPrediction Result")
    print("-----------------")
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f}")

    print("\nTop 3 Predictions:")
    top3 = get_top_k_predictions(result["all_probabilities"], k=3)
    for class_name, prob in top3:
        print(f"{class_name}: {prob:.4f}")


if __name__ == "__main__":
    main()