import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os

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

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, x, class_idx):
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        outputs = self.model(x)
        score = outputs[0, class_idx]
        
        # Backward pass
        score.backward()
        
        # Calculate CAM
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = F.relu(cam)
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-8)
        
        return cam.cpu().detach().numpy()


def load_model(model_path="models/best_densenet121_cutmix_all_mag.pth"):
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print(f"Warning: {model_path} not found. Grad-CAM might throw an error or give bad outputs.")

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
    
    # Generate Heatmap — use last feature block as target layer
    # For DenseNet we hook to the 'features' sequential module root which holds the final feature maps
    grad_cam = GradCAM(model, model.features)
    image_tensor.requires_grad_()
    cam = grad_cam.generate(image_tensor, pred_idx)
    
    # Overlay heatmap
    render_size = (image_tensor.shape[3], image_tensor.shape[2])
    original_img = image.resize(render_size)
    cam_resized = np.array(Image.fromarray(cam).resize(render_size, Image.Resampling.BILINEAR))
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(original_img)
    plt.imshow(cam_resized, cmap='jet', alpha=0.5)
    plt.title(f"Grad-CAM ({CLASS_NAMES[pred_idx]})")
    plt.axis("off")
    
    plt.tight_layout()
    output_heatmap = "prediction_heatmap_densenet.png"
    plt.savefig(output_heatmap)
    print(f"Heatmap saved as '{output_heatmap}'")
    plt.close()

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
    
    if not os.path.exists(image_path):
        print("Error: Image path not found.")
        return

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
