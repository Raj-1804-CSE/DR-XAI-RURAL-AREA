import torch
import torch.nn as nn
from torchvision import models

class DRClassifier(nn.Module):
    """
    Diabetic Retinopathy Classifier using a pre-trained EfficientNet.
    EfficientNet is chosen for its balance between accuracy and parameter
    efficiency, making it suitable for deployment in rural clinics.
    """
    def __init__(self, num_classes=5):
        super(DRClassifier, self).__init__()

        # Load pre-trained EfficientNet-B0
        # Weights=EfficientNet_B0_Weights.DEFAULT is the modern PyTorch way
        self.base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        # Freeze early layers to preserve pre-trained features (Transfer Learning)
        # We freeze the first few blocks and only train the later ones and the head
        for param in self.base_model.features[:4].parameters():
            param.requires_grad = False

        # Replace the classifier head
        # EfficientNet-B0 output features: 1280
        in_features = self.base_model.classifier[1].in_features

        self.base_model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

def get_model(num_classes=5):
    return DRClassifier(num_classes=num_classes)

if __name__ == "__main__":
    # Test the model with a dummy input
    model = DRClassifier()
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Model initialized. Output shape: {output.shape}") # Expected: [1, 5]
