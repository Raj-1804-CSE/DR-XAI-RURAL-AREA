import torch
import torch.nn as nn
from torchvision import models

class DRModelBase(nn.Module):
    """
    Base wrapper to standardize different architectures for the ensemble.
    """
    def __init__(self, model_type, num_classes=5):
        super(DRModelBase, self).__init__()

        if model_type == 'efficientnet':
            self.net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            in_features = self.net.classifier[1].in_features
            self.net.classifier = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        elif model_type == 'resnet':
            self.net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            in_features = self.net.fc.in_features
            self.net.fc = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        elif model_type == 'densenet':
            self.net = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            in_features = self.net.classifier.in_features
            self.net.classifier = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )

    def forward(self, x):
        return self.net(x)

class DREnsembleClassifier(nn.Module):
    """
    Professional Ensemble Classifier.
    Combines EfficientNet, ResNet, and DenseNet to maximize diagnostic reliability.
    """
    def __init__(self, num_classes=5):
        super(DREnsembleClassifier, self).__init__()

        # The three specialized models
        self.model_eff = DRModelBase('efficientnet', num_classes)
        self.model_res = DRModelBase('resnet', num_classes)
        self.model_den = DRModelBase('densenet', num_classes)

    def forward(self, x):
        # Get predictions from all three models
        out_eff = self.model_eff(x)
        out_res = self.model_res(x)
        out_den = self.model_den(x)

        # Soft-Voting: Average the probabilities
        # This reduces the impact of a single model's error
        ensemble_out = (out_eff + out_res + out_den) / 3.0

        return ensemble_out

def get_model(num_classes=5):
    return DREnsembleClassifier(num_classes=num_classes)

if __name__ == "__main__":
    # Test the ensemble with a dummy input
    ensemble = DREnsembleClassifier()
    dummy_input = torch.randn(1, 3, 224, 224)
    output = ensemble(dummy_input)
    print(f"Ensemble initialized. Output shape: {output.shape}") # Expected: [1, 5]
