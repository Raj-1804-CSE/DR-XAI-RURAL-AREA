import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import models

class DRExplainer:
    """
    Explainable AI module for Diabetic Retinopathy.
    Implements Grad-CAM to visualize the regions of the retina
    that most influenced the model's classification.
    """
    def __init__(self, model, target_layer):
        """
        Args:
            model: The trained DRClassifier model.
            target_layer: The layer to extract gradients from (usually the last conv layer).
        """
        self.model = model
        self.model.eval()
        self.target_layer = target_layer

        # Hooks to capture gradients and activations
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=None):
        """
        Generates a Grad-CAM heatmap for a given input image.

        Args:
            input_tensor: Preprocessed image tensor [1, 3, H, W].
            class_idx: The class index to explain. If None, uses the predicted class.
        """
        # Forward pass
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output)

        # Zero out gradients
        self.model.zero_grad()

        # Backward pass for the specific class
        loss = output[0, class_idx]
        loss.backward()

        # Pool the gradients across the channels
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weight the activations by the pooled gradients
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        # Average the weighted activations across the channel dimension
        heatmap = torch.mean(self.activations, dim=1).squeeze()

        # ReLU on the heatmap to keep only positive contributions
        heatmap = F.relu(heatmap)

        # Normalize the heatmap between 0 and 1
        heatmap /= torch.max(heatmap) if torch.max(heatmap) != 0 else 1

        return heatmap.detach().cpu().numpy()

    def apply_heatmap_to_image(self, heatmap, original_img):
        """
        Overlays the Grad-CAM heatmap onto the original fundus image.

        Args:
            heatmap: The numpy heatmap.
            original_img: The original image (RGB) as a numpy array.
        """
        # Resize heatmap to match original image size
        heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

        # Convert to RGB heatmap (Jet colormap)
        heatmap_color = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )

        # Convert original image to uint8 if it's normalized [0, 1]
        if original_img.max() <= 1.0:
            original_img = (original_img * 255).astype(np.uint8)

        # Blend the heatmap with the original image
        superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)

        return superimposed_img

if __name__ == "__main__":
    print("DRExplainer initialized. Ready to generate clinical explanations.")
