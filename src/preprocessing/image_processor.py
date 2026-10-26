import cv2
import numpy as np
from PIL import Image

class DRImageProcessor:
    """
    Preprocessing class for Diabetic Retinopathy fundus images.
    Includes cropping, color enhancement, and normalization.
    """
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def crop_black_borders(self, image):
        """
        Crops the black padding around the retinal image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = gray > 10
        coords = np.argwhere(mask)
        if coords.size == 0:
            return image

        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        return image[y0:y1, x0:x1]

    def apply_ben_graham_filter(self, image):
        """
        Enhances features (vessels, exudates) by subtracting local average color.
        This is a standard technique for DR image analysis.
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(image)

        # Apply Gaussian blur to the L channel to get local average
        blurred = cv2.GaussianBlur(l, (31, 31), 0)
        # Subtract blurred image from original L channel
        l_enhanced = cv2.addWeighted(l, 4, blurred, -4, 128)

        merged = cv2.merge([l_enhanced, a, b])
        enhanced_rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        return enhanced_rgb

    def preprocess(self, image_path):
        """
        Full pipeline: Load -> Crop -> Enhance -> Resize -> Normalize
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")

        # 1. Crop borders
        img = self.crop_black_borders(img)

        # 2. Enhance features
        img = self.apply_ben_graham_filter(img)

        # 3. Resize
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_CUBIC)

        # 4. Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        return img

if __name__ == "__main__":
    # Simple test to verify the processor
    print("DRImageProcessor initialized. Ready for data processing.")
