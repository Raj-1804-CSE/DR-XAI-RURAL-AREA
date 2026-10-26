import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.preprocessing.image_processor import DRImageProcessor
import cv2

class DRDataLoader:
    """
    Handles loading, mapping, and processing of Diabetic Retinopathy datasets.
    Supports CSV-based label mapping.
    """
    def __init__(self, raw_dir, processed_dir, csv_path=None):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.csv_path = csv_path
        self.processor = DRImageProcessor()

        # Ensure processed directory exists
        os.makedirs(self.processed_dir, exist_ok=True)

    def load_labels(self):
        """
        Loads labels from a CSV file.
        Expected columns: 'id_code' (or filename) and 'diagnosis'.
        """
        if self.csv_path is None:
            print("No CSV path provided. Mapping labels based on folder structure...")
            return self._map_from_folders()

        df = pd.read_csv(self.csv_path)
        # Standardize column names for internal use
        # This handles common Kaggle dataset naming (e.g., 'id_code' or 'image_id')
        id_col = [col for col in df.columns if 'id' in col.lower()][0]
        diag_col = [col for col in df.columns if 'diag' in col.lower() or 'level' in col.lower()][0]

        return df[[id_col, diag_col]].rename(columns={id_col: 'image_id', diag_col: 'label'})

    def _map_from_folders(self):
        """
        Fallback: If no CSV, assumes folders are named by their class (0, 1, 2, 3, 4).
        """
        data = []
        for label_dir in os.listdir(self.raw_dir):
            dir_path = os.path.join(self.raw_dir, label_dir)
            if os.path.isdir(dir_path):
                for img_file in os.listdir(dir_path):
                    data.append({'image_id': os.path.join(label_dir, img_file), 'label': int(label_dir)})
        return pd.DataFrame(data)

    def process_and_save(self, df):
        """
        Processes images and saves them to the processed directory.
        """
        processed_paths = []

        print(f"Processing {len(df)} images...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            try:
                img_id = row['image_id']
                # Handle cases where img_id might be an integer (e.g., 000123.jpg)
                if not str(img_id).endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(self.raw_dir, f"{img_id}.jpg")
                else:
                    img_path = os.path.join(self.raw_dir, img_id)

                # Process image
                processed_img = self.processor.preprocess(img_path)

                # Save processed image as numpy array (.npy) for faster loading during training
                # or save as image file. Numpy is better for floating point normalized data.
                save_name = f"{os.path.basename(img_id)}.npy"
                save_path = os.path.join(self.processed_dir, save_name)
                np.save(save_path, processed_img)

                processed_paths.append(save_path)
            except Exception as e:
                print(f"Error processing {img_id}: {e}")
                processed_paths.append(None)

        df['processed_path'] = processed_paths
        return df.dropna(subset=['processed_path'])

if __name__ == "__main__":
    # Example Usage
    # loader = DRDataLoader(
    #     raw_dir='data/raw',
    #     processed_dir='data/processed',
    #     csv_path='data/raw/train.csv'
    # )
    # labels_df = loader.load_labels()
    # final_df = loader.process_and_save(labels_df)
    # final_df.to_csv('data/processed/train_mapped.csv', index=False)
    print("DRDataLoader initialized. Ready for mapping and processing.")
