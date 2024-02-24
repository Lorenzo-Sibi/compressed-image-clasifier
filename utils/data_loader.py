import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path

IMAGE_SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png')
TENSOR_SUPPORTED_EXTENSIONS = ('.npy', '.npz')

class DatasetLoader():
    def __init__(self, main_folder):
        self.main_folder = Path(main_folder)
        self.labels = [folder.name for folder in self.main_folder.iterdir() if folder.is_dir()]

    def load_data(self):
        data_with_labels = []
        for label in self.labels:
            folder = self.main_folder / label
            for file_path in folder.glob('*'):
                # Assuming the files are numpy arrays
                numpy_array = DatasetLoader.load_tensor(file_path)
                data_with_labels.append((numpy_array, label))
        return data_with_labels

    def create_dataset(self):
        data_with_labels = self.load_data()
        data = [{'data': data, 'label': label} for data, label in data_with_labels]
        return pd.DataFrame(data)
    
    @staticmethod
    def load_tensor(file_path):
        if file_path.suffix not in TENSOR_SUPPORTED_EXTENSIONS:
            raise ValueError(f"Extension {file_path.suffix} not suppported.")
        
        try:
            with np.load(file_path) as tensor:
                if file_path.suffix == ".npz":
                    for _, item in tensor.items():
                        tensor = item
            return np.array(tensor).squeeze()
        except Exception as e:
            print(f"Error loading {file_path.stem} file: {str(e)}.", "\nFile path: ", file_path)
            raise RuntimeError(f"Error loading {file_path.stem} file: {str(e)}.") from e