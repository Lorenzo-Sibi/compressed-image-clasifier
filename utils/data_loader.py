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

    def load_dataset(self):
        data = []

        for label in self.labels:
            folder = self.main_folder / label
            file_paths = [str(file_path) for file_path in folder.glob('*') if file_path.suffix in TENSOR_SUPPORTED_EXTENSIONS]
            latenst_spaces = [DatasetLoader.load_tensor(file_path) for file_path in folder.glob('*') if file_path.suffix in TENSOR_SUPPORTED_EXTENSIONS]
            dataset = tf.data.Dataset.from_tensor_slices(latenst_spaces)
            
            # Zip dataset with labels
            dataset = dataset.map(lambda x: (x, label))
            # for x in dataset:
            #     print(x)
            # dataset = dataset.map(map_function)
            data.append(dataset)
        
        # Concatenate datasets from different labels
        dataset = data[0]
        for i in range(1, len(data)):
            dataset = dataset.concatenate(data[i])
        
        return dataset

    @staticmethod
    def _load_tensor(file_path):
        try:
            file_path = Path(file_path.decode())  # Convert back to Path object
            print("Filepath", file_path)
            if file_path.suffix == ".npz":
                with np.load(file_path) as tensor:
                    for _, item in tensor.items():
                        tensor = item
                tensor = np.array(tensor)
            else:  # Assuming it's .npy
                tensor = np.load(file_path)
            return tensor.squeeze(), file_path.parent.name  # Return data and label
        except Exception as e:
            print(f"Error loading {file_path.stem} file: {str(e)}.", "\nFile path: ", file_path)
            raise RuntimeError(f"Error loading {file_path.stem} file: {str(e)}.") from e
        
        

    def create_dataset(self):
        data_with_labels, max_shape = self.load_data()
        data = []
        for numpy_array, label, shape in data_with_labels:
            # Calculate padding dimensions
            pad_width = [(0, max_dim - cur_dim) if max_dim > cur_dim else (0, 0) for cur_dim, max_dim in zip(shape, max_shape)]
            
            # Pad or resize each array to match max_shape
            padded_array = np.pad(numpy_array, pad_width=pad_width, mode='constant', constant_values=0)
            data.append({'data': padded_array, 'label': label})
        return pd.DataFrame(data)

    @staticmethod
    def load_tensor(file_path):
        # file_path = tf.get_static_value(tf_tensor)
        # file_path = Path(file_path)
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


def map_function(element):
# Estrai il percorso del file npz e il label dalla tupla
    file_path, label = element
    
    # Carica il file npz usando np.load()
    npz_data = DatasetLoader.load_tensor(file_path)
    tupla = tuple(npz_data, label)
    return tupla