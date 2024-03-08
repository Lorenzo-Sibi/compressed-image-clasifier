import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path

IMAGE_SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png')
TENSOR_SUPPORTED_EXTENSIONS = ('.npz')

class DatasetLoader():
    def __init__(self, main_folder):
        self.main_folder = Path(main_folder)
        self.labels = [folder.name for folder in self.main_folder.iterdir() if folder.is_dir()]
        self.max_shape = self.calculate_max_shape()
        
    def calculate_max_shape(self):
        max_shape = None

        # Iterate through each subfolder in the dataset folder
        for subfolder in self.main_folder.glob('*'):
            if subfolder.is_dir():
                # Iterate through .npz files in the subfolder
                for file_path in subfolder.glob('*.npz'):
                    try:
                        with np.load(file_path) as tensor:
                            for _, item in tensor.items():
                                tensor_data = item.squeeze()  # Squeeze the tensor data
                        # Update the maximum shape
                        if max_shape is None:
                            max_shape = tensor_data.shape
                        else:
                            max_shape = tuple(max(dim_len, tensor_data.shape[i]) for i, dim_len in enumerate(max_shape))
                    except Exception as e:
                        print(f"Error loading {file_path.stem} file: {str(e)}.", "\nFile path: ", file_path)
                        raise RuntimeError(f"Error loading {file_path.stem} file: {str(e)}.") from e
                    finally:
                        tensor.close()  # Close the .npz file to free up memory

        return max_shape
    
    def load_dataset(self):
        def generator():
            max_shape = self.max_shape  # Maximum shape based on the largest tensor shape
            
            for label in self.labels:
                folder = self.main_folder / label
                file_paths = [str(file_path) for file_path in folder.glob('*') if file_path.suffix in TENSOR_SUPPORTED_EXTENSIONS]
                
                for file_path in file_paths:
                    try:
                        file_path = Path(file_path)
                        tensor_data = DatasetLoader.load_tensor(file_path)
                        
                        pad_width = [(0, max_dim - cur_dim) if max_dim > cur_dim else (0, 0) for cur_dim, max_dim in zip(tensor_data.shape, max_shape)]
                        padded_tensor = np.pad(tensor_data, pad_width=pad_width, mode='constant', constant_values=0)
                        
                        yield padded_tensor, label
                    except Exception as e:
                        raise RuntimeError(e) from e

        dataset = tf.data.Dataset.from_generator(generator, output_signature=(tf.TensorSpec(shape=(None, None, None), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.string)))
        return dataset
        

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
        if file_path.suffix not in TENSOR_SUPPORTED_EXTENSIONS:
            raise ValueError(f"Extension {file_path.suffix} not suppported.")
        
        try:
            with np.load(file_path) as tensor:
                if file_path.suffix == ".npz":
                    for _, item in tensor.items():
                        tensor = item.squeeze()
                else:  # Assuming it's .npy
                    tensor = tensor
            return tensor
        except Exception as e:
            print(f"Error loading {file_path.stem} file: {str(e)}.", "\nFile path: ", file_path)
            raise RuntimeError(f"Error loading {file_path.stem} file: {str(e)}.") from e