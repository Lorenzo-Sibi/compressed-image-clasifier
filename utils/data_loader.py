import tensorflow as tf
import json
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

IMAGE_SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png')
TENSOR_SUPPORTED_EXTENSIONS = ('.npz')

SEED = 2

class DatasetWrapper():
    def __init__(self, directory: Path):
        self.directory = directory
        self.dataset = tf.data.Dataset.load(str(self.directory))
        self.metadata_path = directory / 'dataset_metadata.json'
        self.labels: list[str] = []
        self.max_shape: tuple[int, int, int] = (0, 0, 0)
        self.label_map: dict[str, int] = {}

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"{self.metadata_path} does not exist.")
        
        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Assegna i valori agli attributi dell'oggetto
        self.labels = metadata.get('labels', [])
        self.num_classes = len(self.labels)
        self.max_shape = tuple(metadata.get('max_shape', (0, 0, 0)))
        self.label_map = metadata.get('label_map', {})
        
class DatasetLoader():
    def __init__(self, main_folder:Path, label_map = None):
        self.main_folder = main_folder
        self.labels = [folder.name for folder in self.main_folder.iterdir() if folder.is_dir()]
        self.label_map = {label: i for i, label in enumerate(self.labels)}
        if label_map:
            if len(label_map) != len(self.labels) or all(isinstance(key, int) for key in label_map.keys()):
                raise ValueError(f"Wrong label map {label_map}")
            self.label_map = label_map
        self.max_shape = self._calculate_max_shape()
        
    def _calculate_max_shape(self):
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
                        padded_tensor = padded_tensor.astype(np.float16)
                        
                        num_label = self.label_map[label]
                        yield padded_tensor, num_label
                            
                    except Exception as e:
                        raise RuntimeError(e) from e
            
        dataset = tf.data.Dataset.from_generator(generator, output_signature=(tf.TensorSpec(shape=self.max_shape, dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.int8)))
        return dataset
    
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

    @staticmethod
    def load(input_path:Path)->DatasetWrapper:
        if not input_path.exists():
            raise ValueError(f"{str(input_path)} doesn't exists.")
        ds_wrapper = DatasetWrapper(input_path)
        return ds_wrapper
    
    @staticmethod
    def save(ds_wrapper, output_path:Path)->None:
        if not output_path.exists():
            raise ValueError(f"{str(output_path)} doesn't exists.")
        
        ds = ds_wrapper.dataset
        ds.save(str(output_path))
        
        metadata = {
            attr: getattr(ds_wrapper, attr)
            for attr in dir(ds_wrapper)
            if not attr.startswith('_')
            and not callable(getattr(ds_wrapper, attr))
            and attr != "directory"
            and attr != "dataset"
            and attr != "metadata_path"
        }
        
        metadata_file_path = output_path / "dataset_metadata.json"

        with open(metadata_file_path, "w") as json_file:
            json.dump(metadata, json_file, indent=4)
        
        
    @staticmethod
    def create_dataset(input_dir:Path, output_dir:Path, label_map=None):
        if label_map:
            label_map = json.loads(label_map)
        dataset_loader = DatasetLoader(input_dir, label_map=label_map)
        ds = dataset_loader.load_dataset()

        ds.save(str(output_dir))

        # Construct the metadata dictionary from the dataset object's attributes
        metadata = {
            attr: getattr(dataset_loader, attr)
            for attr in dir(dataset_loader)
            if not attr.startswith('_')
            and not callable(getattr(dataset_loader, attr))
            and attr != "main_folder"
        }
        
        metadata_file_path = output_dir / "dataset_metadata.json"

        with open(metadata_file_path, "w") as json_file:
            json.dump(metadata, json_file, indent=4)

    @staticmethod
    def split_dataset(split_parameter:float, input_dir:Path, output_dir:Path, shuffle=True, seed=SEED):
        assert split_parameter > 0.0 and split_parameter < 1.0,  "Train Size parameter should be >0.0 and <1.0"
        
        ds_wrapper = DatasetLoader.load(input_dir)
        dataset = ds_wrapper.dataset
        
        dataset_size = int(dataset.reduce(0, lambda x, _: x + 1).numpy())
        training_size = int(dataset_size * split_parameter)
        print(f"Training set size: {training_size}. Test set size: {dataset_size - training_size}")
        
        # Shuffle part
        # reshuffle_each_iteration=False to prevent the process running out of memory during shuffle for each epoch 
        if shuffle:
            dataset.shuffle(dataset_size, seed=seed, reshuffle_each_iteration=True)
        
        training_set = dataset.take(training_size)
        test_set = dataset.skip(training_size)
        
        print(training_set, test_set)
        
        ds_wrapper.dataset = training_set
        DatasetLoader.save(ds_wrapper, output_dir / f"{output_dir.stem}_training")
        
        ds_wrapper.dataset = test_set
        DatasetLoader.save(ds_wrapper, output_dir / f"{output_dir.stem}_test")