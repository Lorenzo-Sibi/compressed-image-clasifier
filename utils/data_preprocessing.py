import tensorflow as tf

"""
The following methods are used as modifier for tf.data.Dataset .map() method
"""

def normalize_sample(feature, label):
    """
    Normalizes the input feature tensor using mean and standard deviation.

    Args:
        feature (tf.Tensor): Input feature tensor to be normalized.
        label: Label associated with the feature tensor.

    Returns:
        Tuple[tf.Tensor, Any]: Normalized feature tensor and the original label.
    """
    mean = tf.math.reduce_mean(feature)
    std = tf.math.reduce_std(feature)
    feature = (feature - mean) / std
    return feature, label
    
def flatten_feature(feature, label):
    """
    Flattens the input feature tensor into a 1D tensor.

    Args:
    feature (tf.Tensor): Input feature tensor to be flattened.
    label: Label associated with the feature tensor.

    Returns:
    Tuple[tf.Tensor, Any]: Flattened 1D feature tensor and the original label.
    """
    flattened_feature = tf.reshape(feature, [-1])
    return flattened_feature, label


def apply_mirror_padding(image, label, target_size=(75, 75)):
    """
    Apply mirror (reflective) padding to an image if its dimensions are smaller than the target size.
    Latent space data has some spatial structure that can be meaningfully extended, reflection padding (mirroring the edges of the data) might be a better option than zero padding. 

    Parameters:
    - image: The input image tensor.
    - label: The corresponding label of the image.
    - target_size: A tuple indicating the target height and width (target_height, target_width).

    Returns:
    - image_padded: The padded image if padding was applied, otherwise the original image.
    - label: The label, unchanged.
    """
    image_shape = tf.shape(image)
    height, width = image_shape[0], image_shape[1]

    # Calculate the padding sizes for height and width if they are less than the target size
    padding_height = tf.maximum(target_size[0] - height, 0)
    padding_width = tf.maximum(target_size[1] - width, 0)

    # Apply reflective padding if needed
    if padding_height > 0 or padding_width > 0:
        image_padded = tf.pad(
            image,
            paddings=[[padding_height // 2, padding_height - padding_height // 2],
                      [padding_width // 2, padding_width - padding_width // 2],
                      [0, 0]],
            mode='REFLECT')
    else:
        image_padded = image

    return image_padded, label
