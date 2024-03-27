import tensorflow as tf

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
