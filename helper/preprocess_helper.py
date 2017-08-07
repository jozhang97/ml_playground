import tensorflow as tf
import numpy as np

mean = np.array([[[122.0, 116.0, 104.0]]])

# CALLED BY MODEL
def preprocess(images):
    # We are going to do the preprocessing after storing the unprocessed states in the Transition object
    processed_images = tf.map_fn(preprocess_helper, images)
    return processed_images


def preprocess_helper(image):
    height, width, num_channels = image.get_shape().as_list()
    image = tf.random_crop(image, [height - 4, width - 4, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=63/256)
    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    image = tf.image.per_image_standardization(image)
    image = tf.pad(image, [[2,2],[2,2],[0,0]], "SYMMETRIC")
    image = apply_mean_subtraction(image)
    return image


def apply_mean_subtraction(image):
    return tf.subtract(image, mean)

