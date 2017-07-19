import tensorflow as tf

# CALLED BY MODEL
def preprocess(images):
    # We are going to do the preprocessing after storing the unprocessed states in the Transition object
    # TODO Test preprocess
    processed_images = tf.map_fn(preprocess_helper, images)
    return processed_images

def preprocess_helper(image):
    height, width, num_channels = image.get_shape().as_list()
    image = tf.random_crop(image, [height - 4, width - 4, 3])
    image = apply_mean_subtraction(image)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=63/256)
    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    image = tf.image.per_image_standardization(image)
    image = tf.pad(image, [[2,2],[2,2],[0,0]], "SYMMETRIC")
    return image

def apply_mean_subtraction(image):
    return image
    # TODO Write mean subtraction
    image = tf.transpose(image, perm=[2, 0, 1])
    mean = tf.constant([122.0, 116.0, 104.0])
    image = tf.subtract(image, mean)
    image = tf.transpose(image, perm=[1, 2, 0])
    return image

