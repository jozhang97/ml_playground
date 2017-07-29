import tensorflow as tf

def add_weight_loss(W):
    weight_loss = tf.nn.l2_loss(W, name="weight_loss")
    tf.add_to_collection('weight_losses', weight_loss)

def apply_cnn_layer(x, shape, constant=0.1, stride=3, stddev=0.1):
    W = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    b = tf.Variable(tf.constant(constant, shape=(shape[3],)))
    conv = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
    ret = tf.nn.bias_add(conv, b)
    add_weight_loss(W)
    return ret


def apply_maxpool_layer(x, size=3, stride=2):
    return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding="SAME")


def apply_fc_layer(x, output_size, constant=0.1, stddev=0.1):
    # first, flatten x to be n by d
    shape = x.get_shape().as_list()
    if len(shape) != 2:
        a, b, c, d = shape
        x = tf.reshape(x, [-1, b * c * d])
    input_size = x.get_shape().as_list()[1]
    # multiply out
    W = tf.Variable(tf.truncated_normal([input_size, output_size], stddev))
    b = tf.Variable(tf.constant(constant, shape=(output_size,)))
    multiplied = tf.matmul(x, W)
    added_bias = tf.nn.bias_add(multiplied, b)
    # apply RELU after
    relu = tf.nn.relu(added_bias)
    add_weight_loss(W)
    return relu
