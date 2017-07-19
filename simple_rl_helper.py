from transition import merge_transitions
import numpy as np
import tensorflow as tf

# CALLED BY SIMPLE_RL
def convert_transitions_to_map(transitions, model):
    states, actions, rewards, next_states, is_terminal_next_states = merge_transitions(transitions)
    return {
        model.states: states,
        model.actions: actions,
        model.rewards: rewards,
        model.next_states: next_states,
        model.is_terminal_next_states: is_terminal_next_states
    }

def zero_maxQ_in_terminal_states(maxQ, is_terminals):
    # zero out the elements of the maxQ vector where it is terminal state
    def zero_terminal(reward, is_terminal):
        if is_terminal:
            return 0
        return reward
    return np.vectorize(zero_terminal)(maxQ, is_terminals)

def updateTargetGraph(tfVars,tau=0.9):
    # for the first half of the trainable variables, assign its value to its equivalent on the other half
    # tau is how much we take on from the first half. 1-tau is how much we hold onto
    total_vars = len(tfVars)
    half_vars = total_vars // 2
    op_holder = []
    for idx, var in enumerate(tfVars[0:half_vars]):
        op_holder.append(tfVars[idx+half_vars].assign((var.value()*tau) + ((1-tau)*tfVars[idx+half_vars].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

def restore_model(sess):
    name = 'tmp/my-model-0.meta'
    new_saver = tf.train.import_meta_graph(name)
    new_saver.restore(sess, tf.train.latest_checkpoint('/tmp'))


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


def apply_cnn_layer(x, shape, constant=0.1, stride=3, stddev=0.1):
    W = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    b = tf.Variable(tf.constant(constant, shape=(shape[3],)))
    conv = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')
    ret = tf.nn.bias_add(conv, b)
    return ret


def apply_fc_layer(x, output_size, constant=0.1, stddev=0.1):
    # first, flatten x to be n by d
    shape = x.get_shape().as_list()
    if len(shape) != 1:
        a, b, c, d = shape
        x = tf.reshape(x, [-1, b * c * d])
    input_size = x.get_shape().as_list()[1]
    # multiply out
    W = tf.Variable(tf.truncated_normal([input_size, output_size], stddev))
    b = tf.Variable(tf.constant(constant, shape=(output_size,)))
    multiplied = tf.matmul(x, W)
    added_bias = tf.nn.bias_add(multiplied, b)
    return added_bias


