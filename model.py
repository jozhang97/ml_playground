# Lessons learned here: get_shape() must be followed by .as_list() to look at the dimensions
# Must apply tf.one_hot on a vector, thus its shape must be (None,)
# Using concat instead of stack to combine half-feature vectors
# Q(s,a; theta) means we have run the state through CNN and then have num_actions outputs that are the probabilities
import tensorflow as tf  # look into using keras
from simple_rl_helper import preprocess, apply_cnn_layer, apply_fc_layer


class Model:
    def __init__(self, action_space, observation_space, DISCOUNT_FACTOR=0.9, INITIAL_LEARNING_RATE=0.9):
        self.num_actions = action_space.n
        self.DISCOUNT_FACTOR = DISCOUNT_FACTOR
        self.LEARNING_RATE = INITIAL_LEARNING_RATE
        self.REGULARIZATION_COEFF = 0.9

        self.state_shape = observation_space.shape
        self.states = tf.placeholder(dtype=tf.float32,
                                     shape=(None, self.state_shape[0], self.state_shape[1], self.state_shape[2]))
        self.processed_states = preprocess(self.states)
        self.model_function()
        self.predict_function()
        self.lose_function()
        self.update_function()
        self.summary_function()

    def model_function(self):
        # TODO make the model CNN
        self.layer_conv1 = apply_cnn_layer(self.processed_states, [7,7,3,64])
        self.max_pooled_states = tf.nn.max_pool(self.layer_conv1, ksize=[1, 7, 7, 1], strides=[1, 2, 2, 1],
                                                padding='VALID')
        self.layer_fc1 = apply_fc_layer(self.max_pooled_states, self.num_actions)
        self.Q_list = tf.nn.softmax(self.layer_fc1)

    def predict_function(self):
        self.maxQ = tf.reduce_max(self.Q_list,
                                  axis=1)  # how did they pick best action as opposed to max Q ... idk if right axis
        self.predictions = tf.cast(tf.argmax(self.Q_list, axis=1), tf.int32)

    def lose_function(self):
        self.actions = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.rewards = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.next_states = tf.placeholder(dtype=tf.float32,
                                          shape=(None, self.state_shape[0], self.state_shape[1], self.state_shape[2]))
        self.is_terminal_next_states = tf.placeholder(dtype=tf.bool, shape=(None,))

        self.actions_one_hotted = tf.one_hot(self.actions, depth=self.num_actions, dtype=tf.float32, axis=-1)
        self.Q = tf.reduce_sum(tf.multiply(self.Q_list, self.num_actions), axis=1)
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)

    def update_function(self):
        # Batch gradient descent update
        self.regularization_loss = 0
        self.simple_loss = tf.reduce_mean(tf.nn.l2_loss(self.targetQ - self.Q))
        self.loss = self.regularization_loss + self.simple_loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE)
        self.train_step = self.optimizer.minimize(self.loss)

    def summary_function(self):
        tf.summary.scalar("Regularization_loss", self.regularization_loss)
        tf.summary.scalar("Simple_loss", self.simple_loss)
        tf.summary.scalar("Total_loss", self.loss)
        tf.summary.image("Batch_pictures", self.states)
        tf.summary.image("Batch_pictures_processed", self.processed_states)
        self.merged_summaries = tf.summary.merge_all()

    def update_learning_rate(self, multiplicative_factor):
        # makes learning rate smaller
        # idk if this will work
        self.LEARNING_RATE *= min(multiplicative_factor, 1)
