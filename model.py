# Lessons learned here: get_shape() must be followed by .as_list() to look at the dimensions
# Must apply tf.one_hot on a vector, thus its shape must be (None,)
# Using concat instead of stack to combine half-feature vectors
## states = tf.placeholder(dtype=tf.float32, shape=(None, self.state_shape[0], self.state_shape[1], self.state_shape[2]))
## actions = tf.placeholder(dtype=tf.int32, shape=(None,))
import numpy as np
import tensorflow as tf
import random as random
import time
# from transition import Transition

class Model:
    def __init__(self, initial_state, DISCOUNT_FACTOR=0.9, INITIAL_LEARNING_RATE = 0.9):
        input_size = 28190
        output_size = 1
        self.W = tf.Variable(tf.truncated_normal([input_size, output_size], 0.1))
        self.W_target = self.W
        self.DISCOUNT_FACTOR = DISCOUNT_FACTOR
        self.LEARNING_RATE = INITIAL_LEARNING_RATE


        self.state_shape = initial_state.shape
        self.states = tf.placeholder(dtype=tf.float32, shape=(None, self.state_shape[0], self.state_shape[1], self.state_shape[2]))
        self.actions = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.rewards = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.next_states = tf.placeholder(dtype=tf.float32, shape=(None, self.state_shape[0], self.state_shape[1], self.state_shape[2]))
        self.is_terminal_next_states = tf.placeholder(dtype=tf.bool, shape=(None,))

        self.max_pooled_states = tf.nn.max_pool(self.states, ksize=[1,7,7,1], strides=[1,2,2,1], padding='VALID')
        n, x, y, c = self.max_pooled_states.get_shape().as_list()
        self.states_reshaped = tf.reshape(self.max_pooled_states, [-1, x*y*c])
        self.actions_one_hotted = tf.one_hot(self.actions, depth=self.action_space.n, dtype=tf.float32, axis=-1)
        self.feature_vector = tf.concat([self.states_reshaped, self.actions_one_hotted], 1)
        self.Q = tf.nn.softmax(tf.matmul(self.feature_vector, self.W))
        self.predictions = tf.cast(tf.argmax(self.q), tf.int32)

        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)

        # if is_terminal_next_state:
        #     self.targetQ = reward
        # else:
        #     self.targetQ = reward + self.DISCOUNT_FACTOR * self.evaluate(next_state, self.select_action(next_state, 0.0)))

        # Batch gradient descent update
        self.loss = tf.reduce_mean(tf.nn.l2_loss(self.targetQ - self.Q))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE)
        self.train_step = self.optimizer.minimize(self.loss)


    def evaluate_target(self, state, action):
        # evaluates the value of the state action pair with the target network
        temp = self.W
        self.W = self.W_target
        ret = self.evaluate_target(state, action)
        self.W = temp
        return ret



    def sync_target(self):
        # update target to be equal to current network
        self.W_target = self.W


    def update_learning_rate(self, multiplicative_factor):
        # makes learning rate smaller
        self.LEARNING_RATE *= min(multiplicative_factor, 1)
