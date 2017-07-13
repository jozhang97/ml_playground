import numpy as np
import tensorflow as tf
import random as random
import time
# from transition import Transition
from transition import merge_transitions

class Model:
    # EXTERNALLY,
    # evaluate by running sess.run(model.q, feed_dict={states: states, actions: actions})
    # train by model.train_step
    # select action by by model.select_action_step
    def __init__(self, sess, action_space, observation_space, initial_state, DISCOUNT_FACTOR=0.9, INITIAL_LEARNING_RATE = 0.9):
        input_size = 28190
        output_size = 1
        self.W = tf.Variable(tf.truncated_normal([input_size, output_size], 0.1))
        self.W_target = self.W
        self.DISCOUNT_FACTOR = DISCOUNT_FACTOR
        self.LEARNING_RATE = INITIAL_LEARNING_RATE
        self.sess = sess
        self.action_space = action_space
        self.observation_space = observation_space
        self.state_shape = initial_state.shape
        self.q = self.evaluate()
        self.train_step = self.update()
        # self.select_action_step = self.select_action()

    def evaluate_single(self, state, action):
        # TODO Too slow
        # TODO Need to one hot action and incorporate
        # evaluates the value of the state action pair Q(s,a; theta)
        # q = tf.matmul(self.W, state.flatten().astype(np.float32).reshape([120000, 1]))
        state = tf.nn.max_pool(state, ksize=[3,3,1,1], strides=[2,2,1,1], padding='VALID')
        x,y,z = state.shape
        q = np.dot(self.W, state.flatten().astype(np.float32).reshape([x*y*z, 1]))
        return q[0][0]

    def evaluate(self):
        # Lessons learned here: get_shape() must be followed by .as_list() to look at the dimensions
        # Must apply tf.one_hot on a vector, thus its shape must be (None,)
        # Using concat instead of stack to combine half-feature vectors
        states = tf.placeholder(dtype=tf.float32, shape=(None, self.state_shape[0], self.state_shape[1], self.state_shape[2]))
        actions = tf.placeholder(dtype=tf.int32, shape=(None,))
        states = tf.nn.max_pool(states, ksize=[1,7,7,1], strides=[1,2,2,1], padding='VALID')
        n, x, y, c = states.get_shape().as_list()
        states_reshaped = tf.reshape(states, [-1, x*y*c])
        actions_one_hotted = tf.one_hot(actions, depth=self.action_space.n, dtype=tf.float32, axis=-1)
        feature_vector = tf.concat([states_reshaped, actions_one_hotted], 1)
        q = tf.matmul(feature_vector, self.W)
        return tf.nn.softmax(q)

    def update(self, batch_transitions):
        # batch gradient descent
        # batch_transitions is a set of Transition objects
        for t in batch_transitions:
            self.update_single(t)


    def update_single(self, t):
        # runs gradient descent
        curr_state = tf.placeholder(tf.float32, self.observation_space.shape)
        action = tf.placeholder(tf.int32)
        reward = tf.placeholder(tf.float32)
        next_state = tf.placeholder(tf.float32, self.observation_space.shape)
        is_terminal_next_state = tf.placeholder(tf.bool)

        loss = self.calculate_loss(curr_state, action, reward, next_state, is_terminal_next_state)
        train_step = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE).minimize(loss)
        self.sess.run(train_step, feed_dict={curr_state: t.get_curr_state(), action: t.get_action(),
                                             reward: t.get_reward(), next_state: t.get_next_state(),
                                             is_terminal_next_state: t.get_is_terminal_next_state()})


    def calculate_loss(self, curr_state, action, reward, next_state, is_terminal_next_state):
        y_i = tf.cond(is_terminal_next_state, lambda: reward,
                      lambda: reward + self.DISCOUNT_FACTOR * self.evaluate(next_state, self.select_action(next_state, 0.0)))
        y = self.evaluate(curr_state, action)
        return tf.nn.l2_loss(y - y_i)


    def evaluate_target(self, state, action):
        # evaluates the value of the state action pair with the target network
        temp = self.W
        self.W = self.W_target
        ret = self.evaluate_target(state, action)
        self.W = temp
        return ret


    def select_action(self, state, RANDOM_ACTION_PROBABILITY):
        # picks an action to take
        if random.random() < RANDOM_ACTION_PROBABILITY:
            return self.action_space.sample()
        action_optimal, q_max = 0, 0.0
        for action_index in range(self.action_space.n): # not sure about this
            q = self.evaluate(state, self.action_space.from_jsonable(action_index))
            if q > q_max:
                action_optimal, q_max = action_index, q
            # action_optimal, q_max = tf.cond(q > q_max, lambda: (action_index, q), lambda: (action_optimal, q_max))
        return action_optimal


    def sync_target(self):
        # update target to be equal to current network
        self.W_target = self.W


    def update_learning_rate(self, multiplicative_factor):
        # makes learning rate smaller
        self.LEARNING_RATE *= min(multiplicative_factor, 1)
