import numpy as np
import tensorflow as tf
import random as random
# from transition import Transition

class Model:
    def __init__(self, DISCOUNT_FACTOR=0.9, INITIAL_LEARNING_RATE = 0.9):
        input_size = 120000
        output_size = 8
        self.W = tf.Variable(tf.truncated_normal([output_size, input_size], 0.1))
        self.W_target = self.W
        self.DISCOUNT_FACTOR = DISCOUNT_FACTOR
        self.LEARNING_RATE = INITIAL_LEARNING_RATE

    def evaluate(self, state, action):
        # evaluates the value of the state action pair
        return 0
        q = tf.matmul(self.W, state.flatten().astype(np.float32))
        return q

    def evaluate_target(self, state, action):
        # evaluates the value of the state action pair with the target network
        temp = self.W
        self.W = self.W_target
        ret = self.evaluate_target(state, action)
        self.W = temp
        return ret

    def update(self, batch_transitions):
        # batch gradient descent
        # batch_transitions is a set of Transition objects
        for t in batch_transitions:
            self.update_single(t.get_curr_state(), t.get_action(), t.get_reward(), t.get_next_state())

    def update_single(self, curr_state, action, reward, next_state):
        # runs gradient descent
        return 0
        train_step = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE).minimize(loss)
        # apply sgd
        return 0

    def sync_target(self):
        # update target to be equal to current network
        self.W_target = self.W

    def update_learning_rate(self, multiplicative_factor):
        # makes learning rate smaller
        self.LEARNING_RATE *= min(multiplicative_factor, 1)


    def select_action(self, state, action_space, RANDOM_ACTION_PROBABILITY):
        # picks an action to take
        if random.random() < RANDOM_ACTION_PROBABILITY:
            return action_space.sample()
        return action_space.sample()
        q_max, action_optimal = 0, 0
        for action_index in range(action_space.n): # not sure about this
            q = self.evaluate(state, action_space.from_jsonable(action_index))
            if q > q_max:
                action_optimal = action_index
                q_max = q
        return action_optimal
