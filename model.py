# Lessons learned here: get_shape() must be followed by .as_list() to look at the dimensions
# Must apply tf.one_hot on a vector, thus its shape must be (None,)
# Using concat instead of stack to combine half-feature vectors
# Q(s,a; theta) means we have run the state through CNN and then have num_actions outputs that are the probabilities
import tensorflow as tf

class Model:
    def __init__(self, action_space, observation_space, DISCOUNT_FACTOR=0.9, INITIAL_LEARNING_RATE = 0.9):
        self.num_actions = action_space.n
        self.input_size = 28182
        self.output_size = self.num_actions
        self.W = tf.Variable(tf.truncated_normal([self.input_size, self.output_size], 0.1))
        self.DISCOUNT_FACTOR = DISCOUNT_FACTOR
        self.LEARNING_RATE = INITIAL_LEARNING_RATE

        self.state_shape = observation_space.shape
        self.states = tf.placeholder(dtype=tf.float32, shape=(None, self.state_shape[0], self.state_shape[1], self.state_shape[2]))

        self.max_pooled_states = tf.nn.max_pool(self.states, ksize=[1,7,7,1], strides=[1,2,2,1], padding='VALID')
        n, x, y, c = self.max_pooled_states.get_shape().as_list()
        self.states_reshaped = tf.reshape(self.max_pooled_states, [-1, x*y*c])
        self.Q_list = tf.nn.softmax(tf.matmul(self.states_reshaped, self.W))
        self.maxQ = tf.reduce_max(self.Q_list, axis=1)  # how did they pick best action as opposed to max Q ... idk if right axis
        self.predictions = tf.cast(tf.argmax(self.Q_list), tf.int32)

        self.actions = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.rewards = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.next_states = tf.placeholder(dtype=tf.float32, shape=(None, self.state_shape[0], self.state_shape[1], self.state_shape[2]))
        self.is_terminal_next_states = tf.placeholder(dtype=tf.bool, shape=(None,))

        self.actions_one_hotted = tf.one_hot(self.actions, depth=self.num_actions, dtype=tf.float32, axis=-1)
        self.Q = tf.reduce_sum(tf.multiply(self.Q_list, self.num_actions), axis=1)
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)

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

    def update_learning_rate(self, multiplicative_factor):
        # makes learning rate smaller
        self.LEARNING_RATE *= min(multiplicative_factor, 1)
