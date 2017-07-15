import gym
import tensorflow as tf
from model import Model
from simple_rl_helper import preprocess, convert_transitions_to_map
from replay import Replay
from transition import Transition
import random

# HYPERPARAMETERS
RANDOM_ACTION_PROBABILITY = 0.001  # aka epsilon
DISCOUNT_FACTOR = 0.9  # gamma
REPLAY_MEMORY_SIZE = 100
BATCH_SIZE = 64
NUM_ITER = 2000
INITIAL_LEARNING_RATE = 0.9
TARGET_NETWORK_UPDATE_ITER = 50

# ENVIRONMENT
game_name = "Gopher-v0"
env = gym.make(game_name)
curr_state = preprocess(env.reset())
action_space = env.action_space
observation_space = env.observation_space

# Initalize replay memory and action-value function (model)
model = Model(curr_state)
replay = Replay(REPLAY_MEMORY_SIZE, BATCH_SIZE)

# Set up tf
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for n in range(NUM_ITER):
    env.render()

    action = action_space.sample()
    if random.random() > RANDOM_ACTION_PROBABILITY:
        action = sess.run(model.predictions, feed_dict={})
        # TODO THIS

    next_pstate, reward, done, info = env.step(action)
    next_state = preprocess(next_pstate)

    replay.add_transition(Transition(curr_state, action, reward, next_state, done))
    batch_transitions = replay.pick_batch_transitions()

    # need to at targetQ to the feed_dict as well
    sess.run(model.train_step, feed_dict=convert_transitions_to_map(batch_transitions, model))

    if n != 0 and n % TARGET_NETWORK_UPDATE_ITER == 0:
        model.sync_target()
        # needs update

    if done:
        curr_state = preprocess(env.reset())

    curr_state = next_state
