import gym
import tensorflow as tf
from model import Model
from simple_rl_helper import preprocess, convert_transitions_to_map, zero_maxQ_in_terminal_states, updateTarget, updateTargetGraph
from replay import Replay
from transition import Transition
import random

# HYPERPARAMETERS
RANDOM_ACTION_PROBABILITY = 0.1  # aka epsilon
DISCOUNT_FACTOR = 0.9  # gamma
REPLAY_MEMORY_SIZE = 100
BATCH_SIZE = 64
NUM_ITER = 2000
INITIAL_LEARNING_RATE = 0.9
TARGET_NETWORK_UPDATE_ITER = 50

# ENVIRONMENT
game_name = "Breakout-v0"
env = gym.make(game_name)
curr_state = preprocess(env.reset())
action_space = env.action_space
observation_space = env.observation_space

# Initalize replay memory and action-value function (model)
model = Model(action_space, observation_space)
target_model = Model(action_space, observation_space)
replay = Replay(REPLAY_MEMORY_SIZE, BATCH_SIZE)

# Set up tf
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver() # TODO Implement this
# TODO IMPLEMENT TENSORBOARD
trainables = tf.trainable_variables()
target_sync_ops = updateTargetGraph(trainables)

for n in range(NUM_ITER):
    if n % TARGET_NETWORK_UPDATE_ITER == 0:
        updateTarget(target_sync_ops, sess)
        env.render()

    action = action_space.sample()
    if random.random() > RANDOM_ACTION_PROBABILITY:
        action = sess.run(model.predictions, feed_dict={model.states: [curr_state]})[0]

    next_pstate, reward, done, info = env.step(action)
    next_state = preprocess(next_pstate)

    replay.add_transition(Transition(curr_state, action, reward, next_state, done))

    batch_transitions = replay.pick_batch_transitions()
    train_step_map = convert_transitions_to_map(batch_transitions, model)

    maxQ = sess.run(target_model.maxQ, feed_dict={target_model.states: train_step_map[model.states]})
    maxQ_zeroed = zero_maxQ_in_terminal_states(maxQ, train_step_map[model.is_terminal_next_states])
    targetQ = DISCOUNT_FACTOR * train_step_map[model.rewards] + maxQ_zeroed
    train_step_map[model.targetQ] = targetQ
    sess.run(model.train_step, feed_dict=train_step_map)

    curr_state = next_state
    if done:
        curr_state = preprocess(env.reset())
