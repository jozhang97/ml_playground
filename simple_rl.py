import gym
import tensorflow as tf
from model import Model
from simple_rl_helper import preprocess, convert_transitions_to_map, zero_maxQ_in_terminal_states, updateTarget, updateTargetGraph
from replay import Replay
from transition import Transition
import random

# HYPERPARAMETERS
RANDOM_ACTION_PROBABILITY = 0.3  # aka epsilon
DISCOUNT_FACTOR = 0.9  # gamma
REPLAY_MEMORY_SIZE = 100
BATCH_SIZE = 64
NUM_ITER = 2000
INITIAL_LEARNING_RATE = 0.9
TARGET_NETWORK_UPDATE_ITER = 10

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

train_writer = tf.summary.FileWriter('tensorboard_logs/train', sess.graph)
test_writer = tf.summary.FileWriter('tensorboard_logs/test')

trainables = tf.trainable_variables()
target_sync_ops = updateTargetGraph(trainables)

for i in range(NUM_ITER):
    env.render()
    if i % TARGET_NETWORK_UPDATE_ITER == 0:
        print("Updating target network")
        updateTarget(target_sync_ops, sess)

    action = action_space.sample()
    if random.random() > RANDOM_ACTION_PROBABILITY:
        action = sess.run(model.predictions, feed_dict={model.states: [curr_state]})[0]
        Q_list = sess.run(model.Q_list, feed_dict={model.states: [curr_state]})[0]
        print(Q_list)
        print(action)
        train_writer.flush()

    next_pstate, reward, done, info = env.step(action)
    next_state = preprocess(next_pstate)

    replay.add_transition(Transition(curr_state, action, reward, next_state, done))

    batch_transitions = replay.pick_batch_transitions()
    train_step_map = convert_transitions_to_map(batch_transitions, model)

    maxQ = sess.run(target_model.maxQ, feed_dict={target_model.states: train_step_map[model.states]})
    maxQ_zeroed = zero_maxQ_in_terminal_states(maxQ, train_step_map[model.is_terminal_next_states])
    targetQ = DISCOUNT_FACTOR * train_step_map[model.rewards] + maxQ_zeroed
    train_step_map[model.targetQ] = targetQ
    merged_summaries, _ = sess.run([model.merged_summaries, model.train_step], feed_dict=train_step_map)
    train_writer.add_summary(merged_summaries, i)

    curr_state = next_state
    if done:
        curr_state = preprocess(env.reset())
