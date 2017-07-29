import gym
import tensorflow as tf

from helper.simple_rl_helper import restore_model
from model import Model

NUM_ITER = 2000
# ENVIRONMENT
game_name = "Breakout-v0"
env = gym.make(game_name)
curr_state = env.reset()
action_space = env.action_space
observation_space = env.observation_space

model = Model(action_space, observation_space)

sess = tf.Session()
saver = tf.train.Saver()

restore_model(sess)


for i in range(NUM_ITER):
    env.render()
    action = sess.run(model.predictions, feed_dict={model.states: [curr_state]})[0]
    Q_list = sess.run(model.Q_list, feed_dict={model.states: [curr_state]})[0]
    print(Q_list)
    print(action)

    next_state, reward, done, info = env.step(action)

    curr_state = next_state
    if done:
        curr_state = env.reset()


