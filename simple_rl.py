import gym
from model import Model
from simple_rl_helper import preprocess
from replay import Replay
from transition import Transition

# HYPERPARAMETERS
RANDOM_ACTION_PROBABILITY = 0.001  # aka epsilon
DISCOUNT_FACTOR = 0.9  # gamma
REPLAY_MEMORY_SIZE = 100
BATCH_SIZE = 64
NUM_ITER = 2000
INITIAL_LEARNING_RATE = 0.9
TARGET_NETWORK_UPDATE_ITER = 50

# ENVIRONMENT SETTINGS
game_name = "Gopher-v0"
env = gym.make(game_name)
curr_state = preprocess(env.reset())

# initalize replay memory and action-value function
replay = Replay(REPLAY_MEMORY_SIZE, BATCH_SIZE)
model = Model(DISCOUNT_FACTOR, INITIAL_LEARNING_RATE)
action_space = env.action_space

for n in range(NUM_ITER):
    env.render()

    action = model.select_action(curr_state, action_space, RANDOM_ACTION_PROBABILITY)

    next_pstate, reward, done, info = env.step(action)
    next_state = preprocess(next_pstate)

    replay.add_transition(Transition(curr_state, action, reward, next_state))
    batch_transitions = replay.pick_random_transition()

    model.update(batch_transitions)

    if n != 0 and n % TARGET_NETWORK_UPDATE_ITER == 0:
        model.sync_target()

    if done:
        curr_state = preprocess(env.reset())

    curr_state = next_state
