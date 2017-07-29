import os
import shutil
import numpy as np
import tensorflow as tf

import sys
sys.path.append("/Users/jeff/Documents/Research/playground")
from replay.transition import merge_transitions

def run_all_actions(env):
    for action in range(env.action_space.n):
        env.step(action)

def convert_transitions_to_map(transitions, model):
    states, actions, rewards, next_states, is_terminal_next_states = merge_transitions(transitions)
    return {
        model.states: states,
        model.actions: actions,
        model.rewards: rewards,
        model.next_states: next_states,
        model.is_terminal_next_states: is_terminal_next_states
    }


def zero_maxQ_in_terminal_states(maxQ, is_terminals):
    # zero out the elements of the maxQ vector where it is terminal state
    def zero_terminal(reward, is_terminal):
        if is_terminal:
            return 0
        return reward

    return np.vectorize(zero_terminal)(maxQ, is_terminals)


def updateTargetGraph(tfVars, tau=0.9):
    # for the first half of the trainable variables, assign its value to its equivalent on the other half
    # tau is how much we take on from the first half. 1-tau is how much we hold onto
    total_vars = len(tfVars)
    half_vars = total_vars // 2
    op_holder = []
    for idx, var in enumerate(tfVars[0:half_vars]):
        op_holder.append(
            tfVars[idx + half_vars].assign((var.value() * tau) + ((1 - tau) * tfVars[idx + half_vars].value())))
    return op_holder


def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)


def restore_model(sess):
    name = 'tmp/my-model-0.meta'
    saver = tf.train.import_meta_graph(name)
    saver.restore(sess, tf.train.latest_checkpoint('tmp'))


def remove_previous_logs():
    if os.path.exists("../tensorboard_logs/train"):
        shutil.rmtree("../tensorboard_logs/train")
        print("Removing old tensorboard_logs")
    if os.path.exists("tensorboard_logs/train"):
        shutil.rmtree("tensorboard_logs/train")
        print("Removing old tensorboard_logs")
    if os.path.exists("../tensorboard_logs/test"):
        shutil.rmtree("../tensorboard_logs/test")
        print("Removing old tensorboard_logs")
    if os.path.exists("tensorboard_logs/test"):
        shutil.rmtree("tensorboard_logs/test")
        print("Removing old tensorboard_logs")
