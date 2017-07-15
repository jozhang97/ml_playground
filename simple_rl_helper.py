from transition import merge_transitions
import numpy as np

def preprocess(image):
    # TODO: Implement this
    return image

def convert_transitions_to_map(transitions, model):
    states, actions, rewards, next_states, is_terminal_next_states = merge_transitions(transitions)
    return {
        model.states: states,
        model.actions: actions,
        model.rewards: rewards,
        model.next_states: next_states,
        model.is_terminal_next_states: is_terminal_next_states
    }

def sync_target_model(model, target_model):
    target_model.W = model.W
    # TODO SYNC THE CONV TOO


def zero_maxQ_in_terminal_states(maxQ, is_terminals):
    # zero out the elements of the maxQ vector where it is terminal state
    def zero_terminal(reward, is_terminal):
        if is_terminal:
            return 0
        return reward
    return np.vectorize(zero_terminal)(maxQ, is_terminals)
