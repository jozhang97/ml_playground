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


# TODO UNDERSTAND BELOW
def updateTargetGraph(tfVars,tau=0.001):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

# TODO UNDERSTAND BELOW
def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)
