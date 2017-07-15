from transition import merge_transitions

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