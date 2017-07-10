import numpy as np

class Transition:
    def __init__(self, s_t, a_t, r_t, s_t_1):
        self.s_t = s_t
        self.a_t = a_t
        self.r_t = r_t
        self.s_t_1 = s_t_1

    def __hash__(self):
        self.s_t.flags.writeable = False
        self.s_t_1.flags.writeable = False
        hash_tuple = (hash(self.s_t.data), self.a_t, self.r_t, hash(self.s_t_1.data))
        return hash(hash_tuple)

    def __eq__(self, other):
        if not (type(self) is type(other)):
            return False
        if not np.array_equal(self.s_t, other.s_t):
            return False
        if self.a_t != other.a_t:
            return False
        if self.r_t != other.r_t:
            return False
        if not np.array_equal(self.s_t_1, other.s_t_1):
            return False
        return True

    def get_curr_state(self):
        return self.s_t

    def get_action(self):
        return self.a_t

    def get_reward(self):
        return self.r_t

    def get_next_state(self):
        return self.s_t_1

    def __str__(self):
        return "Current State: " + str(self.s_t) + "\nAction: " + str(self.a_t) + "\nReward: " + str(self.r_t) + "\n Next State: " + str(self.s_t_1)
