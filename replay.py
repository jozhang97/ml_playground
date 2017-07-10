from transition import Transition
import random as random

class Replay():
    def __init__(self, REPLAY_MEMORY_SIZE = 100, BATCH_SIZE = 64):
        self.replay = set() # probably need a hashset or some sort of smarter set
        self.REPLAY_MEMORY_SIZE = REPLAY_MEMORY_SIZE
        self.BATCH_SIZE = BATCH_SIZE

    def add_transition(self, transition):
        if len(self.replay) >= self.REPLAY_MEMORY_SIZE:
            self.replay.pop()
        self.replay.add(transition)

    def pick_random_transition(self):
        return random.sample(self.replay, min(self.BATCH_SIZE, len(self.replay)))

    def __str__(self):
        return str([str(r) for r in self.replay])
