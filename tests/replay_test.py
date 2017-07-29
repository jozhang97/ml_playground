import sys
sys.path.append('../')
from replay.replay import Replay
from replay.transition import Transition
import numpy as np

def f(arr):
    return np.array(arr)

replay = Replay(2, 2)

t1 = Transition(f([]),1,2,f([]))
t11 = Transition(f([]),1,2,f([]))

t2 = Transition(f([1,2,3]),4,2,f([77,2,1]))
t22 = Transition(f([1,2,3]),4,2,f([77,2,1]))

t3 = Transition(f([1,2,1]),4,2,f([77,5,1]))
t33 = Transition(f([1,2,1]),4,2,f([77,5,1]))

replay.add_transition(t1)
replay.add_transition(t2)
replay.add_transition(t3)
replay.add_transition(t11)
replay.add_transition(t22)
replay.add_transition(t33)

print("Replay: " + str(replay))
print("Transition: " + str(t22))

print("Random transition: " + str(replay.pick_random_transition()))
print("Random transition: " + str(replay.pick_random_transition()))
print("Random transition: " + str(replay.pick_random_transition()))
print("Random transition: " + str(replay.pick_random_transition()))
