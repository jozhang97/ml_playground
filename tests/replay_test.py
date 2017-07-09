import sys
sys.path.append('../')
from replay import Replay
from transition import Transition
import numpy as np

def f(arr):
    return np.array(arr)

replay = Replay()

t1 = Transition(f([]),1,2,f([]))
t11 = Transition(f([]),1,2,f([]))

t2 = Transition(f([1,2,3]),4,2,f([77,2,1]))
t22 = Transition(f([1,2,3]),4,2,f([77,2,1]))

replay.add_transition(t1)
replay.add_transition(t2)
replay.add_transition(t11)
replay.add_transition(t22)

print("Replay: " + str(replay))
print("Transition: " + str(t22))
