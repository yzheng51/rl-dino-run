import torch
import numpy as np


class LinearAnneal:
    def __init__(self, start_val, end_val, steps):
        self.p = start_val
        self.end_val = end_val
        self.decay_rate = (start_val - end_val) / steps

    def anneal(self):
        if self.p > self.end_val:
            self.p -= self.decay_rate
        return self.p


class StateProcessor:
    def to_array(self, state):
        state = np.array(state).transpose((2, 0, 1))
        state = np.ascontiguousarray(state, dtype=np.float32) / 255
        return state

    def to_tensor(self, state):
        state = self.to_array(state)
        state = torch.from_numpy(state)
        return state.unsqueeze(0)
