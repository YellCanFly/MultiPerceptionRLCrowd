import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, *args, **kwargs):
        pass

    def store_memory(self, *args, **kwargs):
        raise NotImplementedError

    def sample(self, batch_size=64):
        raise NotImplementedError


class RLModel:
    def __init__(self, *args, **kwargs):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def remember(self, *args, **kwargs):
        raise NotImplementedError

    def choose_action(self, state):
        raise NotImplementedError

    def load_models(self, load_path=None):
        raise NotImplementedError

    def save_models(self):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError
