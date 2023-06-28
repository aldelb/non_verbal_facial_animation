import numpy as np
import torch 

class NoiseGenerator:

    def __init__(self, device):
        super().__init__()
        self.device = device

    def getNoise(self, dim, batch_size):
        begin = np.random.normal(0, 1, (batch_size, 1))
        end = np.random.normal(0, 1, (batch_size, 1))
        final = begin.copy()
        current = begin.copy()
        step = (end - begin)/(dim-1)
        for _ in range(dim-1):
            current = current + step
            final = np.concatenate((final, current), axis=1)
        return torch.FloatTensor(final).to(self.device)  





