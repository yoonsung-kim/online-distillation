import torch
import torch.nn as nn


class DeepMutualLearning(nn.Module):
    def __init__(self, peers):
        super(DeepMutualLearning, self).__init__()

        if len(peers) != 2:
            raise ValueError("# of peers must be 2 in deep mutual learning architecture")

        self.count = 2

        self.model_0 = peers[0]
        self.model_1 = peers[1]

    def forward(self, x):
        outputs = [self.model_0(x), self.model_1(x)]

        return torch.stack(outputs, dim=0)
