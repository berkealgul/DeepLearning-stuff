import torch.nn as nn
from params import TrainConfig as tg


class Brain(nn.Module):
    def __init__(self):
        super(Brain, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(tg.inputs, tg.hidden1),
            nn.ReLU(),
            nn.Linear(tg.hidden1, tg.hidden2),
            nn.ReLU(),
            nn.Linear(tg.hidden2, tg.outputs),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)
