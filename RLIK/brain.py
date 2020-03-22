import torch.nn as nn
import torch
from params import TrainConfig as tg


class Brain(nn.Module):
    def __init__(self):
        super(Brain, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(tg.inputs, tg.hidden1),
            nn.ReLU(),
            nn.Linear(tg.hidden1, tg.hidden2),
            nn.ReLU(),
            nn.Linear(tg.hidden2, tg.outputs),
            nn.ReLU()
        )
        self.actor_target = nn.Sequential(
            nn.Linear(tg.inputs, tg.hidden1),
            nn.ReLU(),
            nn.Linear(tg.hidden1, tg.hidden2),
            nn.ReLU(),
            nn.Linear(tg.hidden2, tg.outputs),
            nn.ReLU()
        )
        self.critic = nn.Sequential(
            nn.Linear(tg.inputs, tg.hidden1),
            nn.ReLU(),
            nn.Linear(tg.hidden1, tg.hidden2),
            nn.ReLU(),
            nn.Linear(tg.hidden2, tg.outputs),
            nn.ReLU()
        )
        self.critic_target = nn.Sequential(
            nn.Linear(tg.inputs, tg.hidden1),
            nn.ReLU(),
            nn.Linear(tg.hidden1, tg.hidden2),
            nn.ReLU(),
            nn.Linear(tg.hidden2, tg.outputs),
            nn.ReLU()
        )

        self.loss_actor = None
        self.loss_critic = None
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=tg.lr)
        self.replayBuffer = ReplayBuffer()

    def predict_action(self, x):
        return self.actor(x)

    # TODO: BİTİR ŞUNU
    def train(self):
        """
        for i in range(tg.steps_each_ep):
            s, a, r, sn = self.replayBuffer.get_sample(i)
            val = torch.cat(s, a, dim=1)
            Qval = self.critic(val)

            val_t = torch.cat(si, self.actor_target(sn), dim=1)
            Qval_t = self.critic_target(val_t)
        """
        

class ReplayBuffer():
    def __init__(self):
        self.states    = []
        self.next_states = []
        self.rewards   = []
        self.actions   = []

    def add_sample(self, s, a, r, sn):
        self.states.append(s)
        self.next_states.append(sn)
        self.rewards.append(r)
        self.actions.append(a)

    def get_sample(self, ids):
        s = self.states[ids]
        sn = self.next_states[ids]
        a = self.actions[ids]
        r = self.rewards[ids]
        return s, a, r, sn
