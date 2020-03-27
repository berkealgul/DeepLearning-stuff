import torch.nn as nn
import torch
from params import TrainConfig as tg
import numpy as np


class Brain():
    def __init__(self):
        self.loss_actor = None
        self.loss_critic = None
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=tg.lr)
        self.replayBuffer = ReplayBuffer()

    def predict_action(self, x):
        return self.actor(x)

    # TODO: BİTİR ŞUNU
    def train(self):
        for i in range(tg.steps_each_ep):
            s, a, r, sn = self.replayBuffer.get_sample(i)
            val = torch.cat(s, a, dim=1)
            Qval = self.critic(val)

            val_t = torch.cat(si, self.actor_target(sn), dim=1)
            Qval_t = self.critic_target(val_t)


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


class ActorNetwork(nn.Module):
    def __init__(self, lr=0.01, in_dims=7, fc1_dims=300, fc2_dims=400, out_dims=3):
        super(ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(in_dims,fc1_dims)
        self.bn1 = nn.BatchNorm1d(fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.bn2 = nn.BatchNorm1d(fc2_dims)
        self.out  = nn.Linear(fc2_dims, out_dims)

        self.device = torch.device('cpu')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def feedforward(self, state):
        action = F.relu(self.bn1(self.fc1(state)))
        action = F.relu(self.bn2(self.fc2(action)))
        action = F.relu(self.out(action))
        return action


class CriticNetwork(nn.Module):
    def __init__(self, lr=0.01, in_dims=7, fc1_dims=300, fc2_dims=400, out_dims=3):
        super(ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(in_dims,fc1_dims)
        self.bn1 = nn.BatchNorm1d(fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.bn2 = nn.BatchNorm1d(fc2_dims)
        self.out  = nn.Linear(fc2_dims, out_dims)
        self.bn3 = nn.BatchNorm1d(out_dims)

        self.device = torch.device('cpu')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def feedforward(self, state, action):
        pass
