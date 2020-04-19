import torch.nn as nn
import torch.nn.functional as F
import torch
from params import TrainConfig as tg
import numpy as np
import os


class Brain():
    def __init__(self, gamma=tg.gamma, tau=tg.tau):
        self.actor = ActorNetwork()
        self.critic = CriticNetwork()
        self.target_actor = ActorNetwork()
        self.target_critic = CriticNetwork()
        self.replayBuffer = ReplayBuffer()
        self.gamma = gamma
        self.tau = tau

    def predict_action(self, state):
        self.actor.eval()
        action = self.actor(state)
        return action

    def train(self, steps):
        s, a, r, sn = self.replayBuffer.get_all_buffer()

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        aVal = self.target_actor.forward(sn)
        qvt = self.target_critic.forward(sn, aVal)

        qVal_t = []
        for i in range(len(r)):
            qVal_t.append(r[i] + self.gamma*qvt[i])

        qVal_t = torch.stack(qVal_t)
        qVal = self.critic.forward(s, a)

        self.critic.train()

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(qVal_t, qVal)
        critic_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm(self.critic.parameters(), 1)
        self.critic.optimizer.step()

        self.critic.eval()
        mu = self.actor.forward(s)
        self.actor.train()
        actor_loss = -self.critic.forward(s, mu)
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm(self.actor.parameters(), 1)
        self.actor.optimizer.step()

        print("loss  a " + str(actor_loss.item()) + " c " + str(critic_loss.item()))

        self.update_target_networks()

    def update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        actor_dict = dict(actor_params)
        target_actor_dict = dict(target_actor_params)
        critic_dict = dict(critic_params)
        target_critic_dict = dict(target_critic_params)

        for name in target_actor_dict:
            target_actor_dict[name] = tau*actor_dict[name].clone() + \
                (1-tau)*target_actor_dict[name].clone()

        for name in target_critic_dict:
            target_critic_dict[name] = tau*critic_dict[name].clone() + \
                (1-tau)*target_critic_dict[name].clone()

    def save(self, file="saves"):
        print("-------------SAVING----------------")
        torch.save({'state_dict': self.critic.state_dict()}, os.path.join(file, "c.pt"))
        torch.save({'state_dict': self.target_critic.state_dict()},os.path.join(file, "ct.pt"))
        torch.save({'state_dict': self.actor.state_dict()},os.path.join(file, "a"))
        torch.save({'state_dict': self.target_actor.state_dict()},os.path.join(file, "at.pt"))

    def load(self, file="saves"):
        print("-------------LOADING----------------")
        self.critic.load_state_dict(torch.load(os.path.join(file, "c.pt"))['state_dict'])
        self.target_critic.load_state_dict(torch.load(os.path.join(file, "ct.pt"))['state_dict'])
        self.actor.load_state_dict(torch.load(os.path.join(file, "a"))['state_dict'])
        self.target_actor.load_state_dict(torch.load(os.path.join(file, "at.pt"))['state_dict'])


class ReplayBuffer():
    def __init__(self, memorySize=tg.steps_each_ep, stateInput=0):
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

    def get_all_buffer(self):
        s = torch.stack(self.states)
        a = torch.stack(self.actions)
        sn = torch.stack(self.next_states)
        r = self.rewards
        return s, a, r, sn


class ActorNetwork(nn.Module):
    def __init__(self, lr=tg.lr, in_dims=tg.inputs, fc1_dims=tg.hidden1, fc2_dims=tg.hidden2, out_dims=tg.outputs):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(in_dims,fc1_dims)
        self.bn1 = nn.LayerNorm(fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.bn2 = nn.LayerNorm(fc2_dims)
        self.out = nn.Linear(fc2_dims, out_dims)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        a = self.fc1(state)
        a = self.bn1(a)
        a = F.relu(a)
        a = self.fc2(a)
        a = self.bn2(a)
        a = F.relu(a)
        a = self.out(a)
        return a


class CriticNetwork(nn.Module):
    def __init__(self, lr=tg.lr, in_dims=tg.inputs, fc1_dims=tg.hidden1, fc2_dims=tg.hidden2, out_dims=tg.outputs):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(in_dims,fc1_dims)
        self.bn1 = nn.LayerNorm(fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.bn2 = nn.LayerNorm(fc2_dims)
        self.action_value = nn.Linear(out_dims, fc2_dims)
        self.q  = nn.Linear(fc2_dims, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, state, action):
        q = self.fc1(state)
        q = self.bn1(q)
        q = F.relu(q)
        q = self.fc2(q)
        q = self.bn2(q)
        q = F.relu(q)

        action_value = self.action_value(action)
        action_value = F.relu(action_value)

        q_action = self.q(torch.add(q, action_value))

        return q_action
