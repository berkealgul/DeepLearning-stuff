import torch.nn as nn
import torch.nn.functional as F
import torch as T
import numpy as np
import os
from params import TrainConfig as tg


class Agent():
    def __init__(self, gamma=tg.gamma, tau=tg.tau):
        self.actor = ActorNetwork()
        self.critic = CriticNetwork()
        self._actor = ActorNetwork()
        self._critic = CriticNetwork()
        self.replayBuffer = ReplayBuffer()
        self.gamma = gamma
        self.tau = tau

    def predict_action(self, state):
        self.actor.eval()
        action = self.actor(state)
        return action

    def train(self):
        if self.replayBuffer.mem_center < tg.batch_size:
            return

        s, _s, a, r, done = self.replayBuffer.get_all_buffer()

        s = T.from_numpy(s).float()
        _s = T.from_numpy(_s).float()
        r = T.from_numpy(r).float()
        a = T.from_numpy(a).float()

        self._actor.eval()
        self._critic.eval()
        self.critic.eval()

        aVal = self._actor.forward(_s)
        qvt = self._critic.forward(_s, aVal)

        qVal_t = []
        for i in range(len(r)):
            qVal_t.append(r[i] + self.gamma*qvt[i])

        qVal_t = T.stack(qVal_t)
        qVal = self.critic.forward(s, a)

        self.critic.train()

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(qVal_t, qVal)
        critic_loss.to(self.critic.device)
        critic_loss.backward(retain_graph=True)
        self.critic.optimizer.step()

        self.critic.eval()
        mu = self.actor.forward(s)
        self.actor.train()
        actor_loss = self.critic.forward(s, mu)
        actor_loss.to(self.actor.device)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        print("loss  a " + str(actor_loss.item()) + " c " + str(critic_loss.item()))

        self.update_target_networks()

    def update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        _actor_params = self._actor.named_parameters()
        _critic_params = self._critic.named_parameters()

        actor_dict = dict(actor_params)
        _actor_dict = dict(_actor_params)
        critic_dict = dict(critic_params)
        _critic_dict = dict(_critic_params)

        for name in _actor_dict:
            _actor_dict[name] = tau*actor_dict[name].clone() + \
                (1-tau)*_actor_dict[name].clone()

        for name in _critic_dict:
            _critic_dict[name] = tau*critic_dict[name].clone() + \
                (1-tau)*_critic_dict[name].clone()

    def save_model(self, file="saves"):
        print("-------------SAVING----------------")
        T.save({'state_dict': self.critic.state_dict()}, os.path.join(file, "c.pt"))
        T.save({'state_dict': self._critic.state_dict()},os.path.join(file, "ct.pt"))
        T.save({'state_dict': self.actor.state_dict()},os.path.join(file, "a"))
        T.save({'state_dict': self._actor.state_dict()},os.path.join(file, "at.pt"))

    def load_model(self, file="saves"):
        print("-------------LOADING----------------")
        self.critic.load_state_dict(T.load(os.path.join(file, "c.pt"))['state_dict'])
        self._critic.load_state_dict(T.load(os.path.join(file, "ct.pt"))['state_dict'])
        self.actor.load_state_dict(T.load(os.path.join(file, "a"))['state_dict'])
        self._actor.load_state_dict(T.load(os.path.join(file, "at.pt"))['state_dict'])


class ReplayBuffer:
	def __init__(self, mem_size=tg.memory_size, input_shape=(1,tg.inputs),
                n_actions=tg.outputs):
		self.mem_size = mem_size
		self.mem_center = 0
		self.state_memory = np.zeros((self.mem_size, *input_shape),dtype=np.float32)
		self.new_state_memory = np.zeros((self.mem_size, *input_shape),dtype=np.float32)
		self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
		self.reward_memory = np.zeros(self.mem_size,dtype=np.float32)
		self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

	def store_translition(self, state, new_state, action, reward, done):
		index = self.mem_center % self.mem_size
		self.state_memory[index] = state
		self.new_state_memory[index] = new_state
		self.action_memory[index] = action
		self.reward_memory[index] = reward
		self.terminal_memory[index] = 1 - int(done)
		self.mem_center += 1

	def sample_buffer(self, batch_size):
		max_mem = min(self.mem_size, self.mem_center)
		batch = np.random.choice(max_mem, batch_size)

		states = self.state_memory[batch]
		new_states = self.new_state_memory[batch]
		actions = self.action_memory[batch]
		rewards = self.reward_memory[batch]
		terminal = self.terminal_memory[batch]

		return states, new_states, actions, rewards, terminal


class ActorNetwork(nn.Module):
    def __init__(self, lr=tg.lr, in_dims=tg.inputs, fc1_dims=tg.hidden1,
                fc2_dims=tg.hidden2, out_dims=tg.outputs):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(in_dims,fc1_dims)
        self.bn1 = nn.LayerNorm(fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.bn2 = nn.LayerNorm(fc2_dims)
        self.out = nn.Linear(fc2_dims, out_dims)
        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        a = self.fc1(state)
        a = self.bn1(a)
        a = T.tanh(a)
        #a = F.relu(a)
        a = self.fc2(a)
        a = self.bn2(a)
        a = T.tanh(a)
        #a = F.relu(a)
        a = self.out(a)
        a = T.tanh(a)
        return a


class CriticNetwork(nn.Module):
    def __init__(self, lr=tg.lr, in_dims=tg.inputs, fc1_dims=tg.hidden1,
                fc2_dims=tg.hidden2, out_dims=tg.outputs):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(in_dims,fc1_dims)
        self.bn1 = nn.LayerNorm(fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.bn2 = nn.LayerNorm(fc2_dims)
        self.action_value = nn.Linear(out_dims, fc2_dims)
        self.q  = nn.Linear(fc2_dims, 1)
        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, action):
        q = self.fc1(state)
        q = self.bn1(q)
        q = T.tanh(q)
        #q = F.relu(q)
        q = self.fc2(q)
        q = self.bn2(q)
        #q = F.relu(q)
        q = T.tanh(q)

        action_value = self.action_value(action)
        #action_value = F.relu(action_value)
        action_value = T.tanh(action_value)

        q_action = self.q(T.add(q, action_value))

        return q_action
