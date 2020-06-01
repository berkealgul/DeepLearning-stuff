import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Agent:
	def __init__(self, lr, gamma, epsilon, env_name, in_dims, n_actions, batch_size, mem_size,
	 			 hid_dims=512, eps_min=0.01, eps_dec=5e-7, replace=1000 ,chkp_dir='saves/'):
		self.gamma = gamma
		self.epsilon = epsilon
		self.eps_dec = eps_dec
		self.eps_min = eps_min
		self.replace = replace
		self.action_space = [i for i in range(n_actions)]
		self.learn_step_cnt = 0
		self.batch_size = batch_size
		self.losses = []

		self.memory = ReplayBuffer(mem_size, in_dims, n_actions)
		self.q_eval = QNetwork(lr, in_dims, hid_dims, n_actions, "Qeval", chkp_dir+env_name)
		self.q_next = QNetwork(lr, in_dims, hid_dims, n_actions, "Qnext", chkp_dir+env_name)

		print("Network device is: ", self.q_eval.device)


	def choice_action(self, observation):
		if np.random.random() > self.epsilon:
			state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
			action = self.q_eval.forward(state)
			action = T.argmax(action).item()
		else:
			action = np.random.choice(self.action_space)

		return action

	def store_translition(self, state, new_state, action, reward, done):
		self.memory.store_translition(state, new_state, action, reward, done)

	def sample_memory(self):
		states, new_states, actions, rewards, done = \
								self.memory.sample_buffer(self.batch_size)

		states = T.tensor(states).to(self.q_eval.device)
		new_states = T.tensor(new_states).to(self.q_eval.device)
		actions = T.tensor(actions).to(self.q_eval.device)
		rewards = T.tensor(rewards).to(self.q_eval.device)
		done = T.tensor(done).to(self.q_eval.device)

		return states, new_states, actions, rewards, done

	def replace_target_net(self):
		if self.learn_step_cnt % self.replace == 0:
			self.q_next.load_state_dict(self.q_eval.state_dict())

	def decrement_epsilon(self):
		if self.epsilon > self.eps_min:
			self.epsilon -= self.eps_dec
		else:
			self.epsilon = self.eps_min

	def save_model(self):
		print("Saving")
		self.q_eval.save_checkpoint()
		self.q_next.save_checkpoint()

	def load_model(self):
		print("Loading")
		try:
			self.q_eval.load_checkpoint()
			self.q_next.load_checkpoint()
		except:
			print("Load failed")

	def train(self):
		if self.memory.mem_center < self.batch_size:
			return

		states, new_states, actions, rewards, terminal = self.sample_memory()

		self.q_eval.optimizer.zero_grad()
		self.replace_target_net()

		i = np.arange(self.batch_size)
		q_pred = self.q_eval.forward(states)[i, actions]
		q_next = self.q_next.forward(new_states).max(dim=1)[0]

		q_next[terminal] = 0.0
		q_target = rewards + self.gamma*q_next

		loss = self.q_eval.loss(q_pred, q_target).to(self.q_eval.device)

		self.losses.append(loss.item())

		loss.backward()
		self.q_eval.optimizer.step()

		self.learn_step_cnt += 1
		self.decrement_epsilon()

	def get_avg_loss(self):
		avg_loss = np.mean(self.losses)
		self.losses.clear()
		return avg_loss


class QNetwork(nn.Module):
	def __init__(self, lr, in_dims, hid_dims, n_actions, name, chkp_dir):
		super(QNetwork, self).__init__()
		self.chpk_file = os.path.join(chkp_dir, name)

		self.conv1 = nn.Conv2d(in_dims[0], 32, 8, stride=4)
		self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
		self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

		fc_inputs = self.calculate_fc_inputs(in_dims)

		self.fc1 = nn.Linear(fc_inputs, hid_dims)
		self.fc2 = nn.Linear(hid_dims, n_actions)

		self.loss = nn.MSELoss()
		self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
		self.device = T.device("cudo:0" if T.cuda.is_available() else "cpu")
		self.to(self.device)


	def calculate_fc_inputs(self, in_dims):
		dims = T.zeros(1, *in_dims)
		dims = self.conv1(dims)
		dims = self.conv2(dims)
		dims = self.conv3(dims)
		return int(np.prod(dims.size()))

	def forward(self, state):
		conv = F.relu(self.conv1(state))
		conv = F.relu(self.conv2(conv))
		conv = F.relu(self.conv3(conv))

		#conv size = (filter h w)
		conv_state = conv.view(conv.size()[0], -1)
		action = F.relu(self.fc1(conv_state))
		action = self.fc2(action)

		return action

	def save_checkpoint(self):
		print("SAVING...")
		T.save(self.state_dict(), self.chpk_file)

	def load_checkpoint(self):
		print("LOADING...")
		self.load_state_dict(T.load(self.chpk_file))


class ReplayBuffer:
	def __init__(self, mem_size, input_shape, n_actions):
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
