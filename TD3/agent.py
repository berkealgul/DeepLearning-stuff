import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


class Agent:
	def __init__(self, alpha, beta, input_shape, n_actions, tau,  env, name,
				gamma=0.99, update_actor_interval=2, layer_1_dims=400,
				layer_2_dims=300, noise=0.1, mem_size=1000000, batch_size=100,
			 	warmup=1000):

		# max and min actions for noise
		self.max_action = env.action_space.high
		self.min_action = env.action_space.low
		self.n_actions = n_actions

		self.noise = noise
		self.gamma = gamma
		self.update_actor_interval = update_actor_interval
		self.tau = tau
		self.batch_size = batch_size
		self.warmup = warmup

		# Time steps for learning and performinf action
		self.learn_step = 0
		self.time_step = 0

		self.critic_1 = CriticNetwork(beta, layer_1_dims, layer_2_dims, input_shape,
										n_actions, "critic_1", "saves/"+name)
		self.critic_2 = CriticNetwork(beta, layer_1_dims, layer_2_dims, input_shape,
										n_actions, "critic_2", "saves/"+name)
		self.actor = ActorNetwork(alpha, layer_1_dims, layer_2_dims, input_shape,
										n_actions, "actor", "saves/"+name)

		self.target_critic_1 = CriticNetwork(beta, layer_1_dims, layer_2_dims, input_shape,
		 								n_actions, "target_critic_1", "saves/"+name)
		self.target_critic_2 = CriticNetwork(beta, layer_1_dims, layer_2_dims, input_shape,
										n_actions, "target_critic_2", "saves/"+name)
		self.target_actor = ActorNetwork(alpha, layer_1_dims, layer_2_dims, input_shape,
										n_actions, "target_actor", "saves/"+name)

		self.memory = ReplayBuffer(mem_size, input_shape, n_actions)

		self.update_target_network_parameters(tau=1)

	def update_target_network_parameters(self, tau=None):
		if tau is None:
			tau = self.tau

		actor_params = self.actor.named_parameters()
		critic_1_params = self.critic_1.named_parameters()
		critic_2_params = self.critic_2.named_parameters()
		target_actor_params = self.target_actor.named_parameters()
		target_critic_1_params = self.target_critic_1.named_parameters()
		target_critic_2_params = self.target_critic_2.named_parameters()

		actor_dict = dict(actor_params)
		critic_1_dict = dict(critic_1_params)
		critic_2_dict = dict(critic_2_params)
		target_actor_dict = dict(target_actor_params)
		target_critic_1_dict = dict(target_critic_1_params)
		target_critic_2_dict = dict(target_critic_2_params)

		for name in target_actor_dict:
			target_actor_dict[name] = tau*actor_dict[name].clone() + \
				(1-tau)*target_actor_dict[name].clone()

		for name in target_critic_1_dict:
			target_critic_1_dict[name] = tau*critic_1_dict[name].clone() + \
				(1-tau)*target_critic_1_dict[name].clone()

		for name in target_critic_2_dict:
			target_critic_2_dict[name] = tau*critic_2_dict[name].clone() + \
				(1-tau)*target_critic_2_dict[name].clone()

		self.target_actor.load_state_dict(target_actor_dict)
		self.target_critic_1.load_state_dict(target_critic_1_dict)
		self.target_critic_2.load_state_dict(target_critic_2_dict)

	def choose_action(self, observation):
		if self.time_step < self.warmup:
			mu = T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,)))
		else:
			state = T.tensor(observation, dtype=T.float).to(self.actor.device)
			mu = self.actor.forward(state).to(self.actor.device)

		action = mu + T.tensor(np.random.normal(scale=self.noise), dtype=T.float).to(self.actor.device)
		action = T.clamp(action, self.min_action[0], self.max_action[0])

		self.time_step += 1

		return action.cpu().detach().numpy()

	def store_memory(self, state, new_state, action, reward, done):
		self.memory.store_transition(state, new_state, action, reward, done)

	def learn(self):
		if self.memory.mem_center < self.batch_size:
			return

		state, new_state, action, reward, done = self.memory.sample_buffer(self.batch_size)

		#All devices same so device type dosent matter here
		state = T.tensor(state, dtype=T.float).to(self.actor.device)
		new_state = T.tensor(new_state, dtype=T.float).to(self.actor.device)
		action = T.tensor(action, dtype=T.float).to(self.actor.device)
		reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
		done = T.tensor(done).to(self.actor.device)

		target_action = self.target_actor.forward(new_state)
		target_action = target_action + T.clamp(T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
		target_action = T.clamp(target_action, self.min_action[0], self.max_action[0])

		_q1 = self.target_critic_1.forward(new_state, target_action)
		_q2 = self.target_critic_2.forward(new_state, target_action)

		_q1[done] = 0.0
		_q2[done] = 0.0

		_q1 = _q1.view(-1)
		_q2 = _q2.view(-1)

		q1 = self.critic_1.forward(state, action)
		q2 = self.critic_2.forward(state, action)

		target = reward + self.gamma * T.min(_q2, _q1)
		target = target.view(self.batch_size, 1)

		self.critic_1.optimizer.zero_grad()
		self.critic_2.optimizer.zero_grad()

		q1_loss = F.mse_loss(target, q1)
		q2_loss = F.mse_loss(target, q2)
		critic_loss = q1_loss + q2_loss
		critic_loss.backward()

		self.critic_1.optimizer.step()
		self.critic_2.optimizer.step()

		self.learn_step += 1

		if self.learn_step % self.update_actor_interval == 0:
			# Update Actor network and target networks
			self.actor.optimizer.zero_grad()
			actor_loss = self.critic_1.forward(state, self.actor.forward(state))
			actor_loss = -T.mean(actor_loss)
			actor_loss.backward()
			self.actor.optimizer.step()

			self.update_target_network_parameters()

	def save(self):
		print("---Saving---")
		self.actor.save_checkpoint()
		self.critic_1.save_checkpoint()
		self.critic_2.save_checkpoint()
		self.target_actor.save_checkpoint()
		self.target_critic_1.save_checkpoint()
		self.target_critic_2.save_checkpoint()

	def load(self):
		print("---Loading---")
		self.actor.load_checkpoint()
		self.critic_1.load_checkpoint()
		self.critic_2.load_checkpoint()
		self.target_actor.load_checkpoint()
		self.target_critic_1.load_checkpoint()
		self.target_critic_2.load_checkpoint()

	def get_device(self):
		return self.actor.device


class ActorNetwork(nn.Module):
	def __init__(self, lr, fc1_dims, fc2_dims, input_shape, n_actions, name, chkp_dir):
		super(ActorNetwork, self).__init__()

		self.fc1 = nn.Linear(*input_shape, fc1_dims)
		self.fc2 = nn.Linear(fc1_dims, fc2_dims)
		self.out = nn.Linear(fc2_dims, n_actions)

		self.chkp_file = os.path.join(chkp_dir, name)

		self.optimizer = optim.Adam(self.parameters(), lr=lr)

		self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
		self.to(self.device)

	def forward(self, state):
		action = self.fc1(state)
		action = F.relu(action)
		action = self.fc2(action)
		action = F.relu(action)
		action = self.out(action)
		action = T.tanh(action)
		return action

	def save_checkpoint(self):
		T.save(self.state_dict(), self.chkp_file)

	def load_checkpoint(self):
		self.load_state_dict(T.load(self.chkp_file, map_location=self.device))


class CriticNetwork(nn.Module):
	def __init__(self, lr, fc1_dims, fc2_dims, state_shape, n_actions, name, chkp_dir):
		super(CriticNetwork, self).__init__()

		self.fc1 = nn.Linear(state_shape[0]+n_actions, fc1_dims)
		self.fc2 = nn.Linear(fc1_dims, fc2_dims)
		self.out = nn.Linear(fc2_dims, 1)

		self.chkp_file = os.path.join(chkp_dir, name)

		self.optimizer = optim.Adam(self.parameters(), lr=lr)

		self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
		self.to(self.device)

	def forward(self, state, action):
		q = self.fc1(T.cat([state, action], dim=1))
		q = F.relu(q)
		q = self.fc2(q)
		q = F.relu(q)
		q = self.out(q)

		return q

	def save_checkpoint(self):
		T.save(self.state_dict(), self.chkp_file)

	def load_checkpoint(self):
		self.load_state_dict(T.load(self.chkp_file, map_location=self.device))


class ReplayBuffer:
	def __init__(self, mem_size, input_shape, n_actions):
		self.mem_size = mem_size
		self.mem_center = 0
		self.state_memory = np.zeros((self.mem_size, *input_shape))
		self.new_state_memory = np.zeros((self.mem_size, *input_shape))
		self.action_memory = np.zeros((self.mem_size, n_actions))
		self.reward_memory = np.zeros(self.mem_size)
		self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

	def store_transition(self, state, new_state, action, reward, done):
		index = self.mem_center % self.mem_size
		self.state_memory[index] = state
		self.new_state_memory[index] = new_state
		self.action_memory[index] = action
		self.reward_memory[index] = reward
		self.terminal_memory[index] = done
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
