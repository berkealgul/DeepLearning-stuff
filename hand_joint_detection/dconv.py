import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DConv(nn.Module):
	def __init__(self, inp_dim, out_dim, lr=0.005, momentum=0.95):
		super(DConv, self).__init__()

		self.conv1 = nn.Conv2d(3, 32, 8, stride=4)
		self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
		self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

		dims = self.calculate_fc_dims(inp_dim)
		self.fc1 = nn.Linear(dims, 16)
		self.fc2 = nn.Linear(16, out_dim)

		self.loss = nn.MSELoss()
		self.optimizer = optim.Adam(self.parameters(), lr=lr)
		self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
		self.chpk_file = os.path.join("saves/", "model")
		self.to(self.device)

	def forward(self, inp):
		inp = F.relu(self.conv1(inp))
		inp = F.relu(self.conv2(inp))
		inp = F.relu(self.conv3(inp))
		inp = inp.view(inp.size()[0], -1)
		inp = F.relu(self.fc1(inp))
		inp = F.relu(self.fc2(inp))
		#inp = T.sigmoid(self.fc2(inp))
		#last layer can change between sigmo or relu
		return inp

	def calculate_fc_dims(self, inp_dim):
		dims = T.zeros(1, 3, inp_dim, inp_dim)
		dims = self.conv1(dims)
		dims = self.conv2(dims)
		dims = self.conv3(dims)
		return int(np.prod(dims.size()))

	def learn(self, pred, answer):
		self.train()

		self.optimizer.zero_grad()
		loss = self.loss(pred, answer).to(self.device)
		loss.backward()
		self.optimizer.step()

		self.eval()
		return loss.item()

	def save_checkpoint(self):
		print("SAVING...")
		T.save(self.state_dict(), self.chpk_file)

	def load_checkpoint(self):
		print("LOADING...")
		self.load_state_dict(T.load(self.chpk_file, map_location=self.device))
