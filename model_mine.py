import torch
import torch.nn as nn
import torchvision

class TextLocalizationCNN(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
		self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
		self.fc1 = nn.Linear(64 * 112 * 56, 512)
		self.dropout = nn.Dropout(0.06)
		self.layernorm3 = nn.LayerNorm(512)
		self.fc2 = nn.Linear(512, 448*224)
		self.relu = nn.ReLU()

	def forward(self, x):
		raw_shape = x.shape
		x = self.conv1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		x = self.conv2(x)
		x = self.relu(x)
		x = self.maxpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc1(x)
		x = self.relu(x)
		x = self.dropout(x)
		x = self.layernorm3(x)
		x = self.fc2(x)
		x = self.relu(x)
		x = x.reshape(raw_shape[0], raw_shape[1], raw_shape[2], raw_shape[3])
		return x


class TextLocalizationClass(nn.Module):
	def __init__(self, input_size, output_size, dropout_prob):
		super().__init__()
		# Using LayerNorm reduces epoch size
		self.encoder = nn.Sequential(
			nn.Linear(input_size, params[0]), nn.Sigmoid(), nn.LayerNorm(params[0]), nn.Dropout(dropout_prob),
			nn.Linear(params[0], params[1]), nn.ReLU(), nn.LayerNorm(params[1]), nn.Dropout(dropout_prob),
			nn.Linear(params[1], output_size),
		)

	def forward(self, x):
		logits = self.encoder(x)
		return logits

