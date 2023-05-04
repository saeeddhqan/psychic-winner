import torch.nn as nn

params = [256, 256, 256]

class net(nn.Module):
	def __init__(self, input_size, output_size, dropout_prob):
		super().__init__()
		# Using LayerNorm reduces epoch size perfectly
		self.encoder = nn.Sequential(
			nn.Linear(input_size, params[0]), nn.Sigmoid(), nn.LayerNorm(params[0]), nn.Dropout(dropout_prob),
			nn.Linear(params[0], params[1]), nn.ReLU(), nn.LayerNorm(params[1]), nn.Dropout(dropout_prob),
			nn.Linear(params[1], output_size),
		)

	def forward(self, x):
		logits = self.encoder(x)
		return logits

class net_search(nn.Module):
	def __init__(self, input_size, output_size, dropout_prob, act_functions):
		"""
			Utilizing this network for Neural Architecture Search(NAS).
			This is a mirror of net.
		"""
		super().__init__()
		self.encoder = nn.Sequential(
			nn.Linear(input_size, params[0]), act_functions[0](), nn.LayerNorm(params[0]), nn.Dropout(dropout_prob),
			nn.Linear(params[0], params[1]), act_functions[1](), nn.LayerNorm(params[1]), nn.Dropout(dropout_prob),
			nn.Linear(params[1], output_size),
		)

	def forward(self, x):
		logits = self.encoder(x)
		return logits
