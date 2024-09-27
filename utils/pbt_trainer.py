import os
import tqdm
import torch
from torch.utils.data import DataLoader

from utils import util

"""
	Trainer class receives model, optimizer, loss and data to to train, save, load, and test a model.
"""


class trainer:

	def __init__(self, model, model_optimizer, model_loss, train_data,
				test_data, batch_size, train_epoch, filename):
		self.model = model
		self.model_optimizer = model_optimizer
		self.model_loss = model_loss
		self.train_data = train_data
		self.test_data = test_data
		self.batch_size = batch_size
		self.train_epoch = train_epoch
		self.task_id = None
		self.task_score = 0
		self.filename = filename
		self.trainable = None

	def set_id(self, num):
		"""
			Set id for the model.
		"""
		self.task_id = num

	def set_score(self, num):
		"""
			Set score for the model.
		"""
		self.task_score = num

	def save(self):
		"""
			Save a model.
		"""
		individual = dict(model_state_dict=self.model.state_dict(),
						optim_state_dict=self.model_optimizer.state_dict(),
						batch_size=self.batch_size)
		torch.save(individual, self.filename)

	def load(self):
		"""
			Load saved models.
		"""
		if not os.path.isfile(self.filename):
			return

		individual = torch.load(self.filename)
		self.model.load_state_dict(individual['model_state_dict'])
		self.model_optimizer.load_state_dict(individual['optim_state_dict'])
		self.batch_size = individual['batch_size']

	def train(self):
		"""
			Train the model on the provided data set.
		"""
		self.model.train()
		dataloader = DataLoader(self.train_data, self.batch_size, True)
		for epoch in range(self.train_epoch):
			for inputs, labels in tqdm.tqdm(dataloader,
							desc='Train (epoch {}, individual {})'.format(epoch, self.task_id),
							ncols=100, leave=False):
				output = self.model(inputs)
				loss = self.model_loss(output, labels)
				self.model_optimizer.zero_grad()
				loss.backward()
				self.model_optimizer.step()

	def eval(self, use_precision=False):
		"""
			Evaluate model on the provided test set.
		"""
		self.model.eval()
		y_test = []
		y_pred = []
		self.batch_size = 5
		dataloader = tqdm.tqdm(DataLoader(self.test_data, self.batch_size, False),
							desc='Eval (individual {})'.format(self.task_id),
							ncols=80, leave=True)

		with torch.no_grad():
			for inputs, labels in dataloader:
				logits = self.model(inputs)
				batch_pred = torch.argmax(logits, dim=1)
				batch_labels = torch.argmax(labels, dim=1)
				y_test.extend(batch_labels.cpu().tolist())
				y_pred.extend(batch_pred.cpu().tolist())

		_, accuracy, precision, _, _, _ = util.model_scoring(y_pred, y_test)
		self.set_score(accuracy)
		if use_precision:
			self.set_score(precision)
