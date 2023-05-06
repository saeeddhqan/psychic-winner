import numpy
import pandas
import random

import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

from util import util, torch_model_arch

import torch
import torch.nn as nn


"""
	This model is a manually designed model.
"""

data = pandas.read_csv(util.dataset_filename)

# Dropping customerID
data.drop(['customerID'], axis=1, inplace=True)

#### Integrating numerical_columns

# TotalCharges contains some str values. I noticed it when I looked at data.info
data = util.integrating_numerical_column('TotalCharges', data)
data[util.target_column] = data[util.target_column].replace(['Yes', 'No'], [1, 0])
data = util.standard_rescaling(util.numerical_columns, data) # My preference
data = util.one_hot_encoding(util.categorized_columns, data)

#### Tensors' journey

batch_size = 5
test_perc = 0.1
dropout_prob = 0.15
epochs = 13


train_loader, test_loader, input_size, \
	classifiers_size, test_size = util.data_splitter_tensor_binary(data, util.target_column, batch_size, test_perc)


model = torch_model_arch.net(input_size, classifiers_size, dropout_prob)
# model.load_state_dict(torch.load('model.pth'))
model.to(util.device)
model_loss = nn.CrossEntropyLoss().to(util.device)
model_optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-5)
# It only works on modern GPUs. It actually enhances the training speed.
# model = torch.compile(model, mode='reduce-overhead')

accuracy = 0.0
# Train the model
for epoch in range(epochs):
	running_loss = 0.0
	# Training loop
	for (inputs, labels) in tqdm.tqdm(train_loader,
			desc='Train(epoch: {:d}, test accuracy: {:.3f})'.format(epoch, accuracy),
			ncols=120, leave=False):
		outputs = model(inputs)
		loss = model_loss(outputs, labels)
		model_optimizer.zero_grad()
		loss.backward()
		model_optimizer.step()

		running_loss += loss.item()

	# Turning off backprop, etc. for evaluation
	model.eval()

	y_test = []
	y_pred = []

	with torch.no_grad():
		for (inputs, labels) in test_loader:
			logits = model(inputs)
			# normalized = torch.softmax(logits, dim=1) # Doing this only because of roc_auc_score
			batch_pred = torch.argmax(logits, dim=1)
			batch_labels = torch.argmax(labels, dim=1)
			# probs = normalized[torch.arange(normalized.size(0)), batch_pred]
			y_test.extend(batch_labels.cpu().tolist())
			y_pred.extend(batch_pred.cpu().tolist())

	c_matrix, accuracy, precision, recall, f1_score = util.model_scoring(y_pred, y_test)


	model.train()

print('\nConfusion matrix:')
print(c_matrix)
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1_score)

torch.save(model.state_dict(), 'model.pth')
