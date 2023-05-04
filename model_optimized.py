
import pandas

import torch
import torch.nn as nn

from util import util, torch_model_arch

"""
	This model is a mirror of manually designed model(model.py) but 
	optimized with NAS and hyperparameter space search.
"""

data = pandas.read_csv(util.dataset_filename)
data.drop(['customerID'], axis=1, inplace=True)

#### Integrating numerical_columns

data = util.integrating_numerical_column('TotalCharges', data)
data[util.target_column] = data[util.target_column].replace(['Yes', 'No'], [1, 0])
data = util.standard_rescaling(util.numerical_columns, data)
data = util.one_hot_encoding(util.categorized_columns, data)

#### Tensors' journey

batch_size = 5
epochs = 6
test_prob = 0.1
lr = 3.8093424453229945e-05
dropout_prob = 0.05195051965108121
train_loader, test_loader, input_size, \
	classifiers_size, test_size = util.data_splitter_tensor_binary(data, util.target_column, batch_size, test_prob)


def main():
	torch.manual_seed(util.seed)
	model = torch_model_arch.net_search(input_size, classifiers_size, dropout_prob, [nn.Tanh, nn.ReLU])
	model.to(util.device)
	model_loss = nn.CrossEntropyLoss().to(util.device)
	model_optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
	for epoch in range(epochs):
		for i, (inputs, labels) in enumerate(train_loader, 0):
			outputs = model(inputs)
			loss = model_loss(outputs, labels)
			model_optimizer.zero_grad()
			loss.backward()
			model_optimizer.step()
	torch.save(model.state_dict(), 'model_optimized.pth')

def eval_mode():
	model = torch_model_arch.net_search(input_size, classifiers_size, dropout_prob, [nn.Tanh, nn.ReLU])
	model.load_state_dict(torch.load('model_optimized.pth'))
	model.to(util.device)
	model.eval()
	y_test = []
	y_pred = []
	correct = 0
	with torch.no_grad():
		for (inputs, labels) in test_loader:
			logits = model(inputs)
			batch_pred = torch.argmax(logits, dim=1)
			batch_labels = torch.argmax(labels, dim=1)
			y_test.extend(batch_labels.cpu().tolist())
			y_pred.extend(batch_pred.cpu().tolist())
			correct += (batch_pred == batch_labels).sum().item()
	c_matrix, accuracy, precision, recall, f1_score = util.model_scoring(y_pred, y_test)
	print('\nConfusion matrix:')
	print(c_matrix)
	print('Accuracy:', accuracy)
	print('Precision:', precision)
	print('Recall:', recall)
	print('F1-score:', f1_score)

main()
eval_mode()
