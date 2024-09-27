
import pandas

import torch
import torch.nn as nn

from utils import util, torch_model_arch

"""
	Here we try to evaluate the model that optimized with a modification of Population Based Training(PBT).
"""

data = pandas.read_csv(util.dataset_filename)
data.drop(['customerID'], axis=1, inplace=True)

# ---- Integrating numerical_columns

data = util.integrating_numerical_column('TotalCharges', data)
data[util.target_column] = data[util.target_column].replace(['Yes', 'No'], [1, 0])
data = util.standard_rescaling(util.numerical_columns, data)
data = util.one_hot_encoding(util.categorized_columns, data)

# ---- Tensors' journey

batch_size = 5
test_prob = 0.1
dropout_prob = 0.05
# model_path = 'data/best_performed_model_pbt_precision.pth'
model_path = 'data/best_performed_model_pbt_accuracy.pth'

train_loader, test_loader, input_size, \
	classifiers_size, test_size = util.data_splitter_tensor_binary(data, util.target_column, batch_size, test_prob)


def eval_mode():
	"""
		It loads the best performed model(model_path) and evaluate it with confusion matrix, etc.
	"""

	model = torch_model_arch.net_search(input_size, classifiers_size, dropout_prob, [nn.Tanh, nn.ReLU])
	load = torch.load(model_path)
	model.load_state_dict(load['model_state_dict'])
	model.to(util.device)
	model.eval()
	y_test = []
	y_pred = []

	with torch.no_grad():
		for (inputs, labels) in test_loader:
			logits = model(inputs)
			batch_pred = torch.argmax(logits, dim=1)
			batch_labels = torch.argmax(labels, dim=1)
			y_test.extend(batch_labels.cpu().tolist())
			y_pred.extend(batch_pred.cpu().tolist())

	c_matrix, accuracy, precision, recall, f1_score, f05 = util.model_scoring(y_pred, y_test)
	print('\nConfusion matrix:')
	print(c_matrix)
	print('Accuracy:', accuracy)
	print('Precision:', precision)
	print('Recall:', recall)
	print('F1-score:', f1_score)
	print('F05-score:', f05)

	return accuracy, precision


if __name__ == '__main__':
	eval_mode()
