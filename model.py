
import pandas

import tqdm

from utils import util, torch_model_arch

import torch
import torch.nn as nn


"""
	This model is a simple manually designed model.
"""

data = pandas.read_csv(util.dataset_filename)

# Dropping customerID
data.drop(['customerID'], axis=1, inplace=True)

# ----- Integrating numerical_columns

# TotalCharges contains some str values. I noticed it when I looked at data.info
data = util.integrating_numerical_column('TotalCharges', data)
data[util.target_column] = data[util.target_column].replace(['Yes', 'No'], [1, 0])
data = util.standard_rescaling(util.numerical_columns, data)  # My preference
data = util.one_hot_encoding(util.categorized_columns, data)

# ----- Tensors' journey

batch_size = 5
test_prob = 0.1
dropout_prob = 0.15
epochs = 6


train_loader, test_loader, input_size, \
	classifiers_size, test_size = util.data_splitter_tensor_binary(data, util.target_column, batch_size, test_prob)


model = torch_model_arch.net(input_size, classifiers_size, dropout_prob)
# model.load_state_dict(torch.load('model.pth'))
model.to(util.device)
model_loss = nn.CrossEntropyLoss().to(util.device)
model_optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-5, momentum=0.3)
# It only works on modern GPUs. It actually enhances the training speed.
# model = torch.compile(model, mode='reduce-overhead')


def main():
	# Train the model
	for epoch in range(epochs):
		# Training loop
		for (inputs, labels) in tqdm.tqdm(train_loader,
				desc='Train(epoch: {:d})'.format(epoch),
				ncols=120, leave=False):
			outputs = model(inputs)
			loss = model_loss(outputs, labels)
			model_optimizer.zero_grad()
			loss.backward()
			model_optimizer.step()

		model.eval()

		y_test = []
		y_pred = []

		with torch.no_grad():
			for inputs, labels in test_loader:
				logits = model(inputs)
				batch_pred = torch.argmax(logits, dim=1)
				batch_labels = torch.argmax(labels, dim=1)

				y_test.extend(batch_labels.cpu().tolist())
				y_pred.extend(batch_pred.cpu().tolist())

		model.train()
		c_matrix, accuracy, precision, recall, f1_score = util.model_scoring(y_pred, y_test)

		print('\nConfusion matrix:')
		print(c_matrix)
		print('Accuracy:', accuracy)
		print('Precision:', precision)
		print('Recall:', recall)
		print('F1-score:', f1_score)
	return accuracy


if __name__ == '__main__':
	main()
