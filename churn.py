
import pandas
import numpy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
from util import util

data = pandas.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")


# print(data.head(5))
# Dropping customerID
data.drop(["customerID"], axis=1, inplace=True)

# Getting some insights
# data.info()

# Recognizing number of categories of each column
# for column in data.columns:
# 	print(column, len(data[column].unique()))
# 	print(data[column].value_counts())
# 	print('-'*10)

# Classified data without target('Churn')
categorized_columns = ['PaymentMethod', 'PaperlessBilling', 'Contract', 'StreamingMovies', 'StreamingTV', 
		'TechSupport', 'DeviceProtection', 'OnlineBackup', 'OnlineSecurity', 'InternetService', 'MultipleLines',
		'PhoneService', 'Dependents', 'Partner', 'SeniorCitizen', 'gender']
numerical_columns = ['TotalCharges', 'MonthlyCharges', 'tenure']
target_mapping = {'Yes': 1.0, 'No': 0.0}

# Integrating numerical_columns

# TotalCharges contains some str values. I noticed when I looked at data.info
data = util.integrating_numerical_column(data, numerical_columns[0])

# Rescaling the numerical_columns so that they have mean 0 and standard deviation 1
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# data.info()

# Replacing categories with one-hot encoding. In this way, every class has its own column

data_encoded = pandas.DataFrame()

for col in categorized_columns:
	encoder = OneHotEncoder()
	encoder.fit(data[[col]])
	col_encoded = encoder.transform(data[[col]]).toarray()
	categories = encoder.categories_[0]
	new_col_names = [f"{col}_{str(cat).lower().replace(' ', '_')}" for cat in categories]
	df_col_encoded = pandas.DataFrame(col_encoded, columns=new_col_names)
	data_encoded = pandas.concat([data_encoded, df_col_encoded], axis=1)

df_final = pandas.concat([data.drop(categorized_columns, axis=1), data_encoded], axis=1)
# df_final.info()

# Assuming target column is "Churn"
X = df_final.drop("Churn", axis=1)
y = df_final["Churn"].replace(target_mapping)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=util.seed)
input_size = X_train.shape[1]
classifiers_size = 2
test_size = X_test.shape[0]
dropout_prob = 0.15
batch_size = 5
epochs = 13

one_hot_matrix = torch.eye(classifiers_size).to(util.device)

train_target, test_target = one_hot_matrix[y_train.values], one_hot_matrix[y_test.values]

train_data = torch.tensor(X_train.values, dtype=torch.float32, device=util.device)
test_train = torch.tensor(X_test.values, dtype=torch.float32, device=util.device)

train_dataset = TensorDataset(train_data, train_target)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Turning off shuffle simply decreases performance by ~1%
test_dataset = TensorDataset(test_train, test_target)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Modeling starts here

class net(nn.Module):
	def __init__(self, input_size, output_size, dropout_prob):
		super().__init__()
		# Using LayerNorm reduces epoch size perfectly
		self.encoder = nn.Sequential(
			nn.Linear(input_size, 256), nn.Sigmoid(), nn.LayerNorm(256), nn.Dropout(dropout_prob),
			nn.Linear(256, 256), nn.ReLU(), nn.LayerNorm(256), nn.Dropout(dropout_prob),
			nn.Linear(256, output_size),
		)

	def forward(self, x):
		logits = self.encoder(x)
		return logits


model = net(input_size, classifiers_size, dropout_prob)
# model.load_state_dict(torch.load('churn.pth'))
model.to(util.device)
model_loss = nn.CrossEntropyLoss().to(util.device)
model_optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-5)
# It only works on modern GPUs. It actually enhances the training speed.
# model = torch.compile(model, mode='reduce-overhead')


# Train the model
for epoch in range(epochs):
	running_loss = 0.0
	for i, (inputs, labels) in enumerate(train_loader, 0):
		outputs = model(inputs)
		loss = model_loss(outputs, labels)
		model_optimizer.zero_grad()
		loss.backward()
		model_optimizer.step()

		running_loss += loss.item()
		if i % 100 == 99:
			print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
			running_loss = 0.0

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


	y_test = numpy.array(y_test)
	y_pred = numpy.array(y_pred)
	c_matrix = confusion_matrix(y_test, y_pred)

	TP, FP = tuple(c_matrix[0]) 
	FN, TN = tuple(c_matrix[1])
	# we could use sklearn.metrics accuracy_score, precision_score, recall_score, and f1_score.
	precision = TP / (TP + FP) 
	accuracy = (TP + TN) / test_size
	recall = TP / (TP + FN)
	f1_score = 2 * (precision * recall) / (precision + recall)
	print('\nConfusion matrix:')
	print(c_matrix)
	print('Accuracy:', accuracy)
	print('Precision:', precision)
	print('Recall:', recall)
	print('F1-score:', f1_score)

	model.train()

torch.save(model.state_dict(), 'churn.pth')
