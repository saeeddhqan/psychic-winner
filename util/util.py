import numpy
import pandas
import random

import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


seed = 123

torch.manual_seed(seed)
random.seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

categorized_columns = ['PaymentMethod', 'PaperlessBilling', 'Contract', 'StreamingMovies', 'StreamingTV', 
        'TechSupport', 'DeviceProtection', 'OnlineBackup', 'OnlineSecurity', 'InternetService', 'MultipleLines',
        'PhoneService', 'Dependents', 'Partner', 'SeniorCitizen', 'gender']
numerical_columns = ['TotalCharges', 'MonthlyCharges', 'tenure']
target_column = 'Churn'
dataset_filename='data/WA_Fn-UseC_-Telco-Customer-Churn.csv'


def integrating_numerical_column(
	column: 'Column name', 
	data: 'Pandas data type'):
	"""
		Substituting string values with column mean
	"""
	# Find the string values in the column
	mask = pandas.to_numeric(data[column], errors='coerce').isna()

	# Replace the string values with the column's mean
	mean_val = numpy.mean(data.loc[~mask, column].astype(float))
	data.loc[mask, column] = mean_val

	# Convert the column back to a numeric dtype
	data[column] = pandas.to_numeric(data[column], errors='coerce')
	return data

def one_hot_encoding(
	categorized_columns: 'Column names that need to be vectorized', 
	data: 'Pandas data type'):
	"""
		Replacing categories with one-hot encoding. In this way, every class has its own column
	"""
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
	return df_final

def standard_rescaling(
	numerical_columns: 'Column names that need to be rescaled',
	data: 'Pandas data type'):
	"""
		Rescaling the numerical_columns so that they have mean 0 and standard deviation 1.
	"""
	scaler = StandardScaler()
	data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
	return data

def minmax_rescaling(
	numerical_columns: 'Column names that need to be rescaled',
	data: 'Pandas data type'):
	"""
		Rescaling the numerical_columns so that they have a value between 0-1.
	"""
	scaler = MinMaxScaler(feature_range = (0,1))
	scaler.fit(data[numerical_columns])
	data[numerical_columns] = pandas.DataFrame(scaler.transform(data[numerical_columns]))
	return data

def model_scoring(
		predicted: 'A 2D list. Each row contains predicted values',
		labels: 'A 2D list. Each row contains label values'):
	"""
		Measuring performance of the model based on confusion matrix and F1 score.
	"""
	y_test = numpy.array(predicted)
	y_pred = numpy.array(labels)

	c_matrix = confusion_matrix(y_test, y_pred)

	TP, FP = tuple(c_matrix[0]) 
	FN, TN = tuple(c_matrix[1])
	# we could use sklearn.metrics accuracy_score, precision_score, recall_score, and f1_score.
	precision = TP / (TP + FP) 
	accuracy = (TP + TN) / (TP + FP + FN + TN)
	recall = TP / (TP + FN)
	f1_score = 2 * (precision * recall) / (precision + recall)

	return c_matrix, accuracy, precision, recall, f1_score


def data_splitter_tensor_binary(
		data: 'Pandas data type',
		target_column: 'Target column',
		batch_size: 'Data batch size',
		test_perc: 'Test size'
	):
	"""
		Splitting data based on test_perc, batch_size and target_column and returning DataLoader instances
	"""
	X = data.drop(target_column, axis=1)
	y = data[target_column]

	# Split the data into train and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_perc, random_state=seed)
	input_size = X_train.shape[1]
	classifiers_size = 2
	test_size = X_test.shape[0]


	one_hot_matrix = torch.eye(classifiers_size).to(device)

	train_target, test_target = one_hot_matrix[y_train.values], one_hot_matrix[y_test.values]

	train_data = torch.tensor(X_train.values, dtype=torch.float32, device=device)
	test_data = torch.tensor(X_test.values, dtype=torch.float32, device=device)

	train_dataset = TensorDataset(train_data, train_target)
	# Turning off shuffle simply decreases performance by ~1%
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_dataset = TensorDataset(test_data, test_target)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

	return train_loader, test_loader, input_size, classifiers_size, test_size
