
import numpy
import pandas
import random

import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, fbeta_score

from imblearn.over_sampling import SMOTE

"""
	Pandas utilities, such as integration, rescaling, spliting and scoring functions are here.
"""

# We use this random seed for the project
seed = 123

torch.manual_seed(seed)
random.seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Columns that contain crisp data
categorized_columns = ['PaymentMethod', 'PaperlessBilling', 'Contract', 'StreamingMovies', 'StreamingTV', 
	'TechSupport', 'DeviceProtection', 'OnlineBackup', 'OnlineSecurity', 'InternetService', 'MultipleLines',
	'PhoneService', 'Dependents', 'Partner', 'SeniorCitizen', 'gender']

# Columns that contain numerical data
numerical_columns = ['TotalCharges', 'MonthlyCharges', 'tenure']

target_column = 'Churn'
dataset_filename = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'


def integrating_numerical_column(
	column: 'Column name', 
	data: 'Pandas data type'):
	"""
		Substituting string values with column mean.
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
		Replacing categories with one-hot encoding. In this way, every class has its own column.
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
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaler.fit(data[numerical_columns])
	data[numerical_columns] = pandas.DataFrame(scaler.transform(data[numerical_columns]))
	return data


def model_scoring(
	predicted: 'A 2D list. Each row contains predicted values',
	labels: 'A 2D list. Each row contains label values'):
	"""
		Measuring performance of the model based on confusion matrix and F1 score.
	"""

	c_matrix = confusion_matrix(labels, predicted)

	accuracy = accuracy_score(labels, predicted)
	precision = precision_score(labels, predicted)
	recall = recall_score(labels, predicted)
	f1 = f1_score(labels, predicted)
	f05_score = fbeta_score(labels, predicted, beta=0.5)

	return c_matrix, accuracy, precision, recall, f1, f05_score


def data_splitter_tensor_binary(
	data: 'Pandas data type',
	target_column: 'Target column',
	batch_size: 'Data batch size',
	test_perc: 'Test size',
	dataloader_ins: bool = True):
	"""
		Splitting data based on test_perc, batch_size and target_column and returning DataLoader instances if its True.
	"""
	X = data.drop(target_column, axis=1)
	y = data[target_column]

	# oversampler = SMOTE(sampling_strategy=1)

	# Split the data into train and test sets
	# _, test_x, _, test_y = train_test_split(X, y, test_size=test_perc, random_state=seed)
	# X, y = oversampler.fit_resample(X, y)
	# train_x, _, train_y, _ = train_test_split(X, y, test_size=test_perc, random_state=seed)

	train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=test_perc, random_state=seed)

	input_size = train_x.shape[1]
	classifiers_size = 2
	test_size = test_x.shape[0]

	one_hot_matrix = torch.eye(classifiers_size).to(device)

	train_target, test_target = one_hot_matrix[train_y.values], one_hot_matrix[test_y.values]

	train_data = torch.tensor(train_x.values, dtype=torch.float32, device=device)
	test_data = torch.tensor(test_x.values, dtype=torch.float32, device=device)
	train_dataset = TensorDataset(train_data, train_target)
	test_dataset = TensorDataset(test_data, test_target)
	if dataloader_ins:
		# Turning off shuffle simply decreases performance by ~1%
		train_dataset = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
		test_dataset = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

	return train_dataset, test_dataset, input_size, classifiers_size, test_size
