
from utils import util

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import pandas
import numpy
from torch.utils.data import DataLoader, TensorDataset


def test_integrating_numerical_column():
	"""
		Testing integrating_numerical_column function for removing non-numerical values.
	"""
	# Create test data with a column containing numeric and string values
	data = pandas.DataFrame({'col1': ['1.0', '2.0', '3.0', '4.0', 'foo', 'bar', None]})

	expected_result = numpy.array([1.0, 2.0, 3.0, 4.0, 2.5, 2.5, 2.5])
	expected_type = numpy.float64
	# Call the function to replace string values with column mean
	result = util.integrating_numerical_column('col1', data)['col1']

	# Ensure that string values were replaced with the column's mean
	assert numpy.allclose(result.values, expected_result), f"Got '{result.values}', wanted '{expected_result}'"

	# Ensure that the column is now of a numeric dtype
	assert result.dtype == numpy.float64, f"Got '{result.dtype}', wanted '{expected_type}'"


def test_one_hot_encoding():
	"""
		Testing one_hot_encoding function for encoding dummy variables.
	"""

	# Create test data with a column containing numeric and string values
	data = pandas.DataFrame({'col1': ['Yes', 'No', 'No', 'No', 'Yes'], 'col2': ['Rarely', 'Seldom', 'Always', 'Always', 'Always']})

	# Call the function to replace string values with column mean
	result = util.one_hot_encoding(['col1', 'col2'], data)

	l1_columns = len(data['col1'].unique())
	l2_columns = len(data['col2'].unique())
	expected_l_columns = l1_columns + l2_columns

	# Expectations
	expected_no = numpy.array([0.0, 1.0, 1.0, 1.0, 0.0])
	expected_yes = numpy.array([1.0, 0.0, 0.0, 0.0, 1.0])
	expected_rarely = numpy.array([1.0, 0.0, 0.0, 0.0, 0.0])
	expected_seldom = numpy.array([0.0, 1.0, 0.0, 0.0, 0.0])
	expected_always = numpy.array([0.0, 0.0, 1.0, 1.0, 1.0])

	# Values
	no = result['col1_no'].values
	yes = result['col1_yes'].values
	rarely = result['col2_rarely'].values
	seldom = result['col2_seldom'].values
	always = result['col2_always'].values

	# Ensure that number of columns are correct
	assert len(result.columns) == expected_l_columns, f"Got '{len(result.columns)}', wanted '{expected_l_columns}'"

	# Ensure that one hot values are correct
	assert numpy.allclose(no, expected_no), f"Got '{no}', wanted '{expected_no}'"
	assert numpy.allclose(yes, expected_yes), f"Got '{yes}', wanted '{expected_yes}'"
	assert numpy.allclose(rarely, expected_rarely), f"Got '{rarely}', wanted '{expected_rarely}'"
	assert numpy.allclose(seldom, expected_seldom), f"Got '{seldom}', wanted '{expected_seldom}'"
	assert numpy.allclose(always, expected_always), f"Got '{always}', wanted '{expected_always}'"


def test_standard_rescaling():
	"""
		Testing the accuracy of standard_rescaling function
	"""

	# Create test data with two columns that need to be rescaled
	data = pandas.DataFrame({'col1': [1.0, 2.0, 3.0, 4.0], 'col2': [2.0, 4.0, 6.0, 8.0]})

	# Call the function to rescale the columns
	result = util.standard_rescaling(['col1', 'col2'], data)

	# Create expected output by manually rescaling the columns
	expected = pandas.DataFrame({'col1': [-1.34164079, -0.4472136, 0.4472136, 1.34164079], 
							'col2': [-1.34164079, -0.4472136, 0.4472136, 1.34164079]})

	# Ensure that the rescaled columns match the expected output
	assert numpy.allclose(result.values, expected.values), f"Got '{result.values}', wanted '{expected.values}'"

	# Ensure that the rescaled columns have mean 0 and standard deviation 1. Note that std might not be
	# precisely 1 and it has some errors due to the floating conversion. Therefore, we use a small deviation here(2e-1)
	assert numpy.allclose(result.mean(), 0) and numpy.allclose(result.std(), 1, atol=2e-1), f"Got '{numpy.allclose(result.mean(), 0) and numpy.allclose(result.std(), 1, atol=2e-1)}', wanted 'True'"


def test_minmax_rescaling():
	"""
		Testing the accuracy of standard_rescaling function
	"""

	# Create test data with two columns that need to be rescaled
	data = pandas.DataFrame({'col1': [1.0, 2.0, 3.0, 4.0], 'col2': [2.0, 4.0, 6.0, 8.0]})

	# Call the function to rescale the columns
	result = util.minmax_rescaling(['col1', 'col2'], data)

	# Create expected output by manually rescaling the columns
	expected = pandas.DataFrame({'col1': [0.0, 0.33333333, 0.66666667, 1.0], 
							'col2': [0.0, 0.33333333, 0.66666667, 1.0]})

	# Ensure that the rescaled columns match the expected output
	assert numpy.allclose(result.values, expected.values), f"Got '{result.values}', wanted '{expected.values}'"

	# Ensure that the rescaled columns have values between 0 and 1
	assert numpy.all(result.values >= 0) and numpy.all(result.values <= 1), f"Got '{numpy.all(result.values >= 0) and numpy.all(result.values <= 1)}', wanted 'True'"


def test_model_scoring():
	"""
		Testing the accuracy of model_scoring function
	"""

	# Create test labels and test predicted
	labels = [1, 0, 1, 1, 0]
	predicted = [1, 0, 0, 1, 0]
	# Getting results
	c_matrix, accuracy, precision, recall, f1 = util.model_scoring(predicted, labels)

	# Expected results. It's ridiculous to get scores with the way that the actual code should be :)
	expected_c_matrix = numpy.array([[2, 0], [1, 2]])
	expected_accuracy = accuracy_score(labels, predicted)
	expected_precision = precision_score(labels, predicted)
	expected_recall = recall_score(labels, predicted)
	expected_f1 = f1_score(labels, predicted)

	# Test confusion matrix
	assert numpy.array_equal(c_matrix, expected_c_matrix), f"Got '{c_matrix}', wanted '{expected_c_matrix}'"

	# Test accuracy
	assert accuracy == expected_accuracy, f"Got '{accuracy}', wanted '{expected_accuracy}'"

	# Test precision
	assert precision == expected_precision, f"Got '{precision}', wanted '{expected_precision}'"

	# Test recall
	assert recall == expected_recall, f"Got '{recall}', wanted '{expected_recall}'"

	# Test F1 score
	assert f1 == expected_f1, f"Got '{f1}', wanted '{expected_f1}'"


def test_data_splitter_tensor_binary():
	"""
		Testing the functionality of data_splitter_tensor_binary function.
	"""

	# Preparing data
	data = pandas.read_csv(util.dataset_filename)
	data.drop(['customerID'], axis=1, inplace=True)
	data = util.integrating_numerical_column('TotalCharges', data)
	data[util.target_column] = data[util.target_column].replace(['Yes', 'No'], [1, 0])
	data = util.standard_rescaling(util.numerical_columns, data)
	data = util.one_hot_encoding(util.categorized_columns, data)
	batch_size = 5
	test_prob = 0.1
	# Getting results
	train_data, test_data, input_size, classifiers_size, test_size = util.data_splitter_tensor_binary(
		data, util.target_column, batch_size, test_prob, True)

	expected_test_size = int(data.shape[0] * test_prob) + 1

	# Test types
	assert isinstance(train_data, DataLoader), f"Got '{type(train_data)}', wanted 'DataLoader' instance"
	assert isinstance(test_data, DataLoader), f"Got '{type(test_data)}', wanted 'DataLoader' instance"
	assert isinstance(input_size, int), f"Got '{type(input_size)}', wanted 'int' instance"
	assert isinstance(classifiers_size, int), f"Got '{type(classifiers_size)}', wanted 'int' instance"
	assert isinstance(test_size, int), f"Got '{type(test_size)}', wanted 'int' instance"

	# Test shapes, and sizes
	# to verify that the dataset is properly loaded into the DataLoader. 
	# Some of the data might be removed during preprocessing, etc
	tensor_size_cond = train_data.dataset.tensors[0].shape[0] == len(train_data.dataset)
	tensor_size_cond2 = train_data.dataset.tensors[0].shape[0] == len(train_data.dataset)
	assert tensor_size_cond, f"Got '{tensor_size_cond}', wanted 'True'"
	assert train_data.batch_size == batch_size, f"Got '{train_data.batch_size}', wanted '{batch_size}'"
	assert test_data.batch_size == batch_size, f"Got '{test_data.batch_size}', wanted '{batch_size}'"
	assert tensor_size_cond2, f"Got '{tensor_size_cond}', wanted 'True'"
	assert input_size == data.shape[1] - 1, f"Got '{input_size}', wanted '{data.shape[1] - 1}'"
	assert classifiers_size == 2, f"Got '{classifiers_size}', wanted '2'"
	assert test_size == expected_test_size, f"Got '{test_size}', wanted '{expected_test_size}'"

	# In this time, the function must return TensorDataset objects instead of DataLoader
	train_data, test_data, input_size, classifiers_size, test_size = util.data_splitter_tensor_binary(
		data, util.target_column, 5, 0.1, False)

	# Test types
	assert isinstance(train_data, TensorDataset), f"Got '{type(train_data)}', wanted 'TensorDataset' instance"
	assert isinstance(test_data, TensorDataset), f"Got '{type(test_data)}', wanted 'TensorDataset' instance"
