import numpy
import pandas
import random
import torch

seed = 123

torch.manual_seed(seed)
random.seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


def integrating_numerical_column(data, column):
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
