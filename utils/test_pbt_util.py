
from utils import pbt_util


def test_smart_mut_polynomial_bounded():
	"""
		Testing the functionality and accuracy of smart_mut_polynomial_bounded. 
		Testing is tricky and sensitive since the output is somehow random.
	"""

	all_genes = [1.59e-05, 2.12e-05, 2.68e-05, 1.59e-05, 1.45e-05]
	all_scores = [0.82, 0.825, 0.83, 0.817, 0.815]
	# Check if the output is within the range of low and up
	gene = (10, 80)
	low = 0
	up = 20
	eta = 0.5
	indpb = 0.9
	mutated = pbt_util.smart_mut_polynomial_bounded(gene, low, up, all_genes, all_scores, eta, indpb)
	assert low <= mutated <= up, f"Mutated value {mutated} is not within the range of {low} and {up}."

	# Check if the output is the same as the input when indpb is 0
	indpb = 0.0
	mutated = pbt_util.smart_mut_polynomial_bounded(gene, low, up, all_genes, all_scores, eta, indpb)
	assert mutated == gene[0], f"Got 'mutated', wanted '{gene[0]}'. Mutated value should be the same as the input when indpb is 0."

	# Check if the output is similar to the input when eta is close to 0
	eta = 0.001
	indpb = 0.9
	mutated = pbt_util.smart_mut_polynomial_bounded(gene, low, up, all_genes, all_scores, eta, indpb)
	assert abs(mutated - gene[0]) < 0.1, 'Mutated value should be similar to the input when eta is close to 0.'

	# Check the direction of mutation. The direction should be positive, which means the mutated value
	# should be bigger than the input value
	gene = (1.44e-05, 0.814)
	low = 1e-6
	up = 1e-4
	eta = 0.5
	mutated = pbt_util.smart_mut_polynomial_bounded(gene, low, up, all_genes, all_scores, eta, indpb)
	assert mutated > gene[0], 'Mutated value should be bigger than gene when bigger genes got better scores.'


def test_get_batch_size():
	"""
		Testing get_batch_size function
	"""

	out = pbt_util.get_batch_size()
	assert isinstance(out, int), f"Got {type(out)}, wanted 'int' instance"
