import torch
import numpy
import random

hyperparams = {'optimizer': ['lr', 'weight_decay']}
hyperparams_bounds = {
	'lr': [1e-4, 1e-5],
	# 'momentum': [1e-1, 2e-1],
	'weight_decay': [2e-2, 1e-3],
}
ind_filename = 'individuals/ind-%03d.pth'

def mutation_polynomial_bounded(
	solution: 'The solution to mutate(list)',
	gene_min_value: 'The minimum allowed gene value(float)',
	gene_max_value: 'The maximum allowed gene value(float)', 
	max_mutation_size: float = 0.2,
	degree: float = 3):
	"""
		Mutate a solution using the Mutation Polynomial Bounded operator.
	"""
	mutated_solution = numpy.copy(solution)
	num_genes = len(solution)

	for i in range(num_genes):
		mutation_size = max_mutation_size * numpy.abs(random.uniform(-1, 1))
		gene_value = solution[i]

		# Compute the new gene value using the polynomial function
		x = numpy.abs(gene_value - gene_min_value) / (gene_max_value - gene_min_value)
		polynomial_value = (1 - x**degree)**degree
		new_gene_value = gene_value + mutation_size * numpy.sign(random.uniform(-1, 1)) * polynomial_value

		# Clamp the new gene value to the allowed range
		new_gene_value = numpy.clip(new_gene_value, gene_min_value, gene_max_value)

		mutated_solution[i] = new_gene_value

	return mutated_solution


def get_optimizer(
	model: 'Torch model'):
	"""
		Returns an optimizer with a random set hyperparam
	"""
	optimizer_class = torch.optim.RMSprop
	lr = numpy.random.choice(numpy.logspace(-5, -4, base=10))
	momentum = numpy.random.choice(numpy.linspace(0.1, .3))
	weight_decay = numpy.random.choice(numpy.linspace(2e-2, 1e-3))
	return optimizer_class(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

def exploit_and_explore(
	top_individual_path: 'Top individual file path', 
	bot_individual_path: 'Worst individual file path',
	hyperparams: 'Things we need to change',
	perturb_factors: tuple = (1.2, 0.8)):
	"""
		Copy parameters from the better model and perform mutation on hyperparams.
		One can add mutation on more parameters(dropout, etc).
		In fact, we are replacing the worst models with a mutated version of top models
	"""
	# Load the top model
	individual = torch.load(top_individual_path)
	state_dict = individual['model_state_dict'] # Parameters of the top model
	optimizer_state_dict = individual['optim_state_dict'] # top model's Optimizer hyperparameters
	batch_size = individual['batch_size']
	# Mutate hyperparams of optimizer
	for hyperparam_name in hyperparams['optimizer']:
		perturb = numpy.random.choice(perturb_factors)
		for param_group in optimizer_state_dict['param_groups']:
			param_group[hyperparam_name] *= perturb
	# Mutate batch_size
	# if hyperparams['batch_size']:
	# 	perturb = numpy.random.choice(perturb_factors)
	# 	batch_size = int(numpy.ceil(perturb * batch_size))

	individual = dict(model_state_dict=state_dict,
					  optim_state_dict=optimizer_state_dict,
					  batch_size=batch_size)
	torch.save(individual, bot_individual_path)


def usual_suspects(
	usual_cortex: 'A list of moderate performed models',
	hyperparams: 'Things we need to change',
	hyperparams_bounds: 'Change bounds for hyperparameters'):
	"""
		It receives a list of moderate performed models and then mutate their parameters
	"""
	for usual_ind in usual_cortex:
		individual = torch.load(usual_ind.filename)
		state_dict = individual['model_state_dict']
		optimizer_state_dict = individual['optim_state_dict']
		batch_size = individual['batch_size']
		# Mutate hyperparams of optimizer
		for hyperparam_name in hyperparams['optimizer']:
			bounds = hyperparams_bounds[hyperparam_name]
			for name in optimizer_state_dict['param_groups'][0]:
				if hyperparam_name == name:
					real_value = optimizer_state_dict['param_groups'][0][name]
					mutated_value = mutation_polynomial_bounded([real_value], bounds[0], bounds[1])
					optimizer_state_dict['param_groups'][0][name] = mutated_value[0]
		individual = dict(model_state_dict=state_dict,
					  optim_state_dict=optimizer_state_dict,
					  batch_size=batch_size)
		torch.save(individual, usual_ind.filename)

