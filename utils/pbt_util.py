
import torch
import numpy
import random
from utils import util

"""
	Utilities for Population Based Training.
"""

hyperparams = {'optimizer': ['lr', 'weight_decay']}
hyperparams_bounds = {
	'lr': [1e-5, 1e-4],
	# 'momentum': [1e-1, 2e-1],
	'weight_decay': [2e-2, 1e-3],
	# 'batch_size': [3, 7],
}

dirname = 'individuals'
ind_filename = dirname + '/ind-%03d.pth'


def smart_mut_polynomial_bounded(
	gene: 'Gene to be mutated', 
	low: 'The minimum allowed gene value',
	up: 'The maximum allowed gene value',
	all_genes: 'The gene values of all individuals',
	all_scores: 'The gene scores of all individuals',
	eta: 'Crowding degree of the mutation. Higher eta will produce a mutant resembling its parent', 
	indpb: 'The probability of occurring mutation'):
	"""
		SMPB is a variation of polynomial mutation as implemented in original NSGA-II algorithm in
		C by Deb that calculates the direction of mutating with bias. It rolls out the directional 
		bias towards individuals with higher fitness value, which means the moderated individuals go toward 
		top ones. I got the basic code from DEAP repo.
	"""

	gene_value, gene_score = gene
	if random.random() > indpb:
		return gene_value

	x = gene_value
	rand = random.random()

	treshold = 0.5

	joined = zip(all_genes, all_scores)
	upper = [x for x in joined if x[1] > gene_score]

	if len(upper) > 0:  # If the gene is the top gene, the condition will be false
		upper_mean = sum([x[0] for x in upper]) / len(upper)
		treshold = 1.0 - len(upper) / len(all_genes)  # It has a bias towards top genes with less frequency. More diversity.
		sign = numpy.sign(upper_mean - gene_value)
		if sign == 1:
			treshold = 1.0 - treshold

	# deviation = abs(upper_mean - x)
	seed = random.random()
	if rand < treshold:  # Positive
		delta = seed * abs(x - up)
		x += delta * eta
	else:  # Negative
		delta = seed * abs(x - low)
		x -= delta * eta

	return x


def get_optimizer(
	model: 'Torch model'):
	"""
		Returns an optimizer with a random set of hyperparameters.
	"""
	optimizer_class = torch.optim.RMSprop
	lr = numpy.random.choice(numpy.logspace(-5, -4, base=10))
	# momentum = numpy.random.choice(numpy.linspace(0.1, .3))
	weight_decay = numpy.random.choice(numpy.linspace(2e-2, 1e-3))
	return optimizer_class(model.parameters(), lr=lr, momentum=0.07, weight_decay=weight_decay)


def get_batch_size():
	"""
		Returns a random batch size. For now, we want to make the space search
		limited. That's why we return an arbitrary number.
	"""

	# return random.randint(hyperparams_bounds['batch_size'][0], hyperparams_bounds['batch_size'][1])
	return 5


def exploit_and_explore(
	top_individual_path: 'Top individual file path', 
	bot_individual_path: 'Worst individual file path',
	hyperparams: 'Things we need to change',
	perturb_factors: tuple = (1.2, 0.8)):
	"""
		Copy parameters from the better model and perform mutation on hyperparams.
		One can add mutation on more parameters(dropout, etc).
		In fact, we are replacing the worst models with a mutated version of top models.
	"""

	# Load the top model
	individual = torch.load(top_individual_path)
	state_dict = individual['model_state_dict']  # Parameters of the top model
	optimizer_state_dict = individual['optim_state_dict']  # Top model's optimizer hyperparameters
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
	population: 'Population(all models)',
	cutoff: 'Removing worst and top cases point',
	elitism: 'Elitism size',
	hyperparams: 'Things we need to change',
	hyperparams_bounds: 'Change bounds for hyperparameters'):
	"""
		It mutates moderate performed models with SMPB.
	"""

	all_scores = [x.task_score for x in population]
	all_genes = {}
	loaded = {x.task_id: torch.load(x.filename) for x in population}
	start = 0
	finish = len(population)

	# The idea behind running this loop twice is: filling genes(all_genes) for the first iteration
	# and the second for mutating genes.
	for tire in range(2):
		if tire == 1:
			start = cutoff
			finish = len(population) - cutoff

		for usual_ind in population[start:finish]:
			individual = loaded[usual_ind.task_id]
			score = usual_ind.task_score
			state_dict = individual['model_state_dict']
			optimizer_state_dict = individual['optim_state_dict']
			batch_size = individual['batch_size']
			params = optimizer_state_dict['param_groups'][0]
			# We do not mutate hyperparameters if the model is not trainable(meaning the model performs well enough).
			if not usual_ind.trainable and tire == 1:
				continue

			for hyparam_name in hyperparams['optimizer']:
				for name in params:
					if hyparam_name != name:
						continue
					real_value = params[name]
					if tire == 0:
						if name in all_genes:
							all_genes[name].append(real_value)
						else:
							all_genes[name] = [real_value]
					else:
						# Mutate hyperparams of optimizer
						bounds = hyperparams_bounds[hyparam_name]
						mutated_value = smart_mut_polynomial_bounded((real_value, score), bounds[0], bounds[1], all_genes[name], all_scores, 0.9, 1)
						params[name] = mutated_value

			if tire == 1:
				optimizer_state_dict['param_groups'][0] = params
				individual = dict(model_state_dict=state_dict,
							optim_state_dict=optimizer_state_dict,
							batch_size=batch_size)
				torch.save(individual, usual_ind.filename)
