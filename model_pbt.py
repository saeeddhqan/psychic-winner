import torch
import numpy
import pathlib
import pandas
import random

import matplotlib.pyplot as plt

from util import util, torch_model_arch, pbt_util, pbt_trainer

data = pandas.read_csv(util.dataset_filename)
data.drop(['customerID'], axis=1, inplace=True)
data = util.integrating_numerical_column('TotalCharges', data)
data[util.target_column] = data[util.target_column].replace(['Yes', 'No'], [1, 0])
data = util.standard_rescaling(util.numerical_columns, data) # My preference
data = util.one_hot_encoding(util.categorized_columns, data)

batch_size = 5
test_prob = 0.1
dropout_prob = 0.05195051965108121
train_epoch = 3

generation_size = 12
population_size = 30
early_stopping = 0.831
act_functions = [torch.nn.Tanh, torch.nn.ReLU]

train_loader, test_loader, input_size, \
	classifiers_size, test_size = util.data_splitter_tensor_binary(data, util.target_column, batch_size, test_prob, dataloader_ins=False)


pathlib.Path('individuals').mkdir(exist_ok=False)
population = []

# Initializing population
for ind in range(population_size):
	model = torch_model_arch.net_search(input_size, classifiers_size, dropout_prob, act_functions)
	model.to(util.device)
	model_optimizer = pbt_util.get_optimizer(model)
	model_loss = torch.nn.CrossEntropyLoss()
	trainer_obj = pbt_trainer.trainer(model=model,
						   model_optimizer=model_optimizer,
						   model_loss=model_loss,
						   train_data=train_loader,
						   test_data=test_loader,
						   batch_size=batch_size,
						   train_epoch=train_epoch,
						   filename=pbt_util.ind_filename % ind)
	trainer_obj.set_id(ind)
	trainer_obj.set_score(0)
	population.append(trainer_obj)

# For plotting
max_fitness_values = []
mean_fitness_values = []

for generation in range(generation_size):
	pop_len = len(population)
	score_sum = 0
	for ind in range(pop_len):
		model = population[ind]
		# Do not train the top models anymore
		if model.task_score < early_stopping:
			model.load()
			model.train()
			model.eval()
			model.save()
		score_sum += model.task_score
		print(model.task_score)
	# Sort population based on the scores(model accuracy)
	tasks = sorted(population, key=lambda x: x.task_score, reverse=True)
	avg = score_sum/pop_len
	mx = tasks[0].task_score
	max_fitness_values.append(avg)
	mean_fitness_values.append(mx)
	print('avg:', avg, ', ', 'max:', f"{mx}, id {tasks[0].task_id}")
	
	# Choose the top 20 percent and the worst 20 percent of individuals in population
	fraction = 0.2
	cutoff = int(numpy.ceil(fraction * len(tasks)))
	tops = tasks[:cutoff]
	bottoms = tasks[len(tasks) - cutoff:]
	elitism = tasks[:3]
	usual_cortex = tasks[3:len(tasks) - cutoff]
	
	# We pick a model randomly from top models and replace the bottom(worst) models with top ones.
	# But before doing so, we just mutate the parameters. That's all.
	for bottom in bottoms:
		top = numpy.random.choice(tops)
		top_individual_path = top.filename
		bot_individual_path = bottom.filename
		pbt_util.exploit_and_explore(top_individual_path, bot_individual_path, pbt_util.hyperparams)

	# Mutate usual_cortex(individuals that are neither at the top nor at the bottom)
	pbt_util.usual_suspects(usual_cortex, pbt_util.hyperparams, pbt_util.hyperparams_bounds)


plt.plot(max_fitness_values, color='red', label='gen max')
plt.plot(mean_fitness_values, color='green', label='gen mean')
plt.xlabel('Generation')
plt.ylabel('Max / Average Fitness')
plt.legend()
plt.title('Max and Average fitness over Generations')
plt.show()
