
import os
import shutil
import pandas
import torch
import pathlib

from utils import util, torch_model_arch, pbt_util, pbt_trainer


def test_trainer_train():
	"""
		Testing the class methods and overall functionality of the class (training accuracy)
	"""

	data = pandas.read_csv(util.dataset_filename)
	data.drop(['customerID'], axis=1, inplace=True)
	data = util.integrating_numerical_column('TotalCharges', data)
	data[util.target_column] = data[util.target_column].replace(['Yes', 'No'], [1, 0])
	data = util.standard_rescaling(util.numerical_columns, data)
	data = util.one_hot_encoding(util.categorized_columns, data)

	batch_size = 5
	test_prob = 0.1
	dropout_prob = 0.05
	train_epoch = 4

	act_functions = [torch.nn.ReLU, torch.nn.ReLU]

	train_loader, test_loader, input_size, \
		classifiers_size, test_size = util.data_splitter_tensor_binary(data, util.target_column, batch_size, test_prob, dataloader_ins=False)

	pbt_util.dirname = 'test_individual'
	# Create a test dir for saving the model and remove it if it exists
	if os.path.exists(pbt_util.dirname):
		shutil.rmtree(pbt_util.dirname)

	pathlib.Path(pbt_util.dirname).mkdir(exist_ok=True)
	filepath = f"{pbt_util.dirname}/test.pth"
	model = torch_model_arch.net_search(input_size, classifiers_size, dropout_prob, act_functions)
	model.to(util.device)
	model_optimizer = pbt_util.get_optimizer(model)
	model_batch_size = pbt_util.get_batch_size()
	model_loss = torch.nn.CrossEntropyLoss()
	trainer_obj = pbt_trainer.trainer(model=model,
						model_optimizer=model_optimizer,
						model_loss=model_loss,
						train_data=train_loader,
						test_data=test_loader,
						batch_size=model_batch_size,
						train_epoch=train_epoch,
						filename=filepath)
	expected_task_id = -1
	expected_set_score = -1

	trainer_obj.set_id(-1)
	trainer_obj.set_score(-1)
	# Test set_id and set_score methods
	assert trainer_obj.task_id == expected_task_id, f"Got '{trainer_obj.task_id}', wanted '{expected_task_id}'"
	assert trainer_obj.task_score == expected_set_score, f"Got '{trainer_obj.task_score}', wanted '{expected_set_score}'"

	trainer_obj.train()
	trainer_obj.eval()
	# Test if the model can learn anything or not
	assert trainer_obj.task_score > 0.70, f"Got '{trainer_obj.task_score}', wanted an accuracy bigger than 0.70"
	got_score = trainer_obj.task_score
	trainer_obj.save()

	# Test if the model has saved or not
	assert os.path.isfile(filepath), f"Expected a saved model in {filepath}."

	trainer_obj.load()

	trainer_obj.eval()
	# Test if load method can actually work
	assert trainer_obj.task_score == got_score, f"Got '{trainer_obj.task_score}', wanted {got_score}"

	# Delete the test directory
	shutil.rmtree(pbt_util.dirname)
