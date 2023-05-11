
import model_optimized_fs


def test_model():
	"""
		Test the correctness of the model that trained using selected features.
	"""

	expected_accuracy = 0.84

	model_optimized_fs.main(model_optimized_fs.selected_features)
	accuracy = model_optimized_fs.eval_mode()
	assert accuracy > expected_accuracy, f"Got '{accuracy}', wanted an accuracy bigger than {expected_accuracy}"
