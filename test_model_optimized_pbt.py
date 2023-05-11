
import model_optimized_pbt


def test_model_optimized_pbt():
	"""
		Test the correctness of PBT model.
	"""

	expected_accuracy = 0.835

	accuracy = model_optimized_pbt.eval_mode()
	assert accuracy > expected_accuracy, f"Got '{accuracy}', wanted an accuracy bigger than {expected_accuracy}"
