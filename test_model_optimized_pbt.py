
import model_optimized_pbt


def test_model_optimized_pbt():
	"""
		Test the correctness of the produced PBT model.
	"""

	expected_accuracy = 0.825
	expected_precision = 0.665

	accuracy, precision = model_optimized_pbt.eval_mode()
	assert accuracy > expected_accuracy, f"Got '{accuracy}', wanted an accuracy bigger than {expected_accuracy}"
	assert precision > expected_precision, f"Got '{precision}', wanted an precision bigger than {expected_precision}"
