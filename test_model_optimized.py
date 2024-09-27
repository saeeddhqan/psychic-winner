
import model_optimized


def test_model():
	"""
		Test the correctness of NAS model.
	"""

	expected_accuracy = 0.83

	model_optimized.main()
	accuracy = model_optimized.eval_mode()
	assert accuracy > expected_accuracy, f"Got '{accuracy}', wanted an accuracy bigger than {expected_accuracy}"
