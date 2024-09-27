
import model


def test_model():
	"""
		Test the correctness of simple model.
	"""
	expected_accuracy = 0.82
	expected_precision = 0.73

	accuracy, precision = model.main()

	assert accuracy > expected_accuracy, f"Got '{accuracy}', wanted an accuracy bigger than {expected_accuracy}"
	assert precision > expected_precision, f"Got '{precision}', wanted an precision bigger than {expected_precision}"
