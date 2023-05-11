
import model


def test_model():
	"""
		Test the correctness of simple model.
	"""
	expected_accuracy = 0.828

	accuracy = model.main()
	assert accuracy > expected_accuracy, f"Got '{accuracy}', wanted an accuracy bigger than {expected_accuracy}"
