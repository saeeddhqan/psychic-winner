import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from dataload import DataSet
from torch.utils.data import DataLoader, random_split
from model import TextLocalizationCNN
import cv2

test_dir = 'dataset/test'
batch_size = 1
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

def create_grayscale_image(name, tensor):
	# Convert tensor to numpy array
	tensor_np = tensor.cpu().numpy()
	# Remove the channel 
	tensor_np = np.squeeze(tensor_np, axis=0)
	# Normalize tensor values to the range [0, 255]
	tensor_np = (tensor_np - np.min(tensor_np)) * (255 / (np.max(tensor_np) - np.min(tensor_np)))
	# Convert to 8-bit unsigned integer
	tensor_np = tensor_np.astype(np.uint8)
	# Save the grayscale image to a file
	output_file = f"results/{name}.jpg"
	plt.imsave(output_file, tensor_np, cmap='gray')
	print("Grayscale image saved to", output_file)

def contrast_enhancement(tensor):
	min_score = torch.min(tensor)
	max_score = torch.max(tensor)
	rescaled_tensor = (tensor - min_score) / (max_score - min_score)
	return rescaled_tensor

def threshold(tensor, threshold_value):
	binary_tensor = torch.zeros_like(tensor)
	binary_tensor[tensor >= threshold_value] = 1
	return binary_tensor

def connected_component_analysis(binary_tensor):
	# Convert the binary tensor to a numpy array
	binary_array = binary_tensor.numpy()
	# Apply connected component analysis using NumPy
	_, labels = cv2.connectedComponents(binary_array.astype(np.uint8))
	# Convert the labels back to a PyTorch tensor
	labels_tensor = torch.from_numpy(labels)

	return labels_tensor

dataset_y = DataSet(test_dir)

test_data = DataLoader(dataset_y, batch_size, shuffle=False)

model = TextLocalizationCNN()
model.load_state_dict(torch.load('model_cnn.pth'))
# model.to(device)
model.eval()

test_items = 20
for i, (images, labels) in enumerate(test_data):
	with torch.no_grad():
		# Move data to the device
		images = images.to(device)
		labels = labels.to(device)
		# Forward pass
		logits = model(images).cpu()
		# logits = contrast_enhancement(logits)
		logits = threshold(logits, 0.37)
		# logits = connected_component_analysis(logits)
		# print(logits.shape);exit()
		# logits /= logits.max()
		create_grayscale_image(f"{i}-tg", labels[0])
		create_grayscale_image(f"{i}-out", logits[0])
		test_items -= 1
	if test_items == 0:
		break

