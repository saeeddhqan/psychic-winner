from os import path, scandir

import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy

DEFAULT_RESOLUTION = [448, 224]
"""
	ColorJitter: Introducing random changes to its brightness, contrast, saturation, and hue
	RandomGrayscale: This transformation randomly converts the image to grayscale with a certain probability
	transforms.ToTensor: This transformation converts the image into a tensor
"""
DEFAULT_TRANSFORM = transforms.Compose(
	[
		transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
		transforms.RandomGrayscale(),
		transforms.Grayscale(),
		transforms.ToTensor(),
	]
)


class DataSet(Dataset):
	def __init__(
		self,
		data_dir,
		resolution=DEFAULT_RESOLUTION,
		transform=DEFAULT_TRANSFORM,
		channel=True,
	):
		super().__init__()

		self.img_files = list(sorted([x for x in scandir(data_dir) if x.name.endswith('.jpg')], key=lambda f: f.name))
		self.box_files = list(sorted([x for x in scandir(data_dir) if x.name.endswith('.txt')], key=lambda f: f.name))
		self.resolution = resolution
		self.transform = transform
		self.channel = channel

	def __getitem__(self, idx):
		# process image tensor
		img = Image.open(self.img_files[idx].path).convert("RGB")
		w = img.width
		h = img.height

		# remember these scaling ratios to scale truth boxes later
		w_scaling = self.resolution[1] / w
		h_scaling = self.resolution[0] / h
		img = transforms.functional.resize(img, self.resolution)
		img = self.transform(img)
		img /= 255
		img = (img / 255 * 0.999) + 0.001
		# target 1: text/non-text classes
		# the elements are {0: non-text, 1: text}
		target = torch.zeros(1, *self.resolution, dtype=torch.float32)

		# process target tensors
		with open(self.box_files[idx], encoding='utf-8', mode='r', errors='ignore') as fp:
			for line in fp:
				# Cool
				linesplit = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',', maxsplit=8)
				if len(linesplit) < 2:
					continue
				coordinates = [int(x) for x in linesplit[:6]]
				box = [
					int(coordinates[0] * w_scaling),
					int(coordinates[1] * h_scaling),
					int(coordinates[4] * w_scaling),
					int(coordinates[5] * h_scaling),
				]
				target[0, box[1]:box[3], box[0]:box[2]] = 1.0
		
		if not self.channel:
			return img.squeeze(0).flatten(), target.squeeze(0).flatten()
		
		return img, target


	def __len__(self):
		return len(self.img_files)


if __name__ == "__main__":
	import random
	import os
	from torch.nn import functional

	filepath = 'n10_pretrained2.pth'

	dataset = DataSet('dataset/train', channel=False)
	for i in range(1):
		x, data = dataset[i]
		print(x.shape, data.shape)
