import torch
import torch.nn as nn
from dataload import DataSet
from torch.utils.data import DataLoader, random_split
from model import TextLocalizationCNN

train_dir = 'dataset/train'
test_dir = 'dataset/test'
batch_size = 3
num_epochs = 80
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


dataset_X = DataSet(train_dir)
dataset_y = DataSet(test_dir)

train_data = DataLoader(dataset_X, batch_size, shuffle=True)
test_data = DataLoader(dataset_y, batch_size, shuffle=False)

model = TextLocalizationCNN()
# model.load_state_dict(torch.load('model_cnn.pth'))
model.to(device)
model_loss = nn.MSELoss()
model_optimizer = torch.optim.RMSprop(model.parameters(), lr=0.00005)


for epoch in range(num_epochs):
	train_loss = 0
	for i, (images, labels) in enumerate(train_data):
		images = images.to(device)
		labels = labels.to(device)

		outputs = model(images)
		loss = model_loss(outputs, labels)
		del images, labels, outputs
		model_optimizer.zero_grad()
		loss.backward()
		model_optimizer.step()

		print(f"\t[Train] Epoch [{epoch+1}/{num_epochs}][{i}], Loss: {loss.item():.4f}", end='\r')
		train_loss += loss.item()

	print(f"[Train] Sum loss {train_loss}")

	model.eval()
	test_loss = 0
	for i, (images, labels) in enumerate(test_data):
		with torch.no_grad():
			images = images.to(device)
			labels = labels.to(device)
			outputs = model(images)
			loss = model_loss(outputs, labels)
			print(f"\t[Test] Epoch [{epoch+1}/{num_epochs}][{i}], Loss: {loss.item():.4f}", end='\r')
			test_loss += loss.item()

	print(f"[Test] Sum loss {test_loss}")
	model.train()


if input('sure?') == 'y':
	torch.save(model.state_dict(), 'model_cnn_booster.pth')
