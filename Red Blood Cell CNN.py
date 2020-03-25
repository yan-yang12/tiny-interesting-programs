import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from tqdm import tqdm

label = 'cell_images/'
imgs_dir = os.listdir(label)

data = []
for x in tqdm(imgs_dir):
  img = cv2.imread("cell_images/" + x,cv2.IMREAD_GRAYSCALE)
  if 'Unpocked' in x:
    data.append([img,np.array([1., 0.])])
  if 'Pocked' in x:
    data.append([img,np.array([0., 1.])])

# train-test split
np.random.shuffle(data)
train = data[:int(0.8*len(data))]
test = data[int(0.8*len(data)):]


class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(1, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)


  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x


cnn = CNN()
optimizer = optim.Adam(cnn.parameters(),lr=0.001)
criterion = nn.MSELoss()
batch_loss_lst=[]
batches = torch.utils.data.DataLoader(train,batch_size=555)
epochs = 50
for epoch in tqdm(range(epochs)):
  for x_555,y_555 in batches:
    x = x_555.view(-1,1,200,200)
    x = x.float()
    optimizer.zero_grad()
    output = cnn.forward(x)
    output = output.double()
    y_555 = y_555.double()
    batch_loss = criterion(output,y_555)
    batch_loss.backward()
    optimizer.step()
  batch_loss_lst.append(batch_loss)

#test
loss = []
for i in range(len(batch_loss_lst)):
  loss.append(batch_loss_lst[i].item())

plt.plot(range(epochs),loss)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("CNN loss vs. epoch")
plt.show()
testloader = torch.utils.data.DataLoader(test, batch_size=2, shuffle=False)
TN = 0
TP = 0
FN = 0
FP = 0
accuracy = 0
case = 0
with torch.no_grad():
  for x, y in testloader:
    for i in range(len(x)):
      case+=1
      new_x = x[i].view(-1,1,200,200)
      new_x = new_x.float()
      output = cnn.forward(new_x,flag=False)
      pred = int(torch.argmax(output))
      true = int(torch.argmax(y[i]))
      if pred == 0:
        if true == 0:
          TN+=1
          accuracy+=1
        if true == 1:
          FN+=1
      if pred == 1:
        if true == 0:
          FP+=1
        if true == 1:
          TP+=1
          accuracy+=1
confusion_matrix = {'TN':TN,'FP':FP,'FN':FN,'TP':TP}
accuracy = accuracy/case
print(case)
print(confusion_matrix)
print("Accuracy is: " + str(accuracy))