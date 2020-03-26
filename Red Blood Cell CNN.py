import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# -----------------------data preparation---------------------------
data = []
for filename in os.listdir('cell_images/'):
    img = cv2.imread("cell_images/" + filename, cv2.IMREAD_GRAYSCALE)

    # crop image
    img = img[20:180, 20:180]

    # remove background (not optimal, but most efficient)
    img[np.where((img >= 126) & (img <= 135))] = 0

    # normalize the data to [0, 1]
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # label the image
    if 'Pocked' in filename:
        data.append([img, 1.])
    else:
        data.append([img, 0.])

# split the data
np.random.shuffle(data)
train = data[0:int(0.8 * len(data))]
test = data[int(0.8 * len(data)):]


# ------------------------neural network----------------------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(2)
        self.conv2 = nn.Conv2d(2, 4, 3)
        self.conv2_bn = nn.BatchNorm2d(4)
        self.conv3 = nn.Conv2d(4, 4, 7)
        self.conv3_bn = nn.BatchNorm2d(4)

        self.fc1 = nn.Linear(256, 200)
        self.dropout = nn.Dropout(p=0.7)
        self.fc2 = nn.Linear(200, 15)
        self.fc3 = nn.Linear(15, 1)

        torch.nn.init.xavier_uniform_(self.conv1.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.conv2.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.conv3.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.fc2.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.fc3.weight, gain=1.0)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1_bn(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_bn(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_bn(self.conv3(x)), 4))
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


# ------------------------main program----------------------------
acc_list = []
for sim in range(100):
    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    batches = torch.utils.data.DataLoader(train, batch_size=222)
    epochs = 30
    for epoch in range(epochs):
        for x_batch, y_batch in batches:
            x = x_batch.view(-1, 1, 160, 160).float()
            optimizer.zero_grad()
            output = model.forward(x).double()
            y_batch = y_batch.double().view(-1, 1)
            batch_loss = criterion(output, y_batch)
            batch_loss.backward()
            optimizer.step()
        # print(f'epoch: {epoch}, loss: {batch_loss.item():.4f}')

    # calculate the confusion matrix and accuracy
    testloader = torch.utils.data.DataLoader(test)
    TN = TP = FN = FP = accuracy = 0
    with torch.no_grad():
        for x, y in testloader:
            for i in range(len(x)):
                elt = x[i].view(-1, 1, 160, 160).float()
                output = model(elt)
                pred = int(output >= 0.5)
                TP += int((pred == 1) & bool(y[i] == 1))
                FP += int((pred == 1) & bool(y[i] == 0))
                TN += int((pred == 0) & bool(y[i] == 0))
                FN += int((pred == 0) & bool(y[i] == 1))

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    print(f'TP:{TP}, FP:{FP}, TN:{TN}, FN:{FN}')
    print(f'Accuracy is: {accuracy:.4f}')
    acc_list.append(accuracy)

plt.hist(acc_list)
plt.title("Accuracy for multiple trainings")
plt.xlabel("Accuracy")
plt.ylabel('number')
plt.show()
arr = np.array(acc_list)
print(np.percentile(arr, 50))