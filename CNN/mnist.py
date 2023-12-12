import time
from typing import List, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

rand_tensor = torch.rand(5,2)
simple_model = nn.Sequential(nn.Linear(2,10), nn.ReLU(), nn.Linear(10,1))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rand_tensor = rand_tensor.to(device)
simple_model = simple_model.to(device)

print(f'input is on {rand_tensor.device}')
print(f'model parameters are on {[param.device for param in simple_model.parameters()]}')
print(f'output is on {simple_model(rand_tensor).device}')

def train(model: nn.Module,
          loss_fn: nn.modules.loss._Loss,
          optimizer: torch.optim.Optimizer,
          train_loader: torch.utils.data.DataLoader,
          epoch: int=0)-> List:
    
    model.train()
    train_loss = []
    t = 0
    for batch_idx, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        if batch_idx*len(images) >= len(train_loader.dataset) * 0.1 * t:
            print(f'Epoch {epoch}: [{batch_idx*len(images)}/{len(train_loader.dataset)}] Loss: {loss.item():.3f}')
            t += 1
    assert len(train_loss) == len(train_loader)
    return train_loss

def test(model: nn.Module,
         loss_fn: nn.modules.loss._Loss,
         test_loader: torch.utils.data.DataLoader,
         epoch: int=0)-> Dict:
    
    model.eval()
    test_loss = 0
    correct = 0
    predictions = []

    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            output = model(images)
            loss = loss_fn(output, targets)
            test_loss += loss.item()
            pred = output.data.max(1, keepdim=True)[1]
            predictions = predictions + pred.tolist()
            correct += pred.eq(targets.data.view_as(pred)).sum()

    avg_test_loss = test_loss / len(test_loader)
    accuracy = correct/len(test_loader.dataset)
    test_stat = {
        "loss": avg_test_loss,
        "accuracy": accuracy,
        "prediction": torch.tensor(predictions)
    }
    print(f"Test result on epoch {epoch}: total sample: {len(test_loader.dataset)}, Avg loss: {test_stat['loss']:.3f}, Acc: {100*test_stat['accuracy']:.3f}%")
    assert "loss" and "accuracy" and "prediction" in test_stat.keys()
    assert len(test_stat["prediction"]) == len(test_loader.dataset)
    assert isinstance(test_stat["prediction"], torch.Tensor)
    return test_stat

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,),(0.3081,))])

train_dataset = torchvision.datasets.MNIST('data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST('data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

class OurFC(nn.Module):
  def __init__(self):
    super(OurFC, self).__init__()
    self.fc1 = nn.Linear(28*28, 128)
    self.relu1 = nn.ReLU()
    self.fc2 = nn.Linear(128, 128)
    self.relu2 = nn.ReLU()
    self.fc3 = nn.Linear(128, 10)

  def forward(self, x):
    x = x.view(x.size(0), -1)
    x = self.fc1(x)
    x = self.relu1(x)
    x = self.fc2(x)
    x = self.relu2(x)
    x = self.fc3(x)
    return x

class OurCNN(nn.Module):
  def __init__(self):
    super(OurCNN, self).__init__()
    self.conv_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1) # 64x1x28x28 -> 64x16x28x28
    self.conv_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1) # 64x16x14x14 -> 64x32x14x14
    self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2) #  64x16x28x28 -> 64x16x14x14 / 64x32x14x14 -> 64x32x7x7
    self.linear_1 = nn.Linear(7 * 7 * 32, 64) # 64x1568 -> 64x64
    self.linear_2 = nn.Linear(64, 10) # 64x64 -> 64x10
    self.dropout = nn.Dropout(p=0.5)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.conv_1(x)
    x = self.relu(x)
    x = self.max_pool2d(x)
    x = self.conv_2(x)
    x = self.relu(x)
    x = self.max_pool2d(x)
    x = x.reshape(x.size(0), -1)
    x = self.linear_1(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.linear_2(x)

    return x
  
criterion = nn.CrossEntropyLoss()

start = time.time()
max_epoch = 2
model = OurFC().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.8)

for epoch in range(1, max_epoch+1):
    train_losses = train(model, criterion, optimizer, train_loader, epoch)
    test_stat = test(model, criterion, test_loader, epoch)
end = time.time()
print(f'Finished Training after {end-start} s ')

start = time.time()
model = OurCNN().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.8)

for epoch in range(1, max_epoch+1):
    train_losses = train(model, criterion, optimizer, train_loader, epoch)
    test_stat = test(model, criterion, test_loader, max_epoch)
end = time.time()
print(f'Finished Training after {end-start} s ')

ourfc = OurFC()
total_params = sum(p.numel() for p in ourfc.parameters())
print(f'OurFC has a total of {total_params} parameters')

ourcnn = OurCNN()
total_params = sum(p.numel() for p in ourcnn.parameters())
print(f'OurCNN has a total of {total_params} parameters')