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

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=transform)

batch_size = 9

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

batch_idx, (images, targets) = next(enumerate(train_loader))
fig, ax = plt.subplots(3,3,figsize = (9,9))
for i in range(3):
    for j in range(3):
        image = images[i*3+j].permute(1,2,0)
        image = image/2 + 0.5
        ax[i,j].imshow(image)
        ax[i,j].set_axis_off()
        ax[i,j].set_title(f'{classes[targets[i*3+j]]}')
plt.show()

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1) # 9x3x32x32 -> 9x16x32x32
    self.conv_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1) # 9x16x16x16 -> 9x32x16x16
    self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2) # 9x16x32x32 -> 9x16x16x16 / 9x32x16x16 -> 9x32x8x8
    self.linear_1 = nn.Linear(8 * 8 * 32, 64) # 9x2048 -> 9x64
    self.linear_2 = nn.Linear(64, 10) # 9x64 -> 9x10
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
  
start = time.time()
max_epoch = 2
net = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
for epoch in range(1, max_epoch + 1):  
    train(net, criterion, optimizer, train_loader, epoch)
output = test(net, criterion, test_loader, epoch)
end = time.time()
print(f'Finished Training after {end-start} s ')

total_images = 3
predictions = output['prediction']
targets = torch.tensor(testset.targets)
fig, ax = plt.subplots(total_images,figsize = (5,3*total_images))
j = 0
for idx, (images, targets) in enumerate(test_loader):
    if predictions[idx] != targets[idx]:
        image = images[idx].permute(1,2,0)
        image = image/2 + 0.5
        ax[j].imshow(image)
        ax[j].set_axis_off()
        ax[j].set_title(f'Ground Truth: {classes[targets[idx]]} / Prediction: {classes[output["prediction"][idx]]}')
        j += 1
    if j == total_images:
        break
plt.show()

######################## Resnet ###############################

resnet18 = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)
resnet18 = resnet18.to(device)

transform = torchvision.transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

trainset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=transform)

batch_size = 128

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

resnet18.fc = nn.Linear(resnet18.fc.in_features, 10)
resnet18.fc.to(device)

criterion = nn.CrossEntropyLoss()
output = test(resnet18, criterion, test_loader)

start = time.time()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet18.parameters(), lr=0.001, momentum=0.9)
max_epoch = 2
for epoch in range(1, max_epoch + 1):
    train(resnet18, criterion, optimizer, train_loader, epoch)

test(resnet18, criterion, test_loader, epoch)
end = time.time()
print(f'Finished Training after {end-start} s ')

resnet18 = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)
resnet18 = resnet18.to(device)
start = time.time()
for name, param in resnet18.named_parameters():
    if 'layer4' in name or 'fc' in name: # only update layer 4, and linear
        param.requires_grad = True
    else:
        param.requires_grad = False
resnet18.fc = nn.Linear(resnet18.fc.in_features, 10)
resnet18.fc.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet18.parameters(), lr=0.001, momentum=0.9)
max_epoch = 2
for epoch in range(1, max_epoch + 1):
    train(resnet18, criterion, optimizer, train_loader, epoch)
test(resnet18, criterion, test_loader)
end = time.time()
print(f'Finished Training after {end-start} s ')