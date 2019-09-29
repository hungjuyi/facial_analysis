import sys
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader
import torch.optim as optim

from model import SimpleNet, ResNet
from ptosis_loader import PtosisLoader

args = {'num_epoch': 20}

#https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

def main():

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)

  transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  #trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
  trainset = PtosisLoader('train', transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
  #testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
  testset = PtosisLoader('val', transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
  #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  classes = [0, 1]

  #net = SimpleNet().to(device)
  net = ResNet(18, len(classes)).to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

  for epoch in range(args['num_epoch']):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
      inputs, labels = data[0].to(device), data[1].to(device)
      optimizer.zero_grad()
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      #if i % 2000 == 1999:
      if i % 10 == 0:
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss / 2000))
        running_loss = 0.0
  
  print('Finished Training')

  correct = 0
  total = 0
  with torch.no_grad():
    for data in trainloader:
      images, labels = data[0].to(device), data[1].to(device)
      outputs = net(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

    print('Training accuracy of the network on the 10000 test images: %d %%' % (
          100 * correct / total))

  correct = 0
  total = 0
  with torch.no_grad():
    for data in testloader:
      images, labels = data[0].to(device), data[1].to(device)
      outputs = net(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

    print('Testing accuracy of the network on the 10000 test images: %d %%' % (
          100 * correct / total))


if __name__ == '__main__':
    main()
