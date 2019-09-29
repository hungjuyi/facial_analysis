import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

class SimpleNet(nn.Module):
  def __init__(self):
    super(SimpleNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
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

#https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
class ResNet(nn.Module):
  def __init__(self, resnet_num_layers, num_classes):
    super(ResNet, self).__init__()
    if resnet_num_layers == 18:
      model_ft = models.resnet18(pretrained=True)
    elif resnet_num_layers == 34:
      model_ft = models.resnet34(pretrained=True)
    elif resnet_num_layers == 50:
      model_ft = models.resnet50(pretrained=True)
    elif resnet_num_layers == 101:
      model_ft = models.resnet101(pretrained=True)
    elif resnet_num_layers == 152:
      model_ft = models.resnet152(pretrained=True)
    else:
      assert(False)

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    self.model = model_ft

  def forward(self, x):
    return self.model(x)
