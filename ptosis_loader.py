import os

import numpy as np
import torch
from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms

class PtosisLoader(data.Dataset):
  def __init__(self, mode, transform):
    self.mode = mode
    self.transform = transform

    root = '/home/eddy/health/data/ptosis_with_without_cut/'
    pos_path = 'with_ptosis'
    neg_path = 'without_ptosis'
    train_pos_list_files = 'train_with_ptosis_50.lst'
    train_neg_list_files = 'train_without_ptosis_50.lst'
    val_pos_list_files = 'val_with_ptosis_50.lst'
    val_neg_list_files = 'val_without_ptosis_50.lst'

    if mode == 'train':
      pos_list_files = train_pos_list_files
      neg_list_files = train_neg_list_files
    else:
      pos_list_files = val_pos_list_files
      neg_list_files = val_neg_list_files

    records = []

    in_file = open(os.path.join(root, pos_list_files), 'r')
    for line in in_file.readlines():
      line = line.strip()
      records.append((os.path.join(root, pos_path, 'small', line), 1))
    in_file.close()
    in_file = open(os.path.join(root, neg_list_files), 'r')
    for line in in_file.readlines():
      line = line.strip()
      records.append((os.path.join(root, neg_path, 'small', line), 0))
    in_file.close()

    #print(records)
    self.records = records


  def __getitem__(self, index):
    input_path, label = self.records[index]
    input_img = Image.open(input_path).convert('RGB')
    input_data = self.transform(input_img)
    #print(input_data.shape)
    return input_data, label

  def __len__(self):
    return len(self.records)

if __name__ == '__main__':
  transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  train_set = PtosisLoader('train', transform)
  print(train_set.__len__())
  print(train_set.__getitem__(0))
  print(train_set.__getitem__(1))
