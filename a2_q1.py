
# !pip install wget

import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import random
import wget
import wandb
import os
import datetime
from zipfile import ZipFile
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
import argparse
import warnings
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

font = {'size'   : 6}

matplotlib.rc('font', **font)


url='https://storage.googleapis.com/wandb_datasets/nature_12K.zip'
filename = os.path.basename(url)

if not os.path.exists(filename) and not os.path.exists("inaturalist_12K"):
  filename = wget.download(url)
  with ZipFile(filename, 'r') as z:
    print('Extracting files...')
    z.extractall()
    print('Done!')
  os.remove(filename)

classes = sorted([name for name in os.listdir("inaturalist_12K/train") if name != '.DS_Store'])
print(classes)

image_size = (256,256)
num_layers = 5
num_classes = len(classes)

#default config
config = {
    "size_filters" : [7,5,5,3,3],
    "activation" : 'ReLU',
    "learning_rate": 1e-3,
    "filters_org": 1,
    "num_filters" : 64,
    "dense_layer_size" : 256,
    "batch_norm": True,
    "data_augment": True,
    "dropout":0.2,
    "batch_size":32,
    "epochs": 10
    }



# print(list_filters)

def data_prep(config = config):



def imshow(img, title):
  npimg = img.numpy()/2 + 0.5
  plt.figure(figsize = (batch_size*3, 3))
  plt.axis('off')
  plt.imshow(np.transpose(npimg, (1,2,0)))
  plt.title(title)
  plt.show()

def show_batch_images(dataloader):
  images, labels = next(iter(dataloader))
  img = torchvision.utils.make_grid(images)
  imshow(img, title = [classes[x.item()] for x in labels])
  # imshow(img, title = [[x.item()] for x in labels])

for i in range(4):
  show_batch_images(train_loader)


# #default config
# config = {
#     "size_filters" : [7,5,5,3,3],
#     "activation" : 'ReLU',
#     "learning_rate": 1e-3,
#     "filters_org": 1,
#     "num_filters" : 64,
#     "dense_layer_size" : 256,
#     "batch_norm": True,
#     "data_augment": True,
#     "dropout":0.2,
#     "batch_size":32,
#     "epochs": 10
#     }

class CNN(nn.Module):
  def __init__(self, config_given = None):
    super(CNN, self).__init__()

    if config_given == None:
      self.size_filters = [7,5,5,3,3]
      self.activation = 'ReLU'
      self.learning_rate = 1e-3
      self.filters_org =  1
      self.num_filters =  64
      self.dense_layer_size =  256
      self.batch_norm =  True
      self.data_augment = True
      self.dropout = 0.2
      self.batch_size = 32
      self.epochs =  10
    else:
      self.size_filters = config_given['size_filters']
      self.activation = config_given['activation']
      self.learning_rate = config_given['learning_rate']
      self.filters_org =  config_given['filters_org']
      self.num_filters =  config_given['num_filters']
      self.dense_layer_size = config_given['dense_layer_size']
      self.batch_norm = config_given['batch_norm']
      self.data_augment = config_given['data_augment']
      self.dropout = config_given['dropout']
      self.batch_size = config_given['batch_size']
      self.epochs =  config_given['epochs']

    self.cnn_model = nn.Sequential(
        nn.Conv2d(3,list_filters[0],config['size_filters'][0], padding=(config['size_filters'][0] - 1)//2), #(N, 3, 256, 256) -> (N, 16, 256, 256)
        nn.GELU() if config["activation"] == "GELU" else 
        nn.SiLU() if config["activation"] == "SiLU" else 
        nn.LeakyReLU() if config["activation"] == "leaky_relu" else 
        nn.ReLU(),
        nn.BatchNorm2d(num_features=list_filters[0]) if config["batch_norm"] else nn.Identity(),
        nn.MaxPool2d(2, stride = 2), #(N, 16, 256, 256) -> (N, 16, 128, 128)
        nn.Conv2d(list_filters[0],list_filters[1],config['size_filters'][1], padding=(config['size_filters'][1] - 1)//2), #(N, 16, 128, 128) -> (N, 16, 128, 128)
        nn.GELU() if config["activation"] == "GELU" else 
        nn.SiLU() if config["activation"] == "SiLU" else 
        nn.LeakyReLU() if config["activation"] == "leaky_relu" else 
        nn.ReLU(),
        nn.BatchNorm2d(num_features=list_filters[1]) if config["batch_norm"] else nn.Identity(),
        nn.MaxPool2d(2, stride = 2), # (N, 16, 128, 128) -> (N, 16, 64, 64)
        nn.Conv2d(list_filters[1],list_filters[2],config['size_filters'][2], padding=(config['size_filters'][2] - 1)//2), #(N, 16, 64, 64) -> (N, 16, 64, 64)
        nn.GELU() if config["activation"] == "GELU" else 
        nn.SiLU() if config["activation"] == "SiLU" else 
        nn.LeakyReLU() if config["activation"] == "leaky_relu" else 
        nn.ReLU(),
        nn.BatchNorm2d(num_features=list_filters[2]) if config["batch_norm"] else nn.Identity(),
        nn.MaxPool2d(2, stride = 2), # (N, 16, 64, 64) -> (N, 16, 32, 32)
        nn.Conv2d(list_filters[2],list_filters[3],config['size_filters'][3], padding=(config['size_filters'][3] - 1)//2), #(N, 16, 32, 32) -> (N, 16, 32, 32)
        nn.GELU() if config["activation"] == "GELU" else 
        nn.SiLU() if config["activation"] == "SiLU" else 
        nn.LeakyReLU() if config["activation"] == "leaky_relu" else 
        nn.ReLU(),
        nn.BatchNorm2d(num_features=list_filters[3]) if config["batch_norm"] else nn.Identity(),
        nn.MaxPool2d(2, stride = 2), # (N, 16, 32, 32) -> (N, 16, 16, 16)
        nn.Conv2d(list_filters[3],list_filters[4],config['size_filters'][4], padding=(config['size_filters'][4] - 1)//2), #(N, 16, 16, 16) -> (N, 16, 16, 16)
        nn.GELU() if config["activation"] == "GELU" else 
        nn.SiLU() if config["activation"] == "SiLU" else 
        nn.LeakyReLU() if config["activation"] == "leaky_relu" else 
        nn.ReLU(),
        nn.BatchNorm2d(num_features=list_filters[4]) if config["batch_norm"] else nn.Identity(),
        nn.MaxPool2d(2, stride = 2), # (N, 16, 16, 16) -> (N, 16, 8, 8)
    )
    self.fully_conn_model = nn.Sequential(
        nn.Linear(list_filters[4]*(8**2), config['dense_layer_size']), #(N, 59536) -> (N, 128)
        nn.GELU() if config["activation"] == "GELU" else 
        nn.SiLU() if config["activation"] == "SiLU" else 
        nn.LeakyReLU() if config["activation"] == "leaky_relu" else 
        nn.ReLU(),
        nn.Dropout(p=config["dropout"]),
        nn.Linear(config['dense_layer_size'], num_classes) # (N, 84) -> (N, 10)
    )
    self.run_name = "loss_{}_lr_{}_ac_{}_in_{}_op_{}_bs_{}_ep_{}_nn_{}".format(self.loss_function, self.learning_rate, self.activations, self.weight_initialization, self.optimizer, self.batch_size, self.epochs, self.hidden_layers)
    self.run_name = 'fs_[7,5,5,3,3]_ac_ReLU_lr_1e-3_fo_1_nf_64_dls_256_bn_True_da_True_do_0.2_bs_32_ep_10'.format()
  
  def forward(self, x):
    # print(x.shape)
    x = self.cnn_model(x)
    # print(x.shape)
    x = x.view(x.size(0), -1) # (N, 16, 5, 5) -> (N, 400)
    # print(x. shape)
    x = self.fully_conn_model(x)
    # print(x.shape)
    return x

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(device)

def evaluation(dataloader):
  total, correct = 0, 0
  for data in dataloader:
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = net(inputs)
    max_values, pred_class = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (pred_class == labels).sum().item()
  return 100*correct/total


sweep_config = {
    "name" : "Assignment2_P1_Q2_",
    "method" : "bayes",
    'metric': {
        'name': 'val_acc',
        'goal': 'maximize'
    },
    "parameters" : {
        "epochs" : {
            "values" : [10, 15, 20]
        },
        "batch_size": {
            "values": [16, 32, 64]
        },
        'activation': {
            'values': ['ReLU', 'leaky_relu', 'GELU', 'SiLU']
        },
        'learning_rate':{
            "values": [0.001,0.005, 0.0001,0.0005]
        },
        "dropout": {
            "values": [0,0.1,0.2,0.3]
        },
        "batch_norm": {
              "values": [True,False]
        },
        "data_augment": {
              "values": [True,False]
        },
        'size_filters':{
            'values': [[7,5,5,3,3], [11,9,7,5,3]]
        },
        'filters_org': {
            'values': [1, 2, 0.5]
        },
        'num_filters': {
            'values': [32, 64, 128]
        },
        "dense_layer_size": {
              "values": [64, 128, 256, 512]
          }        
    }
}




sweep_id = wandb.sweep(sweep_config,project='Testing', entity='dl_research')

def train():
    with wandb.init() as run:
        config = wandb.config

        batch_size = config['batch_size']

        list_filters = []
        for i in range(5):
          if config['filters_org'] == 1:
            list_filters.append(config['num_filters'])
          elif config['filters_org'] == 0.5:
            list_filters.append(int(config['num_filters']/(2**i)))
          else:
            list_filters.append(int(config['num_filters']*(2**i)))

        if config["data_augment"]:
            transform_train = transforms.Compose([
                transforms.RandomRotation(degrees=50),
                transforms.ColorJitter(brightness=(0.2,0.8)),
                transforms.RandomAffine(degrees=0, translate=(0.1,0.2), shear=0, scale=(0.7, 1.3)),
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomVerticalFlip(p=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5)),
                transforms.RandomResizedCrop(256)
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5)),
                transforms.RandomResizedCrop(256)
            ])

        transform_val = transforms.Compose([
            # transforms.RandomResizedCrop(256),
            # transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5)),
            transforms.RandomResizedCrop(256)
        ])

        train_dataset = ImageFolder(
            'inaturalist_12K/train',
            transform=transform_train
        )

        val_dataset = ImageFolder(
            'inaturalist_12K/train',
            transform=transform_val
        )

        # Define the indices to split between training and validation datasets
        num_train = len(train_dataset)
        indices = list(range(num_train))

        split = int(np.floor(0.2 * num_train))

        # Shuffle the indices before splitting
        np.random.seed(0)
        np.random.shuffle(indices)

        train_indices, val_indices = indices[split:], indices[:split]

        # Define the samplers for training and validation sets
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        # Use the samplers to create the DataLoader for validation set
        val_loader = torch.utils.data.DataLoader(
            dataset = val_dataset,
            batch_size=batch_size,
            sampler=val_sampler
            # shuffle = True
        )


        train_loader = torch.utils.data.DataLoader(
            dataset = train_dataset,
            batch_size=batch_size,
            sampler=train_sampler
            # shuffle = True
        )

        net = CNN().to(device)
        loss_fn = nn.CrossEntropyLoss()
        opt = optim.Adam(net.parameters(), lr=config['learning_rate'])
        run.name = net.run_name

        loss_arr = []
        val_loss_arr = []
        loss_ep_arr = []
        val_loss_ep_arr = []
        max_epochs = 3

        for epoch in range(max_epochs):
          for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            opt.zero_grad()
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            opt.step()
            loss_arr.append(loss.item())
          with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
              inputs, labels = data
              inputs, labels = inputs.to(device), labels.to(device)     
              # Forward Pass
              outputs = net(inputs)
              # Find the Loss
              val_loss = loss_fn(outputs, labels)
              val_loss_arr.append(val_loss.item())
        
          loss_ep_arr.append(loss.item())
          val_loss_ep_arr.append(val_loss.item())
          
          print('Epoch: %d/%d, Loss: %0.2f, Val Loss: %0.2f, Validation accuracy: %0.2f, Train accuracy: %0.2f'%(epoch, max_epochs, loss.item(), val_loss.item(), evaluation(val_loader), evaluation(train_loader)))

        
          metrics = {
          "accuracy":evaluation(train_loader),
            "loss":loss.item(),
          "validation_accuracy": evaluation(val_loader),
          "validation_loss": val_loss.item(),
            "epochs":epoch
            }

          wandb.log(metrics)     


wandb.agent(sweep_id, function=train, count=2)

# print("Final Scores: \nModel Hyperparameters: {}\nAccuracy: {}\nLoss: {}\nValidation Accuracy: {}\nValidation Loss {}".format(tuned_models[0]['model'], tuned_models[0]['accuracy'], tuned_models[0]['loss'], tuned_models[0]['validation_accuracy'], tuned_models[0]['validation_loss']))











