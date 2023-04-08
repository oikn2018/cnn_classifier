
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
# print(classes)

image_size = (256,256)
num_layers = 5
num_classes = len(classes)

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



# def imshow(img, title):
#   npimg = img.numpy()/2 + 0.5
#   plt.figure(figsize = (batch_size*3, 3))
#   plt.axis('off')
#   plt.imshow(np.transpose(npimg, (1,2,0)))
#   plt.title(title)
#   plt.show()

# def show_batch_images(dataloader):
#   images, labels = next(iter(dataloader))
#   img = torchvision.utils.make_grid(images)
#   imshow(img, title = [classes[x.item()] for x in labels])
#   # imshow(img, title = [[x.item()] for x in labels])

# for i in range(4):
#   show_batch_images(train_loader)



class CNN(nn.Module):
  def __init__(self, config = None):
    super(CNN, self).__init__()

    if config == None:
      config = {}
      config['size_filters'] = [7,5,5,3,3]
      config['activation'] = 'ReLU'
      config['learning_rate'] = 1e-3
      config['filters_org'] =  1
      config['num_filters'] =  64
      config['dense_layer_size'] =  256
      config['batch_norm'] =  True
      config['data_augment'] = True
      config['dropout'] = 0.2
      config['batch_size'] = 32
      config['epochs'] =  10
    # else:
    #   self.size_filters = config['size_filters']
    #   self.activation = config['activation']
    #   self.learning_rate = config['learning_rate']
    #   self.filters_org =  config['filters_org']
    #   self.num_filters =  config['num_filters']
    #   self.dense_layer_size = config['dense_layer_size']
    #   self.batch_norm = config['batch_norm']
    #   self.data_augment = config['data_augment']
    #   self.dropout = config['dropout']
    #   self.batch_size = config['batch_size']
    #   self.epochs =  config['epochs']

    

    list_filters = []
    for i in range(5):
        if config['filters_org'] == 1:
            list_filters.append(config['num_filters'])
        elif config['filters_org'] == 0.5:
            list_filters.append(int(config['num_filters']/(2**i)))
        else:
            list_filters.append(int(config['num_filters']*(2**i)))

    self.run_name = 'sf_{}_numfil_{}_ac_{}_lr_{}_dls_{}_bn_{}_da_{}_do_{}_bs_{}_ep_{}'.format(config['size_filters'], list_filters, config['activation'], config['learning_rate'], config['dense_layer_size'], config['batch_norm'], config['data_augment'], config['dropout'], config['batch_size'], config['epochs'])

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
    
  
  def forward(self, x):
    # print(x.shape)
    x = self.cnn_model(x)
    # print(x.shape)
    x = x.view(x.size(0), -1) # (N, 16, 5, 5) -> (N, 400)
    # print(x. shape)
    x = self.fully_conn_model(x)
    # print(x.shape)
    return x

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)
torch.cuda.empty_cache() 
def evaluation(dataloader, model):
  total, correct = 0, 0
  for data in dataloader:
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    max_values, pred_class = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (pred_class == labels).sum().item()
  return 100*correct/total


sweep_config_parta = {
    "name" : "Assignment2_P1_Q2_",
    "method" : "bayes",
    'metric': {
        'name': 'validation_accuracy',
        'goal': 'maximize'
    },
    "parameters" : {
        "epochs" : {
            "values" : [10,15,20]
        },
        "batch_size": {
            "values": [8, 16, 32]
        },
        'activation': {
            'values': ['ReLU', 'leaky_relu', 'GELU', 'SiLU']
        },
        'learning_rate':{
            "values": [0.001,0.005, 0.0001,0.0005]
        },
        "dropout": {
            "values": [0,0.1,0.2]
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
            'values': [32, 64]
        },
        "dense_layer_size": {
              "values": [64, 128]
          }        
    }
}




sweep_id_parta = wandb.sweep(sweep_config_parta,project='Testing', entity='dl_research')

def train():
    torch.cuda.empty_cache()
    with wandb.init() as run:

        config = wandb.config

        batch_size = config['batch_size']

        

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

        net = CNN(config=config).to(device)
        loss_fn = nn.CrossEntropyLoss()
        opt = optim.Adam(net.parameters(), lr=config['learning_rate'])
        run.name = net.run_name
        print(run.name)


        # loss = 0
        # val_loss = 0
        # train_acc = 0
        # val_acc = 0
        # max_epochs = 2
        # loss_ep_arr = []
        # val_loss_ep_arr = []

        for epoch in range(config['epochs']):
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                opt.zero_grad()
                outputs = net(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                opt.step()

                del inputs, labels, outputs
                torch.cuda.empty_cache()
                # loss = loss.item()
                # loss_arr.append(loss.item())
            with torch.no_grad():
                for i, data in enumerate(val_loader, 0):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)     
                    # Forward Pass
                    outputs = net(inputs)
                    # Find the Loss
                    val_loss = loss_fn(outputs, labels)

                    # val_loss = val_loss.item()
                    # val_loss_arr.append(val_loss.item())
                    del inputs, labels, outputs
                    torch.cuda.empty_cache()
        
            # loss_ep_arr.append(loss.item())
            # val_loss_ep_arr.append(val_loss.item())

            # loss = loss.item()
            # val_loss = val_loss.item()
            train_acc = evaluation(train_loader, net)
            val_acc = evaluation(val_loader,net)
            print('Epoch: %d/%d, Loss: %0.2f, Val Loss: %0.2f, Validation accuracy: %0.2f, Train accuracy: %0.2f'%((epoch+1), config['epochs'], loss.item(), val_loss.item(), val_acc, train_acc))

        
            metrics = {
            "accuracy":train_acc,
                "loss":loss.item(),
            "validation_accuracy": val_acc,
            "validation_loss": val_loss.item(),
            "epochs":(epoch+1)
            }

            wandb.log(metrics)     


wandb.agent(sweep_id_parta, function=train, count=10)

# print("Final Scores: \nModel Hyperparameters: {}\nAccuracy: {}\nLoss: {}\nValidation Accuracy: {}\nValidation Loss {}".format(tuned_models[0]['model'], tuned_models[0]['accuracy'], tuned_models[0]['loss'], tuned_models[0]['validation_accuracy'], tuned_models[0]['validation_loss']))


