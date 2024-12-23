
# !pip install wget

import torch
import gdown
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import random
import wget
from torchvision.utils import make_grid
import wandb
import os
import datetime
from matplotlib import gridspec
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

# Getting the Dataset
url='https://storage.googleapis.com/wandb_datasets/nature_12K.zip'
filename = os.path.basename(url)

if not os.path.exists(filename) and not os.path.exists("inaturalist_12K"):
  filename = wget.download(url)
  with ZipFile(filename, 'r') as z:
    print('Extracting files...')
    z.extractall()
    print('Done!')
  os.remove(filename)



url = 'https://drive.google.com/uc?id=1-yEMoh5h3DHms7LD_ot7hpMbB9Xymyk2&export=download'
if not os.path.exists('best_model.pth'):
    gdown.download(url = url, output='best_model.pth', quiet=False, fuzzy=True)

#default Config
config = {
   "wandb_project": 'Testing',
   "wandb_entity": 'dl_research',
    "size_filters" : [7,5,5,3,3],
    "activation" : 'ReLU',
    "learning_rate": 0.0001,
    "filters_org": 2,
    "num_filters" : 64,
    "dense_layer_size" : 128,
    "batch_norm": False,
    "data_augment": False,
    "dropout":0.1,
    "batch_size":8,
    "epochs": 15
    }



wandb.init(project="Testing", entity="dl_research", name="Question 4")

#Adding Command Line Arguments
parser = argparse.ArgumentParser()

parser.add_argument("-wp","--wandb_project", default=config['wandb_project'], type=str, required=False, help='Project name used to track experiments in Weights & Biases dashboard')

parser.add_argument("-we", "--wandb_entity", default=config['wandb_entity'], type=str, required=False, help='Wandb Entity used to track experiments in the Weights & Biases dashboard.')

parser.add_argument('-sf','--size_filters', nargs='+', type=int, default = config['size_filters']
                    ,help=f"Choose Filter sizes to be used in each of five layers. Give 5 integer values separated by a space.")

parser.add_argument('-ac','--activation', type=type(config['activation']), default = config['activation'],help=f"Choose Activation Function to be used in each of five layers. Choices: ['ReLU', 'GELU', 'SiLU', 'leaky_relu']", choices = ["ReLU", "GELU", "leaky_relu", "SiLU"])

parser.add_argument('-lr','--learning_rate', type=type(config['learning_rate']), default = config['learning_rate'],help=f"Choose Learning rate of the model.")

parser.add_argument('-fo','--filters_org', type=type(config['filters_org']), default = config['filters_org'],help=f"Choose scaling of number of filters in model, i.e. if value set to 2, number of filters will double at each convolution layer.")

parser.add_argument('-nf','--num_filters', type=type(config['num_filters']), default = config['num_filters'],help=f"Choose Number of filters at initial convolution layer. Henceforth, it will be multiplied by filters_org parameter to get number of filters at each convolution layer.")
                    
parser.add_argument('-dls','--dense_layer_size', type=type(config['dense_layer_size']), default = config['dense_layer_size'], help=f"Choose Number of neurons in fully connected dense layer")


parser.add_argument('-bn','--batch_norm', type=type(config['batch_norm']), default = config['batch_norm'],help=f"Choose Boolean value: True/False corresponding to whether to apply Batch Normalization or not.")

parser.add_argument('-da','--data_augment', type=type(config['data_augment']), default = config['data_augment'], help=f"Choose Boolean value: True/False corresponding to whether to apply Data Augmentation or not.")

parser.add_argument('-do','--dropout', type=type(config['dropout']), default = config['dropout'], help=f"Choose Dropout added to the dense layer")

parser.add_argument('-bs','--batch_size', type=type(config['batch_size']), default = config['batch_size'], help=f"Choose Batch Size to be used")

parser.add_argument('-ep','--epochs', type=type(config['epochs']), nargs='?', default = config['epochs'], help=f"Number of epochs for which to train model")


args = parser.parse_args()
config  = vars(args)




classes = sorted([name for name in os.listdir("inaturalist_12K/train") if name != '.DS_Store'])
# print(classes)

image_size = (256,256)
num_layers = 5
num_classes = len(classes)




class CNN(nn.Module):
  def __init__(self, config = None):
    super(CNN, self).__init__()

    if config == None:
       config = config

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

print('GPU Allocated: {}, Available: {}'.format(torch.cuda.get_device_name(torch.cuda.current_device()), torch.cuda.is_available()))
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


torch.cuda.empty_cache()


batch_size = config['batch_size']

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5)),
    transforms.RandomResizedCrop(256, antialias=True)
])

test_dataset = ImageFolder(
    'inaturalist_12K/val',
    transform=transform_test
)


# Use the samplers to create the DataLoader for test set
test_loader = torch.utils.data.DataLoader(
    dataset = test_dataset,
    batch_size=batch_size,
    shuffle = True
)



net = CNN(config=config).to(device)
opt = optim.Adam(net.parameters(), lr=config['learning_rate'])
checkpoint = torch.load('best_model.pth')
epoch = checkpoint['epoch']

net.load_state_dict(checkpoint['model_state_dict'])

loss_fn = nn.CrossEntropyLoss()
opt.load_state_dict(checkpoint['optimizer_state_dict'])



name = net.run_name
print("Best Model Config: {}".format(name))
net.eval()

with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)     
        # Forward Pass
        outputs = net(inputs)
        # Find the Loss
        test_loss = loss_fn(outputs, labels)

        del inputs, labels, outputs
        torch.cuda.empty_cache()

test_acc = evaluation(test_loader,net)
print('Test Loss: %0.2f, Test accuracy: %0.2f'%(test_loss.item(), test_acc))





image_list = []
label_list = []
i=0

def imshow(img, title):
   npimg = img.numpy()/2 + 0.5
   plt.figure()
   plt.axis('off')
   plt.imshow(np.transpose(npimg, (1,2,0)))
   plt.title(title)
   plt.show()



for count in range(len(classes)):
   
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    # for image, label in random.sample(list(zip(images, labels)), 3):
    outputs = net(images)
    _, pred_labels = torch.max(outputs.data, 1)
    pred_labels = pred_labels.flatten().tolist()
    labels = labels.flatten().tolist()

    randlist = random.sample(range(8), 3)
    
    classify = ""
    for index in randlist:
        true_class = classes[pred_labels[index]]
        pred_class = classes[labels[index]]
        flag = 'Correct Classification' if true_class == pred_class else 'Wrong Classification'
        classify = 'True Class: ' + true_class + "\n" + 'Predicted Class: ' + pred_class + "\n" + flag

        image_list.append(images[index].to('cpu'))
        label_list.append(classify)



wandb.log({"Sample images from Test Data and Predictions by Best Model": [wandb.Image(image, caption=label, grouping = 3) for image, label in zip(image_list, label_list)]})
