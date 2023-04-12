# !pip install wandb
# !wandb login --relogin

#importing required packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import wandb
import os
import argparse
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

font = {'size'   : 6}

matplotlib.rc('font', **font)


np.random.seed(42)


#default config
config = {
    "size_filters" : [7,5,5,3,3],
    "activation" : 'ReLU',
    "learning_rate": 1e-3,
    "filters_org": 1,
    "num_filters" : 64,
    "dense_layer_size" : 256,
    "batch_norm": True,
    "data_augment": False,
    "dropout":0.1,
    "batch_size":32,
    "epochs": 10
    }

parser = argparse.ArgumentParser()

parser.add_argument("-e", "--epochs", default=config['epochs'], type=type(config['epochs']), required=False, 
                    help=f'Number of epochs to train neural network.')

parser.add_argument('-ac','--activation', type=type(config['activation']), required = False, default = config['activation']
                    ,help=f'choices: ["GELU", "SiLU", "leaky_relu", "ReLU"]', choices=["GELU", "SiLU", "tanh", "ReLU"])

parser.add_argument('-lr','--learning_rate', type=type(config['learning_rate']), default = config['learning_rate']
                    ,help=f"Learning rate for CNN Model")

parser.add_argument('-fl','--num_filters', type=type(config['num_filters']), default = config['num_filters']
                    ,help=f"Number of filters to be used on each ConvLayer")

parser.add_argument('-bn','--batch_norm', type=type(config['batch_norm']), default = config['batch_norm']
                    ,help=f"Boolean Value of whether to do Batch Norm or not.")

parser.add_argument('-do','--dropout', type=type(config['dropout']), default = config['dropout']
                    ,help=f"Dropout added to the dense layer")

parser.add_argument('-da','--data_augment', type=type(config['data_augment']), default = config['data_augment']
                    ,help=f"Boolean Value of whether to use Data augmentation or not.")

parser.add_argument('-fl','--filters_org', type=type(config['filters_org']), default = config['filters_org']
                    ,help=f"Number of filters either remains same/doubled/halved at every consecutive ConvLayer")

parser.add_argument('-bs','--batch_size', type=type(config['batch_size']), default = config['batch_size']
                    ,help=f"Batch Size for processing images batchwise.")

parser.add_argument('-dl','--dense_layer_size', type=type(config['dense_layer_size']), default = config['dense_layer_size']
                    ,help=f"Number of neurons in fully connected dense layer")

parser.add_argument('-fs','--size_filters', type=type(config['size_filters']),default = config['kernel_sizes']
                    ,help=f"Kernel sizes for respective CNN layers.")



args = parser.parse_args()
config  = vars(args)

# # wandb_project = args.wandb_project
# # wandb_entity = args.wandb_entity
# dataset = args.dataset
# epochs = args.epochs
# batch_size = args.batch_size
# loss = args.loss
# optimizer = args.optimizer
# learning_rate = args.learning_rate
# momentum = args.momentum
# beta = args.beta
# beta1 = args.beta1
# beta2 = args.beta2
# epsilon = args.epsilon
# weight_decay = args.weight_decay
# weight_init = args.weight_init
# num_layers = args.num_layers
# hidden_size = args.hidden_size
# activation = args.activation


# sweep_config_train = {
#   "name" : "cs6910_assignment1_fashion-mnist_sweep",
#   "method" : "bayes",
#   "metric" : {
#       "name" : "validation_accuracy",
#       "goal" : "maximize"
#   },
#   "parameters" : {
#     "epochs" : {
#       "values" : [epochs]
#     },
#     "learning_rate" :{
#       "values" : [learning_rate]
#     },
#     "no_hidden_layers":{
#         "values" : [num_layers]
#     },
#     "hidden_layers_size":{
#         "values" : [hidden_size]
#     },
#     "weight_decay":{
#       "values": [weight_decay] 
#     },
#     "optimizer":{
#         "values": [optimizer]
#     },
#     "batch_size":{
#         "values":[batch_size]
#     },
#     "weight_initialization":{
#         "values": [weight_init]
#     },
#     "activations":{
#         "values": [activation]
#     }
#   }
# }

sweep_config = {
    "name" : "Assignment2_P1_Q2_",
    "method" : "bayes",
    'metric': {
        'name': 'val_acc',
        'goal': 'maximize'
    },
    "parameters" : {
        "epochs" : {
            "values" : [10, 15, 20 , 25, 30]
        },
        "batch_size": {
            "values": [16, 32, 64, 128]
        },
        'activation': {
            'values': ['relu', 'leaky_relu', 'gelu', 'silu']
        },
        'learning_rate':{
            "values": [0.001,0.0001,0.0003,0.0005]
        },
        "dropout": {
            "values": [0,0.1,0.2,0.3]
        },
        "batch_norm": {
              "values": [True,False]
        },
        "data_aug": {
              "values": [True,False]
        },
        'size_filters':{
            'values': [[7,5,5,3,3], [11,9,7,5,3]]
        },
        'filter_organization': {
            'values': [1, 2, 0.5]
        },
        'number_initial_filters': {
            'values': [16, 32, 64, 128]
        },
        "neurons_in_dense_layer": {
              "values": [32, 64, 128, 256, 512, 1024]
          }        
    }
}




sweep_id_train = wandb.sweep(sweep_config,project=wandb_project, entity=wandb_entity)

tuned_models = []
def train():
    with wandb.init() as run:


        # config = wandb.config
        model = FeedForwardNN(config=None)
        run.name = model.run_name
        print("Hyperparameter Settings: {}".format(run.name))
        train_acc,train_loss,val_acc,val_loss = 0,0,0,0
        for epoch in range(epochs):
            train_acc,train_loss,val_acc,val_loss = model.fit()  # model training code here
            metrics = {
            "accuracy":train_acc,
             "loss":train_loss,
            "validation_accuracy": val_acc,
            "validation_loss": val_loss,
             "epochs":epoch
             }
            print({
            "epochs": epoch,
            "accuracy":train_acc,
            "loss":train_loss,
            "validation_accuracy": val_acc,
            "validation_loss": val_loss,
            })
            wandb.log(metrics) 
        tuned_models.append({
            "accuracy":train_acc,
            "loss":train_loss,
            "validation_accuracy": val_acc,
            "validation_loss": val_loss,
            "model": run.name
        })          


wandb.agent(sweep_id_train, function=train, count=1)

print("Final Scores: \nModel Hyperparameters: {}\nAccuracy: {}\nLoss: {}\nValidation Accuracy: {}\nValidation Loss {}".format(tuned_models[0]['model'], tuned_models[0]['accuracy'], tuned_models[0]['loss'], tuned_models[0]['validation_accuracy'], tuned_models[0]['validation_loss']))