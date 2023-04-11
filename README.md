# CS6910 Assignment 2
Name: Oikantik Nath | Roll: CS22S013 | Course: CS6910 Fundamentals of Deep Learning | [WandB Report](https://wandb.ai/dl_research/Testing/reports/CS6910-Assignment-2--VmlldzozOTQ4OTQ0?accessToken=2tquusi34lylzkeg6anhqvij3zsl2t25yfmo9h0jnec7i8ejhz3xhh11l5rga40q)

## Part A: Training from scratch
### Question 2
Code for running sweeps can be accessed [here](https://github.com/oikn2018/CS6910_assignment_2/blob/main/q2_sweep.py).

**Configuration for Sweeps:**

```python
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
```

### Run the Code
To run the code, execute in cmd: 
`python q2_sweep.py`

### Question 4
Code for running sweeps can be accessed [here](https://github.com/oikn2018/CS6910_assignment_2/blob/main/q4.py).
### Run the Code
To run the code, execute in cmd: 
#### Format:
`python q4.py -wp <wandb_project_name> -we <wandb_entity_name> -e <epochs> -b <batch_size> -o <optimizer> -lr <learning_rate> -w_i <weight_initialization_method> -nhl <num_hidden_layers> -sz <size_hidden_layer> -a <activation_function>`

#### To test it on the best model achieved:
`python train.py -wp Testing -we dl_research -e 20 -b 64 -o nadam -lr 0.005 -w_i Xavier -nhl 5 -sz 512 -a sigmoid`

## Question 7
The confusion matrix code is available [here](https://github.com/oikn2018/CS6910_assignment_1/blob/main/Q7.py). It is tested on the Fashion-MNIST test data and the output confusion matrix is logged in the report.


## Question 10
Since the MNIST dataset is much simpler in terms of image complexity compared to Fashion-MNIST dataset which I have used in my experimentation, so I suggest the following 3 configurations that give me the best accuracy scores on the Fashion-MNIST dataset. You can access code [here](https://github.com/oikn2018/CS6910_assignment_1/blob/main/Q10.py).

- Configuration 1: 
```python
config = { 
	"epochs" : 20,
	"learning_rate": 0.005,
	"no_hidden_layers": 5, 
	"hidden_layers_size": 512,
	"weight_decay": 0,
	"optimizer": "nadam",
	"batch_size": 64,
	"weight_initialization" : "xavier" ,
	"activations" : "sigmoid",
}
```
To run above configuration, download [code](https://github.com/oikn2018/CS6910_assignment_1/blob/main/Q10.py) from GitHub and execute following command on cmd:
`python Q10.py -wp Testing -we dl_research -d mnist -e 20 -b 64 -o nadam -lr 0.005 -w_i Xavier -nhl 5 -sz 512 -a sigmoid`

- Configuration 2: 
```python
config = { 
	"epochs" : 20,
	"learning_rate": 0.005,
	"no_hidden_layers": 5, 
	"hidden_layers_size": 256,
	"weight_decay": 0,
	"optimizer": "nadam",
	"batch_size": 32,
	"weight_initialization" : "xavier" ,
	"activations" : "tanh",
}
```
To run above configuration, download [code](https://github.com/oikn2018/CS6910_assignment_1/blob/main/Q10.py) from GitHub and execute following command on cmd:
`python Q10.py -wp Testing -we dl_research -d mnist -e 20 -b 32 -o nadam -lr 0.005 -w_i Xavier -nhl 5 -sz 256 -a tanh`


- Configuration 3: 
```python
config = { 
	"epochs" : 20,
	"learning_rate": 0.0001,
	"no_hidden_layers": 5, 
	"hidden_layers_size": 256,
	"weight_decay": 0,
	"optimizer": "adam",
	"batch_size": 128,
	"weight_initialization" : "xavier" ,
	"activations" : "relu",
}
```
To run above configuration, download [code](https://github.com/oikn2018/CS6910_assignment_1/blob/main/Q10.py) from GitHub and execute following command on cmd:
`python Q10.py -wp Testing -we dl_research -d mnist -e 20 -b 128 -o adam -lr 0.0001 -w_i Xavier -nhl 5 -sz 256 -a relu`

---
The codes are organized as follows:

| Question | Location | Function | 
|----------|----------|----------|
| Question 1 | [Question-1](https://github.com/oikn2018/CS6910_assignment_1/blob/main/Q1.ipynb) | Plotting Sample Images of Each Class | 
| Question 2-4 | [Question-2-4](https://github.com/oikn2018/CS6910_assignment_1/blob/main/train.py) | Feedforward Neural Network Training and Evaluating Accuracies |
| Question 7 | [Question-7](https://github.com/oikn2018/CS6910_assignment_1/blob/main/Q7.py) | Confusion Matrix for Test Data on Best Model | 
| Question 10 | [Question-10](https://github.com/oikn2018/CS6910_assignment_1/blob/main/Q10.py) | 3 Best Hyperparameter configurations for MNIST | 
