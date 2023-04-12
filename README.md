# CS6910 Assignment 2
Name: Oikantik Nath | Roll: CS22S013 | Course: CS6910 Fundamentals of Deep Learning | [WandB Report](https://wandb.ai/dl_research/Testing/reports/CS6910-Assignment-2--VmlldzozOTQ4OTQ0?accessToken=2tquusi34lylzkeg6anhqvij3zsl2t25yfmo9h0jnec7i8ejhz3xhh11l5rga40q)

## Part A: Training from scratch
### Question 2
Code for running sweeps can be accessed [here](https://github.com/oikn2018/CS6910_assignment_2/blob/main/PartA/q2_sweep.py).

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
Code to log a single run with desired hyperparameters vis cmd is given at this [link](https://github.com/oikn2018/CS6910_assignment_2/blob/main/PartA/q2.py).

#### Format: 
`python q2.py -wp <wandb_project> -we <wandb_entity> -sf <size_filters> -ac <activation> -lr <learning_rate> -fo <filters_org> -nf <num_filters> -dls <dense_layer_size> -bn <batch_norm> -da <data_augment> -do <dropout> -bs <batch_size> -ep <epochs>`

To obtain validation accuracy of my best model: 
`python q2.py`

NOTE: 
best_model.pth file containing the State Dict of my best model and optimizer can be found at this [link](https://drive.google.com/uc?id=1-yEMoh5h3DHms7LD_ot7hpMbB9Xymyk2&export=download)

### Question 4
Code for this question can be accessed [here](https://github.com/oikn2018/CS6910_assignment_2/blob/main/PartA/q4_test.py).

### Run the Code
To test my best model, execute in cmd: 
```python
pip install gdown
pip install --upgrade gdown
python q4_test.py
```

## Part B: Fine-tuning a pre-trained model
### Question 3
Code for this question can be accessed [here](https://github.com/oikn2018/CS6910_assignment_2/blob/main/PartB/a2_partB_pretrained.ipynb).

---
The codes are organized as follows:

| Question | Location | Function | 
|----------|----------|----------|
| Part A Question 2 | [Part_A_Question_2](https://github.com/oikn2018/CS6910_assignment_2/blob/main/PartA/q2_sweep.py) | Running Sweeps | 
| Part A Question 4 | [Part_A_Question_4](https://github.com/oikn2018/CS6910_assignment_2/blob/main/PartA/q4_test.py) | Test Best Model Obtained and Log Predicted vs True Class for some Test Images |
| Part B Question 3 | [Part_B_Question_3](https://github.com/oikn2018/CS6910_assignment_2/blob/main/PartB/a2_partB_pretrained.ipynb) | Pretrained ResNet50 trained on iNaturalist Dataset | 

