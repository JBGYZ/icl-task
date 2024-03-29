# Task Investigation: In-Context Learning Task for continous linear regression

## Overview

Welcome to the README file for our project investigation on In-Context Learning Task for continous linear regression. This project aims to benchmark different neural networks performance against the ICL linear regression tasks.

## Project Goals

The primary goals of this investigation include:

1. **Understanding In-Context Learning:** Explore in-context learning performance for common neural network architectures.

2. **Report sample complexity and performance:** Use MSE normalized by dimension to quantify the performance.


## Replication
The plots used in the report are generated by running 4 simulations with random seeds 0, 1, 2, 3 and averaging the results with the following commands: 
1. **Shallow Networks:** 

``` bash
python main.py --n_tasks 0 --n_dims 4 --n_points 8 --batch_size 256 \
 --data_seed 0 --task_seed 0 --noise_seed 0 --data_scale 1.0 --task_scale 1.0 --noise_scale 0 --dtype float32 --lr 0.0001 \
 --max_iterations 500000 --network cnn \
 --n_hidden_layers 4 --n_hidden_neurons 128 --dropout 0.1

python main.py --n_tasks 0 --n_dims 4 --n_points 8 --batch_size 256 \
 --data_seed 0 --task_seed 0 --noise_seed 0 --data_scale 1.0 --task_scale 1.0 --noise_scale 0 --dtype float32 --lr 0.0001 \
 --max_iterations 500000 --network transformer_fcn \
 --n_hidden_layers 4 --n_hidden_neurons 128 --dim_feedforward 128 --dropout 0.1

python main.py --n_tasks 0 --n_dims 4 --n_points 8 --batch_size 256 \
 --data_seed 0 --task_seed 0 --noise_seed 0 --data_scale 1.0 --task_scale 1.0 --noise_scale 0 --dtype float32 --lr 0.0001 \
 --max_iterations 500000 --network mlpmixer_fcn \
 --n_hidden_layers 4 --n_hidden_neurons 128 --dropout 0.1

python main.py --n_tasks 0 --n_dims 4 --n_points 8 --batch_size 256 \
 --data_seed 0 --task_seed 0 --noise_seed 0 --data_scale 1.0 --task_scale 1.0 --noise_scale 0 --dtype float32 --lr 0.0001 \
 --max_iterations 500000 --network fcn \
 --n_hidden_layers 4 --n_hidden_neurons 128 --dropout 0.1
```

2. **Deep Networks:** 

``` bash
python main.py --n_tasks 0 --n_dims 4 --n_points 8 --batch_size 256 \
 --data_seed 0 --task_seed 0 --noise_seed 0 --data_scale 1.0 --task_scale 1.0 --noise_scale 0 --dtype float32 --lr 0.0001 \
 --max_iterations 500000 --network cnn \
 --n_hidden_layers 4 --n_hidden_neurons 128 --dropout 0.1

python main.py --n_tasks 0 --n_dims 4 --n_points 8 --batch_size 256 \
 --data_seed 0 --task_seed 0 --noise_seed 0 --data_scale 1.0 --task_scale 1.0 --noise_scale 0 --dtype float32 --lr 0.0001 \
 --max_iterations 500000 --network transformer_fcn \
 --n_hidden_layers 4 --n_hidden_neurons 128 --dim_feedforward 128 --dropout 0.1

python main.py --n_tasks 0 --n_dims 4 --n_points 8 --batch_size 256 \
 --data_seed 0 --task_seed 0 --noise_seed 0 --data_scale 1.0 --task_scale 1.0 --noise_scale 0 --dtype float32 --lr 0.0001 \
 --max_iterations 500000 --network mlpmixer_fcn \
 --n_hidden_layers 4 --n_hidden_neurons 128 --dropout 0.1

python main.py --n_tasks 0 --n_dims 4 --n_points 8 --batch_size 256 \
 --data_seed 0 --task_seed 0 --noise_seed 0 --data_scale 1.0 --task_scale 1.0 --noise_scale 0 --dtype float32 --lr 0.0001 \
 --max_iterations 500000 --network fcn \
 --n_hidden_layers 4 --n_hidden_neurons 128 --dropout 0.1
```

## Dependencies

This file may be used to create an environment using:
$ conda create --name <env> --file <this file>