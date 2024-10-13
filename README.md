# SAMDGPT
## Description

The repository is organised as follows:

- **data/**: contains data we use.
- **cache/**: stores intermediate results during pre-training.
- **checkpoints/**: contains checkpoints of the models after pre-training.
- **result/**: stores results when downstream task is done.
- **src/**: implements pre-training and downstream task.

## Package Dependencies

- python 3.8.16
- pytorch 1.10.1
- cuda 11.3
- pyG 2.1.0

## Running experiments
### Data Preparation
To download the datasets and generate fewshot data, run `python ./src/utils/generate_fewshot.py`

### Node Classification and Graph Classification
Default dataset is Photo. You need to change the corresponding parameters in *execute.py* to train and evaluate on other datasets.

Pretrain and Prompt tune:
`python ./src/execute.py \
  --dataset {Name of the main dataset, default: Cora} \  
  --pretrain_datasets {List of pretrain datasets, default: ['Wisconsin', 'Texas']} \  
  --downstream_task {Type of downstream task, options: node or graph, default: node} \  
  --alpha {Combination weight parameter, default: 1.0} \  
  --beta {Combination weight parameter, default: 1.0} \  
  --gpu {GPU device number, default: 0} \  
  --shot_num {Number of examples for few-shot learning, default: 1} \  
  --pretrain_method {Pretraining method, options: GRAPHCL, LP, or splitLP, default: GRAPHCL} \  
  --backbone {Type of backbone network, default: gcn}  
`

