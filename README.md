# DeepPrime2Sec

## Summary


## Conda environment set up

In order to install the required libraries for running DeepPrime2Sec use the following conda command:

```
conda create --name deepprime2sec --file installations/env_linux.txt
```

Subsequently, you need to activate the created virtual environment before running:

```
source activate DeepPrime2Sec
```


## Running example

In order to run the DeepPrime2Sec, you can simply use the following command.
Every details on different deep learning models: architecture, hyper parameter, training parameters, etc, will be provided in the yaml config file. Later in the README we have detailed how this file should be created.

```
python deepprime2sec.py --config sample_configs/model_a.yaml
```


### Sample output


## Features to use



## How to configure input for different deep learning models


### Model (a) CNN + BiLSTM


Sample config file
```
deep_learning_model: a_cnn_bilstm
model_paramters:
  convs:
  - 3
  - 5
  - 7
  - 11
  - 21
  dense_size: 1000
  dropout_rate: 0.5
  features_to_use:
  - onehot
  - sequence_profile
  lr: 0.001
  lstm_size: 1000
run_parameters:
  domain_name: baseline
  epochs: 10
  gpu: 1
  patience: 10
  setting_name: baseline
  test_batch_size: 100
  train_batch_size: 64
```
## Model (b) CNN + BiLSTM + Highway Connection of PSSM


Sample config file
```
deep_learning_model: model_b_cnn_bilstm_highway
model_paramters:
  convs:
  - 3
  - 5
  - 7
  - 11
  - 21
  dense_size: 1000
  dropout_rate: 0.5
  features_to_use:
  - onehot
  - sequence_profile
  filter_size: 256
  lr: 0.001
  lstm_size: 1000
  use_CRF: false
run_parameters:
  domain_name: baseline
  epochs: 100
  gpu: 1
  patience: 10
  setting_name: baseline
  test_batch_size: 100
  train_batch_size: 64

```


## Model (c) CNN + BiLSTM + Conditional Random Field Layer

Sample config file
```
deep_learning_model: model_c_cnn_bilstm
model_paramters:
  CRF_input_dim: 200
  convs:
  - 3
  - 5
  - 7
  - 11
  - 21
  dense_size: 1000
  dropout_rate: 0.5
  features_to_use:
  - onehot
  - sequence_profile
  filter_size: 256
  lr: 0.001
  lstm_size: 1000
run_parameters:
  domain_name: baseline
  epochs: 100
  gpu: 1
  patience: 10
  setting_name: baseline
  test_batch_size: 100
  train_batch_size: 64

```

## Model (d) CNN + BiLSTM + Attention mechanism

Sample config file
```
deep_learning_model: model_d_cnn_bilstm_attention
model_paramters:
  attention_type: additive
  attention_units: 32
  convs:
  - 3
  - 5
  - 7
  - 11
  - 21
  dense_size: 1000
  dropout_rate: 0.5
  features_to_use:
  - onehot
  - sequence_profile
  filter_size: 256
  lr: 0.001
  lstm_size: 1000
  use_CRF: false
run_parameters:
  domain_name: baseline
  epochs: 100
  gpu: 1
  patience: 10
  setting_name: baseline
  test_batch_size: 100
  train_batch_size: 64
```

## Model (e) CNN

Sample config file
```
deep_learning_model: model_e_cnn
model_paramters:
  convs:
  - 3
  - 5
  - 7
  - 11
  - 21
  dense_size: 1000
  dropout_rate: 0.5
  features_to_use:
  - onehot
  - sequence_profile
  filter_size: 256
  lr: 0.001
  use_CRF: false
run_parameters:
  domain_name: baseline
  epochs: 100
  gpu: 1
  patience: 10
  setting_name: baseline
  test_batch_size: 100
  train_batch_size: 64
```

## Model (f) Multiscale CNN

Sample config file
```
deep_learning_model: model_f_multiscale_cnn
model_paramters:
  cnn_regularizer: 5.0e-05
  convs:
  - 3
  - 5
  - 7
  - 11
  - 21
  dropout_rate: 0.5
  features_to_use:
  - onehot
  - sequence_profile
  filter_size: 256
  lr: 0.001
  multiscalecnn_layers: 3
  use_CRF: false
run_parameters:
  domain_name: baseline
  epochs: 100
  gpu: 1
  patience: 10
  setting_name: baseline
  test_batch_size: 100
  train_batch_size: 64
```
