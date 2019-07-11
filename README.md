# DeepPrime2Sec


## Model (a) CNN + BiLSTM

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
