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

