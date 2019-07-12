import yaml

config_model_a = {'run_parameters':
                      {'domain_name': 'baseline', 'gpu': 1, 'setting_name': 'baseline', 'train_batch_size': 64,
                       'test_batch_size': 100, 'patience': 10, 'epochs': 100},
                  'deep_learning_model': 'model_a_cnn_bilstm',
                  'model_paramters': {'convs': [3, 5, 7, 11, 21], 'dense_size': 1000, 'lstm_size': 1000,
                                      'dropout_rate' : 0.5, 'filter_size':256,'lr' : 0.001, 'features_to_use': ['onehot',
                                                                                              'pssm']}}

config_model_b = {'run_parameters':
                      {'domain_name': 'baseline', 'gpu': 1, 'setting_name': 'baseline', 'train_batch_size': 64,
                       'test_batch_size': 100, 'patience': 10, 'epochs': 100},
                  'deep_learning_model': 'model_b_cnn_bilstm_highway',
                  'model_paramters': {'convs': [3, 5, 7, 11, 21], 'dense_size': 1000, 'lstm_size': 1000,
                                      'dropout_rate' : 0.5,'filter_size':256, 'lr' : 0.001, 'features_to_use': ['onehot',
                                                                                              'pssm'], 'use_CRF':False}}

config_model_c = {'run_parameters':
                      {'domain_name': 'baseline', 'gpu': 1, 'setting_name': 'baseline', 'train_batch_size': 64,
                       'test_batch_size': 100, 'patience': 10, 'epochs': 100},
                  'deep_learning_model': 'model_c_cnn_bilstm_crf',
                  'model_paramters': {'convs': [3, 5, 7, 11, 21], 'dense_size': 1000, 'lstm_size': 1000,
                                      'dropout_rate' : 0.5, 'filter_size':256, 'lr' : 0.001, 'features_to_use': ['onehot',
                                                                                              'pssm'], 'CRF_input_dim':200}}
config_model_d = {'run_parameters':
                      {'domain_name': 'baseline', 'gpu': 1, 'setting_name': 'baseline', 'train_batch_size': 64,
                       'test_batch_size': 100, 'patience': 10, 'epochs': 100},
                  'deep_learning_model': 'model_d_cnn_bilstm_attention',
                  'model_paramters': {'convs': [3, 5, 7, 11, 21], 'dense_size': 1000, 'lstm_size': 1000,
                                      'dropout_rate' : 0.5, 'filter_size':256,'lr' : 0.001, 'features_to_use': ['onehot',
                                                                                              'pssm'], 'use_CRF':False, 'attention_units':32, 'attention_type':'additive'}}

config_model_e = {'run_parameters':
                      {'domain_name': 'baseline', 'gpu': 1, 'setting_name': 'baseline', 'train_batch_size': 64,
                       'test_batch_size': 100, 'patience': 10, 'epochs': 100},
                  'deep_learning_model': 'model_e_cnn',
                  'model_paramters': {'convs': [3, 5, 7, 11, 21], 'dense_size': 1000,
                                      'dropout_rate' : 0.5, 'lr' : 0.001, 'filter_size':256,'features_to_use': ['onehot',
                                                                                              'pssm'], 'use_CRF':False}}

                    #multiplicative

config_model_f = {'run_parameters':
                      {'domain_name': 'baseline', 'gpu': 1, 'setting_name': 'baseline', 'train_batch_size': 64,
                       'test_batch_size': 100, 'patience': 10, 'epochs': 100},
                  'deep_learning_model': 'model_f_multiscale_cnn',
                  'model_paramters': {'convs': [3, 5, 7, 11, 21],
                                      'dropout_rate' : 0.5, 'lr' : 0.001, 'filter_size':256, 'features_to_use': ['onehot',
                                                                                              'pssm'], 'use_CRF':False, 'lr':0.001, 'cnn_regularizer':0.00005, 'multiscalecnn_layers':3}}

models = ['a','b','c','d','e','f']

for idx, config in enumerate([config_model_a,config_model_b, config_model_c, config_model_d, config_model_e, config_model_f]):
    c = yaml.dump(config)
    f = open('sample_configs/model_'+models[idx]+'.yaml', 'w')
    f.write(c)
    f.close()


