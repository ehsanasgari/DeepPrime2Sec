
from utility.training import training_loop
from models.a_cnn_bilstm import model_a_cnn_bilstm


training_loop('1', model_a_cnn_bilstm, {'run_parameters' : {'domain_name':'baseline', 'setting_name':'baseline', 'train_batch_size':64, 'test_batch_size':100, 'patience':10, 'epochs':10, 'features_to_use':['onehot', 'sequence_profile']}, 'model_paramters' : {'convs':[3, 5, 7], 'dense_size':200, 'lstm_size':400,'use_CRF':False}});
