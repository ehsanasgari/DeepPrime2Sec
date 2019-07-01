
from utility.training import training_loop
from models.a_cnn_bilstm import model_a_cnn_bilstm
from models.b_cnn_bilstm_highway import model_b_cnn_bilstm_highway
from models.c_cnn_bilstm_crf import model_c_cnn_bilstm
from models.d_cnn_bilstm_attention import model_d_cnn_bilstm_attention
from models.e_cnn import model_e_cnn
from models.f_multiscale_cnn import model_f_multiscale_cnn


training_loop(model_a_cnn_bilstm, '1', **{'run_parameters' : {'domain_name':'baseline', 'setting_name':'baseline', 'train_batch_size':64, 'test_batch_size':100, 'patience':10, 'epochs':10, 'features_to_use':['onehot', 'sequence_profile']}, 'model_paramters' : {'convs':[3, 5, 7, 11, 21], 'dense_size':1000, 'lstm_size':1000,'use_CRF':False}});
