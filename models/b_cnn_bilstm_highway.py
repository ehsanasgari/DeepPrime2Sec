import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import numpy as np
from keras.models import Model
from keras.layers import Dense, CuDNNLSTM, Bidirectional, Input, Dropout, concatenate, Conv1D, \
    BatchNormalization
from keras.layers.wrappers import TimeDistributed
from layers.crf import ChainCRF
from layers.utility import slice_tensor
from keras import optimizers
from keras import regularizers

np.random.seed(0)


def model_b_cnn_bilstm_highway(n_classes, convs=[3, 5, 7], dense_size=200, lstm_size=400, dropout_rate=0.5,
                               features_to_use=['onehot', 'pssm'], filter_size=256, lr=0.001,
                               use_CRF=False):
    '''
    :param n_classes:
    :param convs:
    :param dense_size:
    :param lstm_size:
    :param dropout_rate:
    :param features_to_use:
    :param filter_size:
    :param lr:
    :param use_CRF:
    :return:
    '''

    visible = Input(shape=(None, 408))

    # slice different feature types
    biophysical = slice_tensor(2, 0, 16, name='biophysicalfeatures')(visible)
    embedding = slice_tensor(2, 16, 66, name='skipgramembd')(visible)
    onehot = slice_tensor(2, 66, 87, name='onehot')(visible)
    pssm = slice_tensor(2, 87, 108, name='pssm')(visible)
    # we need batchnorm for the highway
    batchnorm_profile = BatchNormalization(name='batchnormseqprof')(pssm)
    elmo = slice_tensor(2, 108, 408, name='elmo')(visible)

    # create input based-on selected features
    input_dict = {'pssm': pssm, 'onehot': onehot, 'embedding': embedding, 'elmo': elmo,
                  'biophysical': biophysical}
    features = []
    for feature in features_to_use:
        features.append(input_dict[feature])

    ## batch normalization on the input features
    if len(features_to_use) == 1:
        conclayers = features
        input = BatchNormalization(name='batchnorm_input')(features[0])
    else:
        input = BatchNormalization(name='batchnorm_input')(concatenate(features))
        conclayers = [input]

    # performing the conlvolutions
    for idx, conv in enumerate(convs):
        idx = str(idx + 1)
        conclayers.append(BatchNormalization(name='batch_norm_conv' + idx)(
            Conv1D(filter_size, conv, activation="relu", padding="same", name='conv' + idx,
                   kernel_regularizer=regularizers.l2(0.001))(input)))
    conc = concatenate(conclayers)

    # Dropout and Dense Layer before LSTM
    if dropout_rate > 0:
        drop_before = Dropout(dropout_rate, name='dropoutonconvs')(conc)
        dense_convinp = Dense(dense_size, activation='relu', name='denseonconvs')(drop_before)
    else:
        dense_convinp = Dense(dense_size, activation='relu', name='denseonconvs')(conc)

    # Batch normalize the results of dropout
    dense_convinpn = BatchNormalization(name='batch_norm_dense')(dense_convinp)

    # LSTM
    lstm = Bidirectional(CuDNNLSTM(lstm_size, return_sequences=True, name='bilstm'))(dense_convinpn)
    drop_after_lstm = Dropout(dropout_rate)(lstm)

    # Highway
    dense_out = Dense(dense_size, activation='relu')(drop_after_lstm)
    highway_layer = concatenate([dense_out, batchnorm_profile])
    highway_out = Dense(dense_size, activation='relu')(highway_layer)

    if use_CRF:
        timedist = TimeDistributed(Dense(n_classes, name='crf_in'))(highway_out)
        crf = ChainCRF(name="crf1")
        crf_output = crf(timedist)
        model = Model(inputs=visible, outputs=crf_output)
        adam = optimizers.Adam(lr=lr)
        model.compile(loss=crf.loss, optimizer=adam, weighted_metrics=['accuracy'], sample_weight_mode='temporal')
    else:
        timedist = TimeDistributed(Dense(n_classes, activation='softmax'))(highway_out)
        model = Model(inputs=visible, outputs=timedist)
        adam = optimizers.Adam(lr=lr)
        model.compile(loss='categorical_crossentropy', optimizer=adam, weighted_metrics=['accuracy'],
                      sample_weight_mode='temporal')
    print(model.summary())
    return model, 'model_b_cnn_bilstm_highway#' + '#'.join(features_to_use) + '@conv' + '_'.join(
        [str(c) for c in convs]) + '@dense_' + str(dense_size) + '@lstm' + str(lstm_size) + '@droplstm' + str(
        dropout_rate) + '@filtersize_' + str(filter_size) + '@lr_' + str(lr) + '@crf_' + str(use_CRF)
