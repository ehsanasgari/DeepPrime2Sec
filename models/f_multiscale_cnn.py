import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import numpy as np

np.random.seed(7)
from keras.models import Model
from keras.layers import Dense, Input, Dropout, concatenate, Conv1D, \
    BatchNormalization
from keras.layers.wrappers import TimeDistributed
from layers.crf import ChainCRF
from layers.utility import slice_tensor, multiscale_CNN
from keras import optimizers
from keras import regularizers



def model_f_multiscale_cnn(n_classes, convs=[3, 5, 7], dropout_rate=0.5,
                features_to_use=['onehot', 'pssm'], filter_size=256, lr=0.001, multiscalecnn_layers=3, cnn_regularizer=0.00005,
                use_CRF=False):
    '''
    :param n_classes:
    :param convs:
    :param dropout_rate:
    :param features_to_use:
    :param filter_size:
    :param lr:
    :param multicnn_layers:
    :param cnn_regularizer:
    :param use_CRF:
    :return:
    '''
    visible = Input(shape=(None, 408))

    # slice different feature types
    biophysical = slice_tensor(2, 0, 16, name='biophysicalfeatures')(visible)
    embedding = slice_tensor(2, 16, 66, name='skipgramembd')(visible)
    onehot = slice_tensor(2, 66, 87, name='onehot')(visible)
    pssm = slice_tensor(2, 87, 108, name='pssm')(visible)
    elmo = slice_tensor(2, 108, 408, name='elmo')(visible)

    input_dict = {'pssm': pssm, 'onehot': onehot, 'embedding': embedding, 'elmo': elmo,
                  'biophysical': biophysical}

    gating = Dense(len(convs) * filter_size, activation='sigmoid')

    # create input
    features = []
    for feature in features_to_use:
        features.append(input_dict[feature])

    if len(features_to_use) == 1:
        conclayers = features
        input = BatchNormalization(name='batchnorm_input')(features[0])
    else:
        input = BatchNormalization(name='batchnorm_input')(concatenate(features))
        conclayers = []

    # performing the conlvolutions
    for idx, conv in enumerate(convs):
        idx = str(idx + 1)
        conclayers.append(Conv1D(filter_size, conv, activation="relu", padding="same", name='conv' + idx,
                                 kernel_regularizer=regularizers.l2(cnn_regularizer))(input))
    current_multi_cnn_input = concatenate(conclayers)

    # Multiscale CNN application
    for layer_idx in range(multiscalecnn_layers-1):
        current_multi_cnn_output = multiscale_CNN(current_multi_cnn_input, gating, filter_size, convs, cnn_regularizer)
        current_multi_cnn_input = Dropout(dropout_rate)(current_multi_cnn_output)
    dense_out = Dense(len(convs) * filter_size, activation='relu')(current_multi_cnn_input)

    if use_CRF:
        timedist = TimeDistributed(Dense(n_classes, name='timedist'))(dense_out)
        crf = ChainCRF(name="crf1")
        crf_output = crf(timedist)
        model = Model(inputs=visible, outputs=crf_output)
        adam = optimizers.Adam(lr=lr)
        model.compile(loss=crf.loss, optimizer=adam, weighted_metrics=['accuracy'], sample_weight_mode='temporal')
    else:
        timedist = TimeDistributed(Dense(n_classes, activation='softmax'))(dense_out)
        model = Model(inputs=visible, outputs=timedist)
        adam = optimizers.Adam(lr=lr)
        model.compile(loss='categorical_crossentropy', optimizer=adam, weighted_metrics=['accuracy'],
                      sample_weight_mode='temporal')
    print(model.summary())
    return model, 'model_f_multiscale_cnn#' + '#'.join(features_to_use) + '@conv' + '_'.join(
        [str(c) for c in convs]) +  '@dropout_rate' + str(
        dropout_rate) + '@filtersize_' + str(filter_size) + '@lr_' + str(lr) + '@use_CRF_' + str(
        use_CRF) + '@multiscalecnn_layers' + str(multiscalecnn_layers) + '@cnn_regularizer' + str(cnn_regularizer)
