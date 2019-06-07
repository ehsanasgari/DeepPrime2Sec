'''
   Replicated the architecture from
   Deep Recurrent Conditional Random Field Network for Protein Secondary Prediction
'''

import numpy as np
np.random.seed(7)
from keras.models import Model
from keras.layers import Masking, Lambda, Dense, LSTM, CuDNNLSTM, Reshape, Flatten, Activation, RepeatVector, Permute, Bidirectional, Embedding, Input, Dropout, concatenate, Masking, Conv1D, \
    multiply, BatchNormalization, merge
from keras.layers.wrappers import TimeDistributed
from keras import regularizers
from models.layers import ChainCRF, slice_tensor
from models.keras_layers import ParallelizedWrapper, PositionEmbedding
import keras.backend as K
from keras import optimizers
from keras_multi_head import MultiHead
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

def cnn_cnnhpss(n_classes, convs=[3,5,7], dense_size=200, drop_lstm=0.5, features_to_use=['onehot','sequence_profile'], use_CRF=False, filter_size=256):
    '''
    :param max_length:
    :param n_classes:
    :param embedding_layer:
    :param vocab_size:
    :return:
    '''
    visible = Input(shape=(None,408))
    biophysical = slice_tensor(2,0,16,name='biophysicalfeatures')(visible)
    embedding = slice_tensor(2,16,66,name='skipgramembd')(visible)
    onehot = slice_tensor(2,66,87,name='onehot')(visible)
    prof = slice_tensor(2,87,108,name='sequenceprofile')(visible)
    elmo = slice_tensor(2,108,408,name='elmo')(visible)

    lstm_size = 1000
    input_dict={'sequence_profile':prof, 'onehot':onehot,'embedding':embedding,'elmo':elmo,'biophysical':biophysical}

    # create input
    features=[]
    for feature in features_to_use:
        features.append(input_dict[feature])


    if len(features_to_use)==1:
        conclayers=features
        input=BatchNormalization(name='batchnorm_input') (features[0])
    else:
        input =BatchNormalization(name='batchnorm_input') (concatenate(features))
        conclayers=[]

    # performing the conlvolutions
    for idx,conv in enumerate(convs):
        idx=str(idx+1)
        conclayers.append(BatchNormalization(name='batch_norm_conv'+idx) (Conv1D(filter_size, conv, activation="relu", padding="same", name='conv'+idx,
                   kernel_regularizer=regularizers.l2(0.00005))(input)))

    conc = concatenate(conclayers)

    sec_conv = []
    # performing the conlvolutions
    for idx,conv in enumerate(convs):
        idx=str(idx+1)
        sec_conv.append(BatchNormalization(name='batch_norm_conv_2nd_'+idx) (Conv1D(filter_size, conv, activation="relu", padding="same", name='conv_2nd_'+idx,
                   kernel_regularizer=regularizers.l2(0.00005))(conc)))

    sec_conc = concatenate(sec_conv)

    z_t = Dense(1, activation='sigmoid')(conc)
    
    #conv_input =  conc +  sec_conc
    ff = Lambda(lambda a: z_t*a[0] + (1-z_t)*a[1])([conc, sec_conc])
    
    
    third_conv = []
    # performing the conlvolutions
    for idx,conv in enumerate(convs):
        idx=str(idx+1)
        third_conv.append(BatchNormalization(name='batch_norm_conv_3nd_'+idx) (Conv1D(filter_size, conv, activation="relu", padding="same", name='conv_3nd_'+idx,
                   kernel_regularizer=regularizers.l2(0.00005))(ff)))
    third_conc = concatenate(third_conv)
    
    z_t_2 = Dense(1, activation='sigmoid')(ff)
    
    #conv_input =  conc +  sec_conc
    ff_2 = Lambda(lambda a: z_t_2*a[0] + (1-z_t_2)*a[1])([ff, third_conc])
    
    
    #ff = conv_input#Dense(100, activation='relu', name='final_ff')(conv_input)
    
    #if drop_lstm > 0:
    #    drop0 = Dropout(drop_lstm,name='dropoutonconvs')(ff)
    #    dense_convinp = Dense(dense_size, activation='relu', name='denseonconvs')(drop0)
    #else:
    #    dense_convinp = Dense(dense_size, activation='relu', name='denseonconvs')(ff)

    #dense_convinpn=BatchNormalization(name='batch_norm_dense') (dense_convinp)

    #lstm = Bidirectional(CuDNNLSTM(100, return_sequences=True, name='bilstm'))(dense_convinp)
    drop1 = Dropout(0.5)(ff_2)
    dense_out = Dense(100, activation='relu')(drop1)

    if use_CRF:
        timedist = TimeDistributed(Dense(n_classes, name='dense'))(dense_out)
        crf = ChainCRF(name="crf1")
        crf_output = crf(timedist)
        print(crf_output)
        print(K.int_shape(crf_output))
        model = Model(inputs=visible, outputs=crf_output)
        adam=optimizers.Adam(lr=0.002)
        model.compile(loss=crf.loss, optimizer=adam, weighted_metrics= ['accuracy'], sample_weight_mode='temporal')
    else:
        timedist = TimeDistributed(Dense(n_classes, activation='softmax'))(dense_out)
        model = Model(inputs=visible, outputs=timedist)
        adam=optimizers.Adam(lr=0.002)
        model.compile(loss='categorical_crossentropy', optimizer=adam, weighted_metrics= ['accuracy'], sample_weight_mode='temporal')
    print(model.summary())
    return model, 'model#'+'#'.join(features_to_use)+'@conv'+'_'.join([str(c) for c in convs])+'@dense_'+str(dense_size)+'@lstm'+str(lstm_size)+'@droplstm'+str(drop_lstm)+'@filtersize_'+str(filter_size)



# the same as CNN
def cnn_multihead(n_classes, convs=[3,5,7], dense_size=1000, features_to_use=['onehot','sequence_profile'], use_CRF=False, filter_size=256):
    '''
    :param max_length:
    :param n_classes:
    :param embedding_layer:
    :param vocab_size:
    :return:
    '''
    visible = Input(shape=(None,408))
    biophysical = slice_tensor(2,0,16,name='biophysicalfeatures')(visible)
    embedding = slice_tensor(2,16,66,name='skipgramembd')(visible)
    onehot = slice_tensor(2,66,87,name='onehot')(visible)
    prof = slice_tensor(2,87,108,name='sequenceprofile')(visible)
    elmo = slice_tensor(2,108,408,name='elmo')(visible)

    input_dict={'sequence_profile':prof, 'onehot':onehot,'embedding':embedding,'elmo':elmo,'biophysical':biophysical}

    # create input
    features=[]
    for feature in features_to_use:
        features.append(input_dict[feature])

    if len(features_to_use)==1:
        conclayers=features
        input=BatchNormalization(name='batchnorm_input') (features[0])
    else:
        input =BatchNormalization(name='batchnorm_input') (concatenate(features))

    multihead = MultiHead([
    Conv1D(filters=256, kernel_size=1, padding='same',kernel_regularizer=regularizers.l2(0.001)),
    Conv1D(filters=256, kernel_size=3, padding='same',kernel_regularizer=regularizers.l2(0.001)),
    Conv1D(filters=256, kernel_size=5, padding='same',kernel_regularizer=regularizers.l2(0.001)),
    Conv1D(filters=256, kernel_size=7, padding='same',kernel_regularizer=regularizers.l2(0.001)),
    Conv1D(filters=256, kernel_size=11, padding='same',kernel_regularizer=regularizers.l2(0.001)),
    Conv1D(filters=256, kernel_size=21, padding='same',kernel_regularizer=regularizers.l2(0.001)),
    ], name='Multi-CNNs', reg_factor=0.001,)(input)

    multihead_res = TimeDistributed(Flatten(), name='Flatten')(multihead)
    
    dropped = Dropout(0.5,name='dropoutonconvs')(multihead_res)
    dense_convinp = Dense(dense_size, activation='relu', name='denseonconvs')(dropped)
    dense_convinpn = BatchNormalization(name='batch_norm_dense') (dense_convinp)
    
    
    if use_CRF:
        timedist = TimeDistributed(Dense(n_classes, name='dense'))(dense_convinpn)
        crf = ChainCRF(name="crf1")
        crf_output = crf(timedist)
        print(crf_output)
        print(K.int_shape(crf_output))
        model = Model(inputs=visible, outputs=crf_output)
        adam=optimizers.Adam(lr=0.001)
        model.compile(loss=crf.loss, optimizer=adam, weighted_metrics= ['accuracy'], sample_weight_mode='temporal')
    else:
        timedist = TimeDistributed(Dense(n_classes, activation='softmax'))(dense_convinpn)
        model = Model(inputs=visible, outputs=timedist)
        adam=optimizers.Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=adam, weighted_metrics= ['accuracy'], sample_weight_mode='temporal')
        
    print(model.summary())
    return model, 'model#'+'#'.join(features_to_use)+'@conv'+'_'.join([str(c) for c in convs])+'@dense_'+str(dense_size)+'@filtersize_'+str(filter_size)

def cnn_baseline(n_classes, convs=[3,5,7], dense_size=1000, features_to_use=['onehot','sequence_profile'], use_CRF=False, filter_size=256):
    '''
    :param max_length:
    :param n_classes:
    :param embedding_layer:
    :param vocab_size:
    :return:
    '''
    visible = Input(shape=(None,408))
    biophysical = slice_tensor(2,0,16,name='biophysicalfeatures')(visible)
    embedding = slice_tensor(2,16,66,name='skipgramembd')(visible)
    onehot = slice_tensor(2,66,87,name='onehot')(visible)
    prof = slice_tensor(2,87,108,name='sequenceprofile')(visible)
    elmo = slice_tensor(2,108,408,name='elmo')(visible)

    input_dict={'sequence_profile':prof, 'onehot':onehot,'embedding':embedding,'elmo':elmo,'biophysical':biophysical}

    # create input
    features=[]
    for feature in features_to_use:
        features.append(input_dict[feature])


    if len(features_to_use)==1:
        conclayers=features
        input=BatchNormalization(name='batchnorm_input') (features[0])
    else:
        input =BatchNormalization(name='batchnorm_input') (concatenate(features))
        conclayers=[input]

    # performing the conlvolutions
    for idx,conv in enumerate(convs):
        idx=str(idx+1)
        conclayers.append(BatchNormalization(name='batch_norm_conv'+idx) (Conv1D(filter_size, conv, activation="relu", padding="same", name='conv'+idx,
                   kernel_regularizer=regularizers.l2(0.001))(input)))

    if len(convs)==0:
        conc = input
    else:
        conc = concatenate(conclayers)

    dropped = Dropout(0.5,name='dropoutonconvs')(conc)
    dense_convinp = Dense(dense_size, activation='relu', name='denseonconvs')(dropped)
    dense_convinpn = BatchNormalization(name='batch_norm_dense') (dense_convinp)
    
    
    if use_CRF:
        timedist = TimeDistributed(Dense(n_classes, name='dense'))(dense_convinpn)
        crf = ChainCRF(name="crf1")
        crf_output = crf(timedist)
        print(crf_output)
        print(K.int_shape(crf_output))
        model = Model(inputs=visible, outputs=crf_output)
        adam=optimizers.Adam(lr=0.001)
        model.compile(loss=crf.loss, optimizer=adam, weighted_metrics= ['accuracy'], sample_weight_mode='temporal')
    else:
        timedist = TimeDistributed(Dense(n_classes, activation='softmax'))(dense_convinpn)
        model = Model(inputs=visible, outputs=timedist)
        adam=optimizers.Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=adam, weighted_metrics= ['accuracy'], sample_weight_mode='temporal')
        
    print(model.summary())
    return model, 'model#'+'#'.join(features_to_use)+'@conv'+'_'.join([str(c) for c in convs])+'@dense_'+str(dense_size)+'@filtersize_'+str(filter_size)


