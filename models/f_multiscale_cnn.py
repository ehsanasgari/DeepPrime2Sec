import numpy as np
np.random.seed(7)
from keras.models import Model
from keras.layers import Masking, Lambda, Dense, LSTM, CuDNNLSTM, Reshape, Flatten, Activation, RepeatVector, Permute, Bidirectional, Embedding, Input, Dropout, concatenate, Masking, Conv1D, \
    multiply, BatchNormalization, merge
from keras.layers.wrappers import TimeDistributed
from models.layers import ChainCRF, slice_tensor
from keras import optimizers
from keras import backend as K
from keras import initializers, regularizers, constraints

def model_f_multiscale_cnn(n_classes, convs=[3,5,7], dense_size=200, drop_lstm=0.5, features_to_use=['onehot','sequence_profile'], use_CRF=False, filter_size=256):
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
    return model, 'model_f_multiscale_cnn#'+'#'.join(features_to_use)+'@conv'+'_'.join([str(c) for c in convs])+'@dense_'+str(dense_size)+'@lstm'+str(lstm_size)+'@droplstm'+str(drop_lstm)+'@filtersize_'+str(filter_size)


