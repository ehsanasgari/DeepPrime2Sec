import numpy as np
np.random.seed(7)
from keras.models import Model
from keras.layers import Masking, Lambda, Dense, LSTM, CuDNNLSTM, Reshape, Flatten, Activation, RepeatVector, Permute, Bidirectional, Embedding, Input, Dropout, concatenate, Masking, Conv1D, \
    multiply, BatchNormalization, merge
from keras.layers.wrappers import TimeDistributed
from layers.crf import ChainCRF
from layers.utility import slice_tensor
from keras import optimizers
from keras import backend as K
from keras import initializers, regularizers, constraints

def model_e_cnn(n_classes, convs=[3,5,7], dense_size=1000, features_to_use=['onehot','sequence_profile'], use_CRF=False, filter_size=256):
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
    return model, 'model_e_cnn#'+'#'.join(features_to_use)+'@conv'+'_'.join([str(c) for c in convs])+'@dense_'+str(dense_size)+'@filtersize_'+str(filter_size)
