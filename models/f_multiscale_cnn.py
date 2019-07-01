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

def multiscaleCNN(input_layer, gating_layer, filter_size, convs):
    z_t = gating_layer(input_layer)
    conclayers = []
    for idx, conv in enumerate(convs):
        conclayers.append(Conv1D(filter_size, conv, activation="relu", padding="same",
                   kernel_regularizer=regularizers.l2(0.00005))(input_layer))
    conc = concatenate(conclayers)
    output = Lambda(lambda a: z_t*a[0] + (1-z_t)*a[1])([input_layer, conc])
    return output

def model_f_multiscale_cnn(n_classes, convs=[7,9,11], dropout_rate=0.5, features_to_use=['onehot','sequence_profile'], use_CRF=False, filter_size=100):
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

    gating = Dense(len(convs)*filter_size, activation='sigmoid')

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
        conclayers.append(Conv1D(filter_size, conv, activation="relu", padding="same", name='conv'+idx,
                   kernel_regularizer=regularizers.l2(0.00005))(input))

    conc = concatenate(conclayers)

    out_level1 = multiscaleCNN( conc, gating, filter_size, convs)
    out_level1 = Dropout(dropout_rate)(out_level1)

    out_level2 = multiscaleCNN( out_level1, gating, filter_size, convs)
    out_level2 = Dropout(dropout_rate)(out_level2)

    out_level3 = multiscaleCNN( out_level2, gating, filter_size, convs)
    out_level3 = Dropout(dropout_rate)(out_level3)

    dense_out = Dense(len(convs)*filter_size, activation='relu')(out_level3)

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
    return model, 'model_f_multiscale_cnn#'+'#'.join(features_to_use)+'@conv'+'_'.join([str(c) for c in convs])+'@filtersize_'+str(filter_size)
