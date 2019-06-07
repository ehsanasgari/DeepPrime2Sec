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
from keras_self_attention import SeqSelfAttention as Attention
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras_self_attention import SeqSelfAttention
import keras


class AttentionLayerMasking(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(AttentionLayerMasking, self).__init__(**kwargs)


    def build(self, input_shape):
        input_embedding_dim=input_shape[-1]
        
        self.kernel = self.add_weight(name='kernel', 
                            shape=(input_embedding_dim,1),
                            initializer='uniform',
                            trainable=True)
        super(AttentionLayerMasking, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        
        # dot product 
        x=K.dot(x, self.kernel)
        # exponentiate
        x=K.exp(x)
        
        # zero out elements that are masked
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = K.expand_dims(mask, axis=-1)
            x = x * mask
        
        # normalize by sum
        x /= K.sum(x, axis=1, keepdims=True)
        x=K.squeeze(x, axis=2)

        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])

def att_david(n_classes, convs=[3,5,7], dense_size=200, lstm_size=400, drop_lstm=0.5, features_to_use=['onehot','sequence_profile'], use_CRF=False, filter_size=256):
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

    if drop_lstm > 0:
        drop0 = Dropout(drop_lstm,name='dropoutonconvs')(conc)
        dense_convinp = Dense(dense_size, activation='relu', name='denseonconvs')(drop0)
    else:
        dense_convinp = Dense(dense_size, activation='relu', name='denseonconvs')(conc)

    dense_convinpn=BatchNormalization(name='batch_norm_dense') (dense_convinp)

    lstm = Bidirectional(CuDNNLSTM(lstm_size, return_sequences=True, name='bilstm'), merge_mode='concat')(dense_convinpn)

    # first let's transform each word embedding into a new vector to use for measuring its importance
    attention_key_dim=300
    attention_input=Dense(attention_key_dim, activation='tanh')(lstm)

    # next we'll pass those transformed inputs through an attention layer, getting back a normalized
    # attention value a_i for each token i; \forall i, 0 <= a_i <= 1; for a document with N words, 
    # \sum_{i=0}^N a_i = 1
    
    attention_output = AttentionLayerMasking(42, name="attention")(attention_input)
    
    # now let's multiply those attention weights by original inputs to get a weighted average over them
    sent_representation = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=1), name='dot')([attention_output,lstm])    
    
    if use_CRF:
        timedist = TimeDistributed(Dense(n_classes, name='dense'))(sent_representation)
        crf = ChainCRF(name="crf1")
        crf_output = crf(timedist)
        print(crf_output)
        print(K.int_shape(crf_output))
        model = Model(inputs=visible, outputs=crf_output)
        adam=optimizers.Adam(lr=0.001)
        model.compile(loss=crf.loss, optimizer=adam, weighted_metrics= ['accuracy'], sample_weight_mode='temporal')
    else:
        timedist = TimeDistributed(Dense(n_classes, activation='softmax'))(sent_representation)
        model = Model(inputs=visible, outputs=timedist)
        adam=optimizers.Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=adam, weighted_metrics= ['accuracy'], sample_weight_mode='temporal')
    print(model.summary())
    return model, 'model#'+'#'.join(features_to_use)+'@conv'+'_'.join([str(c) for c in convs])+'@dense_'+str(dense_size)+'@lstm'+str(lstm_size)+'@droplstm'+str(drop_lstm)+'@filtersize_'+str(filter_size)

    
    


class SeqWeightedAttention(keras.layers.Layer):
    r"""Y = \text{softmax}(XW + b) X
    See: https://arxiv.org/pdf/1708.00524.pdf
    """

    def __init__(self, use_bias=True, return_attention=False, **kwargs):
        super(SeqWeightedAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.use_bias = use_bias
        self.return_attention = return_attention
        self.W, self.b = None, None

    def get_config(self):
        config = {
            'use_bias': self.use_bias,
            'return_attention': self.return_attention,
        }
        base_config = super(SeqWeightedAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.W = self.add_weight(shape=(int(input_shape[2]), 1),
                                 name='{}_W'.format(self.name),
                                 initializer=keras.initializers.get('uniform'))
        if self.use_bias:
            self.b = self.add_weight(shape=(1,),
                                     name='{}_b'.format(self.name),
                                     initializer=keras.initializers.get('zeros'))
        super(SeqWeightedAttention, self).build(input_shape)

    def call(self, x, mask=None):
        logits = K.dot(x, self.W)
        if self.use_bias:
            logits += self.b
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return input_shape[0], output_len

    def compute_mask(self, _, input_mask=None):
        if self.return_attention:
            return [None, None]
        return None

    @staticmethod
    def get_custom_objects():
        return {'SeqWeightedAttention': SeqWeightedAttention}
    
    
def normal_attention_weighted(n_classes, convs=[3,5,7], dense_size=200, lstm_size=400, drop_lstm=0.5, features_to_use=['onehot','sequence_profile'], use_CRF=False, filter_size=256):
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

    if drop_lstm > 0:
        drop0 = Dropout(drop_lstm,name='dropoutonconvs')(conc)
        dense_convinp = Dense(dense_size, activation='relu', name='denseonconvs')(drop0)
    else:
        dense_convinp = Dense(dense_size, activation='relu', name='denseonconvs')(conc)

    dense_convinpn=BatchNormalization(name='batch_norm_dense') (dense_convinp)

    lstm = Bidirectional(CuDNNLSTM(lstm_size, return_sequences=True, name='bilstm'), merge_mode='concat')(dense_convinpn)
    sent_representation = SeqWeightedAttention(use_bias=True, return_attention=False)(lstm)    
    
    if use_CRF:
        timedist = TimeDistributed(Dense(n_classes, name='dense'))(sent_representation)
        crf = ChainCRF(name="crf1")
        crf_output = crf(timedist)
        print(crf_output)
        print(K.int_shape(crf_output))
        model = Model(inputs=visible, outputs=crf_output)
        adam=optimizers.Adam(lr=0.001)
        model.compile(loss=crf.loss, optimizer=adam, weighted_metrics= ['accuracy'], sample_weight_mode='temporal')
    else:
        timedist = TimeDistributed(Dense(n_classes, activation='softmax'))(sent_representation)
        model = Model(inputs=visible, outputs=timedist)
        adam=optimizers.Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=adam, weighted_metrics= ['accuracy'], sample_weight_mode='temporal')
    print(model.summary())
    return model, 'model#'+'#'.join(features_to_use)+'@conv'+'_'.join([str(c) for c in convs])+'@dense_'+str(dense_size)+'@lstm'+str(lstm_size)+'@droplstm'+str(drop_lstm)+'@filtersize_'+str(filter_size)

def att_baseline(n_classes, convs=[3,5,7], dense_size=200, lstm_size=400, drop_lstm=0.5, features_to_use=['onehot','sequence_profile'], use_CRF=False, filter_size=256):
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

    if drop_lstm > 0:
        drop0 = Dropout(drop_lstm,name='dropoutonconvs')(conc)
        dense_convinp = Dense(dense_size, activation='relu', name='denseonconvs')(drop0)
    else:
        dense_convinp = Dense(dense_size, activation='relu', name='denseonconvs')(conc)

    dense_convinpn=BatchNormalization(name='batch_norm_dense') (dense_convinp)

    lstm = Bidirectional(CuDNNLSTM(lstm_size, return_sequences=True, name='bilstm'), merge_mode='concat')(dense_convinpn)
    sent_representation = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_ADD,
                           name='Attention')(lstm)    
    if use_CRF:
        timedist = TimeDistributed(Dense(n_classes, name='dense'))(sent_representation)
        crf = ChainCRF(name="crf1")
        crf_output = crf(timedist)
        print(crf_output)
        print(K.int_shape(crf_output))
        model = Model(inputs=visible, outputs=crf_output)
        adam=optimizers.Adam(lr=0.001)
        model.compile(loss=crf.loss, optimizer=adam, weighted_metrics= ['accuracy'], sample_weight_mode='temporal')
    else:
        timedist = TimeDistributed(Dense(n_classes, activation='softmax'))(sent_representation)
        model = Model(inputs=visible, outputs=timedist)
        adam=optimizers.Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=adam, weighted_metrics= ['accuracy'], sample_weight_mode='temporal')
    print(model.summary())
    return model, 'model#'+'#'.join(features_to_use)+'@conv'+'_'.join([str(c) for c in convs])+'@dense_'+str(dense_size)+'@lstm'+str(lstm_size)+'@droplstm'+str(drop_lstm)+'@filtersize_'+str(filter_size)


def att_local(n_classes, convs=[3,5,7], dense_size=200, lstm_size=400, drop_lstm=0.5, features_to_use=['onehot','sequence_profile'], use_CRF=False, filter_size=256, context=15):
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

    if drop_lstm > 0:
        drop0 = Dropout(drop_lstm,name='dropoutonconvs')(conc)
        dense_convinp = Dense(dense_size, activation='relu', name='denseonconvs')(drop0)
    else:
        dense_convinp = Dense(dense_size, activation='relu', name='denseonconvs')(conc)

    dense_convinpn=BatchNormalization(name='batch_norm_dense') (dense_convinp)

    lstm = Bidirectional(CuDNNLSTM(lstm_size, return_sequences=True, name='bilstm'), merge_mode='concat')(dense_convinpn)
    sent_representation = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_ADD,
                           attention_width=context,
                           kernel_regularizer=keras.regularizers.l2(1e-4),
                           bias_regularizer=keras.regularizers.l1(1e-4),
                           attention_regularizer_weight=1e-4,
                           attention_activation='sigmoid',
                           name='Attention')(lstm)    
    if use_CRF:
        timedist = TimeDistributed(Dense(n_classes, name='dense'))(sent_representation)
        crf = ChainCRF(name="crf1")
        crf_output = crf(timedist)
        print(crf_output)
        print(K.int_shape(crf_output))
        model = Model(inputs=visible, outputs=crf_output)
        adam=optimizers.Adam(lr=0.001)
        model.compile(loss=crf.loss, optimizer=adam, weighted_metrics= ['accuracy'], sample_weight_mode='temporal')
    else:
        timedist = TimeDistributed(Dense(n_classes, activation='softmax'))(sent_representation)
        model = Model(inputs=visible, outputs=timedist)
        adam=optimizers.Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=adam, weighted_metrics= ['accuracy'], sample_weight_mode='temporal')
    print(model.summary())
    return model, 'model#'+'#'.join(features_to_use)+'@conv'+'_'.join([str(c) for c in convs])+'@dense_'+str(dense_size)+'@lstm'+str(lstm_size)+'@droplstm'+str(drop_lstm)+'@context'+str(context)+'@filtersize_'+str(filter_size)


