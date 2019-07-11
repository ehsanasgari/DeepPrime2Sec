from keras import regularizers
from keras.layers import Lambda, concatenate, Conv1D


def slice_tensor(dimension, start, end, name='sliced_layer'):
    '''
    :param dimension:
    :param start:
    :param end:
    :return:
    '''

    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]

    return Lambda(func, name=name)


def multiscale_CNN(input_layer, gating_layer, filter_size, convs, kernel_regularizer=0.00005):
    '''
    :param input_layer:
    :param gating_layer:
    :param filter_size:
    :param convs:
    :param kernel_regularizer:
    :return:
    '''
    z_t = gating_layer(input_layer)
    conclayers = []
    for idx, conv in enumerate(convs):
        conclayers.append(Conv1D(filter_size, conv, activation="relu", padding="same",
                                 kernel_regularizer=regularizers.l2(kernel_regularizer))(input_layer))
    conc = concatenate(conclayers)
    output = Lambda(lambda a: z_t * a[0] + (1 - z_t) * a[1])([input_layer, conc])
    return output
