import numpy as np

from utility.file_utility import FileUtility


def train_batch_generator_408(batch_size=64):
    '''
    :param batch_size:
    :return:
    '''
    start_idx = 0
    train_lengths = [int(j) for j in FileUtility.load_list(
        'datasets/train_length.txt')]
    X_train = np.load('datasets/X_train_408.npy')
    Y_train = np.array(
        np.load('datasets/train_mat_Y.npy'))
    while True:
        if not start_idx < len(train_lengths):
            start_idx = 0
        X = X_train[start_idx:(min(start_idx + batch_size, len(train_lengths))),
            0:train_lengths[min(start_idx + batch_size, len(train_lengths)) - 1]]
        Y = Y_train[start_idx:(min(start_idx + batch_size, len(train_lengths))),
            0:train_lengths[min(start_idx + batch_size, len(train_lengths)) - 1], :]

        W = []
        for idx in range(start_idx, (min(start_idx + batch_size, len(train_lengths)))):
            W.append([1 if l < train_lengths[idx] else 0 for l in
                      range(0, train_lengths[min(start_idx + batch_size, len(train_lengths)) - 1])])

        start_idx += batch_size

        yield X, Y, np.array(W)


def validation_batch_generator_408(batch_size=100):
    '''
    :param batch_size:
    :return:
    '''
    test_lengths = [int(i) for i in FileUtility.load_list(
        'datasets/test_length.txt')]
    X_test = np.load('datasets/X_test_408.npy')
    Y_test = np.array(
        np.load('datasets/test_mat_Y.npy'))
    start_idx = 0
    while True:
        if not start_idx < len(test_lengths):
            start_idx = 0
        X = X_test[start_idx:(min(start_idx + batch_size, len(test_lengths))),
            0:test_lengths[min(start_idx + batch_size, len(test_lengths)) - 1]]
        Y = Y_test[start_idx:(min(start_idx + batch_size, len(test_lengths))),
            0:test_lengths[min(start_idx + batch_size, len(test_lengths)) - 1], :]
        W = []
        for idx in range(start_idx, (min(start_idx + batch_size, len(test_lengths)))):
            W.append([1 if l < test_lengths[idx] else 0 for l in
                      range(0, test_lengths[min(start_idx + batch_size, len(test_lengths)) - 1])])

        start_idx += batch_size
        yield X, Y, np.array(W)


def validation_batches_fortest_408(batchsize=100):
    '''
    :param batchsize:
    :return:
    '''
    test_lengths = [int(i) for i in FileUtility.load_list(
        'datasets/test_length.txt')]
    X_test = np.load('datasets/X_test_408.npy')
    Y_test = np.array(
        np.load('datasets/test_mat_Y.npy'))
    start_idx = 0
    while start_idx < len(test_lengths):
        X = X_test[start_idx:(min(start_idx + batchsize, len(test_lengths))),
            0:test_lengths[min(start_idx + batchsize, len(test_lengths)) - 1]]
        Y = Y_test[start_idx:(min(start_idx + batchsize, len(test_lengths))),
            0:test_lengths[min(start_idx + batchsize, len(test_lengths)) - 1], :]
        W = []
        for idx in range(start_idx, (min(start_idx + batchsize, len(test_lengths)))):
            W.append([1 if l < test_lengths[idx] else 0 for l in
                      range(0, test_lengths[min(start_idx + batchsize, len(test_lengths)) - 1])])

        start_idx += batchsize
        yield X, Y, np.array(W)
