import numpy as np
from old.data_utility import FileUtility
import tqdm
from sklearn.preprocessing import normalize

def train_batch_generator_aug408(batchsize=64):
    '''
    :param batchsize:
    :return:
    '''
    start_idx = 0
    train_lengths = [int(j) for j in FileUtility.load_list(
        '/home/easgari/projects/DeepSeq2Sec/data/s8_all_features/augmented_lengths.txt')]

    Y_train=np.load('/home/easgari/projects/DeepSeq2Sec/data/s8_all_features/Y_augmented_0.npy')
    for idx in tqdm.tqdm(range(1,10)):
        Y_train=np.concatenate([Y_train, np.load('/home/easgari/projects/DeepSeq2Sec/data/s8_all_features/Y_augmented_'+str(idx)+'.npy')])

    X_train=np.load('/home/easgari/projects/DeepSeq2Sec/data/s8_all_features/X_augmented_0.npy')
    for idx in tqdm.tqdm(range(1,10)):
        X_train=np.concatenate([X_train, np.load('/home/easgari/projects/DeepSeq2Sec/data/s8_all_features/X_augmented_'+str(idx)+'.npy')])


    while True:
        if not start_idx < len(train_lengths):
            start_idx = 0
        X = X_train[start_idx:(min(start_idx + batchsize, len(train_lengths))),
            0:train_lengths[min(start_idx + batchsize, len(train_lengths)) - 1]]
        Y = Y_train[start_idx:(min(start_idx + batchsize, len(train_lengths))),
            0:train_lengths[min(start_idx + batchsize, len(train_lengths)) - 1], :]

        W = []
        for idx in range(start_idx, (min(start_idx + batchsize, len(train_lengths)))):
            W.append([1 if l < train_lengths[idx] else 0 for l in
                      range(0, train_lengths[min(start_idx + batchsize, len(train_lengths)) - 1])])

        start_idx += batchsize

        yield X, Y, np.array(W)


def train_batch_generator_408(batchsize=64):
    '''
    :param batchsize:
    :return:
    '''
    start_idx = 0
    train_lengths = [int(j) for j in FileUtility.load_list(
        '/home/easgari/projects/DeepSeq2Sec/data/s8_all_features/train_length.txt')]
    X_train = np.load('/home/easgari/projects/DeepSeq2Sec/data/s8_all_features/X_train_408.npy')
    Y_train = np.array(
        np.load('/home/easgari/projects/DeepSeq2Sec/data/s8_all_features/train_mat_Y.npy'))
    while True:
        if not start_idx < len(train_lengths):
            start_idx = 0
        X = X_train[start_idx:(min(start_idx + batchsize, len(train_lengths))),
            0:train_lengths[min(start_idx + batchsize, len(train_lengths)) - 1]]
        Y = Y_train[start_idx:(min(start_idx + batchsize, len(train_lengths))),
            0:train_lengths[min(start_idx + batchsize, len(train_lengths)) - 1], :]

        W = []
        for idx in range(start_idx, (min(start_idx + batchsize, len(train_lengths)))):
            W.append([1 if l < train_lengths[idx] else 0 for l in
                      range(0, train_lengths[min(start_idx + batchsize, len(train_lengths)) - 1])])

        start_idx += batchsize

        yield X, Y, np.array(W)


def validation_batch_generator_408(batchsize=100):
    '''
    :param batchsize:
    :return:
    '''
    test_lengths = [int(i) for i in FileUtility.load_list(
        '/home/easgari/projects/DeepSeq2Sec/data/s8_all_features/test_length.txt')]
    X_test = np.load('/home/easgari/projects/DeepSeq2Sec/data/s8_all_features/X_test_408.npy')
    Y_test = np.array(
        np.load('/home/easgari/projects/DeepSeq2Sec/data/s8_all_features/test_mat_Y.npy'))
    start_idx = 0
    while True:
        if not start_idx < len(test_lengths):
            start_idx = 0
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

def train_batch_generator(batchsize=64):
    '''
    :param batchsize:
    :return:
    '''
    start_idx = 0
    train_lengths = [int(j) for j in FileUtility.load_list(
        '/home/easgari/projects/DeepSeq2Sec/data/s8_features/train_length.txt')]
    X_train = np.load('/home/easgari/projects/DeepSeq2Sec/data/s8_features/X_train_ext.npy')
    Y_train = np.array(
        np.load('/home/easgari/projects/DeepSeq2Sec/data/s8_features/train_mat_Y.npy'))
    while True:
        if not start_idx < len(train_lengths):
            start_idx = 0
        X = X_train[start_idx:(min(start_idx + batchsize, len(train_lengths))),
            0:train_lengths[min(start_idx + batchsize, len(train_lengths)) - 1]]
        Y = Y_train[start_idx:(min(start_idx + batchsize, len(train_lengths))),
            0:train_lengths[min(start_idx + batchsize, len(train_lengths)) - 1], :]

        W = []
        for idx in range(start_idx, (min(start_idx + batchsize, len(train_lengths)))):
            W.append([1 if l < train_lengths[idx] else 0 for l in
                      range(0, train_lengths[min(start_idx + batchsize, len(train_lengths)) - 1])])

        start_idx += batchsize

        yield X, Y, np.array(W)


def validation_batch_generator(batchsize=10):
    '''
    :param batchsize:
    :return:
    '''
    test_lengths = [int(i) for i in FileUtility.load_list(
        '/home/easgari/projects/DeepSeq2Sec/data/s8_features/test_length.txt')]
    X_test = np.load('/home/easgari/projects/DeepSeq2Sec/data/s8_features/X_test_ext.npy')
    Y_test = np.array(
        np.load('/home/easgari/projects/DeepSeq2Sec/data/s8_features/test_mat_Y.npy'))
    start_idx = 0
    while True:
        if not start_idx < len(test_lengths):
            start_idx = 0
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

def train_batch_generator_profile_predict(batchsize=64):
    '''
    :param batchsize:
    :return:
    '''
    start_idx = 0
    train_lengths = [int(j) for j in FileUtility.load_list(
        '/home/easgari/projects/DeepSeq2Sec/data/s8_all_features/train_length.txt')]
    X_train = np.load('/home/easgari/projects/DeepSeq2Sec/data/s8_all_features/X_train_408.npy')

    while True:
        if not start_idx < len(train_lengths):
            start_idx = 0
        X = X_train[start_idx:(min(start_idx + batchsize, len(train_lengths))),
            0:train_lengths[min(start_idx + batchsize, len(train_lengths)) - 1],]
        Y = X_train[start_idx:(min(start_idx + batchsize, len(train_lengths))),
            0:train_lengths[min(start_idx + batchsize, len(train_lengths)) - 1], 87:108]

        W = []
        for idx in range(start_idx, (min(start_idx + batchsize, len(train_lengths)))):
            W.append([1 if l < train_lengths[idx] else 0 for l in
                      range(0, train_lengths[min(start_idx + batchsize, len(train_lengths)) - 1])])

        start_idx += batchsize

        yield X, Y, np.array(W)

def validation_batch_generator_profile_predict(batchsize=100):
    '''
    :param batchsize:
    :return:
    '''
    test_lengths = [int(i) for i in FileUtility.load_list(
        '/home/easgari/projects/DeepSeq2Sec/data/s8_all_features/test_length.txt')]
    X_test = np.load('/home/easgari/projects/DeepSeq2Sec/data/s8_all_features/X_test_408.npy')
    start_idx = 0
    while True:
        if not start_idx < len(test_lengths):
            start_idx = 0
        X = X_test[start_idx:(min(start_idx + batchsize, len(test_lengths))),
            0:test_lengths[min(start_idx + batchsize, len(test_lengths)) - 1]]
        Y = X_test[start_idx:(min(start_idx + batchsize, len(test_lengths))),
            0:test_lengths[min(start_idx + batchsize, len(test_lengths)) - 1], 87:108]
        W = []
        for idx in range(start_idx, (min(start_idx + batchsize, len(test_lengths)))):
            W.append([1 if l < test_lengths[idx] else 0 for l in
                      range(0, test_lengths[min(start_idx + batchsize, len(test_lengths)) - 1])])

        start_idx += batchsize
        yield X, Y, np.array(W)
