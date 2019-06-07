
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping
from data_utility.file_utility import FileUtility
from data_utility.labeling_utility import LabelingData
from data_utility.feedgenerator import train_batch_generator_408, validation_batch_generator_408



def training_loop( gpu='1', model, **kwargs):

    # which GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    # read files
    train_file = '../DeepSeq2Sec/data/s8_all_features/train.txt'
    test_file = '../DeepSeq2Sec/data/s8_all_features/test.txt'
    LD = LabelingData(train_file, test_file)
    train_lengths = [int(j) for j in FileUtility.load_list('/'.join(train_file.split('/')[0:-1]) + '/train_length.txt')]
    test_lengths = [int(i) for i in FileUtility.load_list('/'.join(test_file.split('/')[0:-1]) + '/test_length.txt')]

    # train/test batch parameters
    train_batch_size = run_parameters['train_batch_size']
    test_batch_size = run_parameters['test_batch_size']
    patience = run_parameters['patience']
    epochs = run_parameters['epochs']

    # model
    model, params = model(LD.n_classes, model_paramters)

    # output directory
    FileUtility.ensure_dir('results/')
    FileUtility.ensure_dir('results/' + run_parameters['domain_name']+ '/')
    FileUtility.ensure_dir('results/' + run_parameters['domain_name']+ '/' + run_parameters['setting_name'] + '/')
    FileUtility.ensure_dir('results/' + run_parameters['domain_name']+ '/' + run_parameters['setting_name'] + '/' + params + '/')
    full_path = 'results/' + run_parameters['domain_name']+ '/' + run_parameters['setting_name'] + '/' + params + '/'

    # save model
    with open(full_path+ 'config.txt', 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    # check points
    filepath = full_path + "/weights-improvement-{epoch:02d}-{weighted_acc:.3f}-{val_weighted_acc:.3f}.hdf5"

    checkpoint = ModelCheckpoint(filepath, monitor='val_weighted_acc', verbose=1, save_best_only=True, mode='max',
                                 period=1)
    earlystopping = EarlyStopping(monitor='val_weighted_acc', min_delta=0, patience=patience, verbose=0, mode='max',
                                  baseline=None)
    callbacks_list = [checkpoint, earlystopping]

    # calculate the sizes
    steps_per_epoch = len(train_lengths) / train_batch_size if len(train_lengths) % train_batch_size == 0 else int(
        len(train_lengths) / train_batch_size) + 1
    validation_steps = int(len(test_lengths) / test_batch_size) if len(test_lengths) % test_batch_size == 0 else int(
        len(test_lengths) / test_batch_size) + 1

    # feed model
    h = model.fit_generator(train_batch_generator_408(train_batch_size), steps_per_epoch=steps_per_epoch,
                            validation_data=validation_batch_generator_408(test_batch_size),
                            validation_steps=validation_steps,
                            shuffle=False, epochs=epochs, verbose=1, callbacks=callbacks_list)

    # save the history
    FileUtility.save_obj(full_path+'history', h.history)
