import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import scipy

from keras.callbacks import ModelCheckpoint, EarlyStopping
from utility.file_utility import FileUtility
from utility.labeling_utility import LabelingData
from utility.feed_generation_utility import train_batch_generator_408, validation_batch_generator_408, validation_batches_fortest_408
from utility.vis_utility import create_mat_plot
import tqdm
import numpy as np
import itertools
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from fpdf import FPDF, HTMLMixin
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import matplotlib

class MyFPDF(FPDF, HTMLMixin):
    pass

# predefined models
from models.a_cnn_bilstm import model_a_cnn_bilstm
from models.b_cnn_bilstm_highway import model_b_cnn_bilstm_highway
from models.c_cnn_bilstm_crf import model_c_cnn_bilstm_crf
from models.d_cnn_bilstm_attention import model_d_cnn_bilstm_attention
from models.e_cnn import model_e_cnn
from models.f_multiscale_cnn import model_f_multiscale_cnn

def training_loop(**kwargs):
    run_parameters = kwargs['run_parameters']
    model_paramters = kwargs['model_paramters']
    model = eval(kwargs['deep_learning_model'])

    # which GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = str(run_parameters['gpu'])

    # read files
    train_file = 'datasets/train.txt'
    test_file = 'datasets/test.txt'
    LD = LabelingData(train_file, test_file)
    train_lengths = [int(j) for j in FileUtility.load_list('/'.join(train_file.split('/')[0:-1]) + '/train_length.txt')]
    test_lengths = [int(i) for i in FileUtility.load_list('/'.join(test_file.split('/')[0:-1]) + '/test_length.txt')]

    # train/test batch parameters
    train_batch_size = run_parameters['train_batch_size']
    test_batch_size = run_parameters['test_batch_size']
    patience = run_parameters['patience']
    epochs = run_parameters['epochs']

    # model
    model, params = model(LD.n_classes, **model_paramters)

    # output directory
    FileUtility.ensure_dir('results/')
    FileUtility.ensure_dir('results/' + run_parameters['domain_name'] + '/')
    FileUtility.ensure_dir('results/' + run_parameters['domain_name'] + '/' + run_parameters['setting_name'] + '/')
    FileUtility.ensure_dir(
        'results/' + run_parameters['domain_name'] + '/' + run_parameters['setting_name'] + '/' + params + '/')
    full_path = 'results/' + run_parameters['domain_name'] + '/' + run_parameters['setting_name'] + '/' + params + '/'

    # save model
    with open(full_path + 'config.txt', 'w') as fh:
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
    FileUtility.save_obj(full_path + 'history', h.history)


    # Analysis of the performance
    pred_test = [(model.predict_on_batch(x),y,w) for x,y,w in tqdm.tqdm(validation_batches_fortest_408(1))]

    acc_test, conf_mat, conf_mat_column_mapping, contingency_metric, chi2_res_pval, gtest_res_pval = generate_report(full_path, pred_test, run_parameters['domain_name'], run_parameters['setting_name'])



def generate_report(full_path, pred_test, domain, setting):
    '''
    :param pred_test: test results
    :return:
    '''
    # Error location analysis
    error_edge=0
    error_NOTedge=0
    correct_edge=0
    correct_NOTedge=0

    all_pred = []
    all_true = []

    for i in tqdm.tqdm(range(0,514)):
        pred=np.array([np.argmax(x, axis=1) for x in pred_test[i][0]])
        true=np.array([np.argmax(x, axis=1) for x in pred_test[i][1]])
        all_pred = all_pred + pred.tolist()
        all_true = all_true + true.tolist()
        diff=np.diff(true)
        errors = [y for x,y in np.argwhere(pred!=true)]
        corrects = list(set(list(range(len(pred[0]))))-set(errors))
        edges_edge  = [y for x,y in np.argwhere(diff!=0)]
        edges_before = [x-1 for x in edges_edge if x-1>=0]
        edges_after = [x+1 for x in edges_edge if x+1<len(pred[0])]
        edges = list(set(edges_edge + edges_before + edges_after))
        # contingency matrix
        error_edge = error_edge+len(list(set(errors).intersection(edges)))
        error_NOTedge = error_NOTedge+len(list(set(errors)-set(edges)))

        correct_edge = correct_edge+len(list(set(corrects).intersection(edges)))
        correct_NOTedge = correct_NOTedge+len(list(set(corrects)-set(edges)))

    all_pred = list(itertools.chain(*all_pred))
    all_true = list(itertools.chain(*all_true))

    acc_test = accuracy_score(all_true, all_pred)
    f1_macro = f1_score(all_true, all_pred, average='macro')
    f1_micro = f1_score(all_true, all_pred, average='micro')

    conf_mat = confusion_matrix(all_true, all_pred, labels=list(range(1,9)))
    conf_mat_column_mapping = {3: 'E (Beta sheet)', 4: 'G (3-10 Helix)', 2: 'B (Beta bridge)', 6: 'H (Alpha helix)', 8: 'T (Turn)', 1: 'L (Loop)', 7: 'S (Bend)', 5: 'I (Pi Helix)'}

    contingency_metric = [[error_edge, error_NOTedge],[correct_edge, correct_NOTedge]]

    # Chi2 test
    chi2_res = scipy.stats.chi2_contingency([[error_edge, error_NOTedge],[correct_edge, correct_NOTedge]], correction=True)
    chi2_res_pval = chi2_res[1]

    #log-likelihood ratio (i.e. the “G-test”)
    gtest_res = scipy.stats.chi2_contingency([[error_edge, error_NOTedge],[correct_edge, correct_NOTedge]], lambda_="log-likelihood", correction=True)
    gtest_res_pval = gtest_res[1]
    #https://stackoverflow.com/questions/51864730/python-what-is-the-process-to-create-pdf-reports-with-charts-from-a-db

    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    create_mat_plot(conf_mat,[conf_mat_column_mapping[x] for x in list(range(1,9))], 'Confusion matrix of protein secondary structure prediction', full_path+'confusion'+F"{domain}_{setting}",'Predicted Label', 'True Label' ,filetype='png', annot=False, cmap=cmap )


    pdf = MyFPDF()
    pdf.add_page()
    pdf.set_xy(0, 0)

    html = F"""
    
    <h2>DeepPrime2Sec Report on Protein Secondary Structure Prediction</h2>
    <h3>Experiment name: {domain} - {setting} </h3>
    <hr/>
    
    <H3 align="left">The performance on CB513</H3>
    <h4>Report on the accuracy</h4>
    
    <table border="1" align="center" width="70%">
    <thead><tr><th width="30%">Test-set Accuray</th><th width="30%">Test-set micro F1</th><th width="30%">Test-set macro F1</th></tr></thead>
    <tbody>
    <tr><td>{round(acc_test,3)}</td><td>{round(f1_micro,3)}</td><td>{round(f1_macro,3)}</td></tr>
    </tbody>
    </table>

    <hr/>
    
    <h4>Confusion matrix</h4>
    
    
    """

    pdf.write_html(html)
    pdf.image(full_path+'confusion'+F"{domain}_{setting}"+'.png', x = 50, y = None, w = 100, h = 0, type = '', link = '')

    html=F"""
    <center>
    <image src='confusion{domain}_{setting}.png'/>
    </center>
    
    <hr/>
    
    <h4>Error analysis</h4>
    
    <h5>Contingency table for location analysis of the misclassified amino acids</h5>
    <table border="1" align="center" width="100%">
    <thead><tr><th width="30%">\</th><th width="30%">Located at the PSS transition</th><th width="30%">NOT Located at the PSS transition</th></tr></thead>
    <tbody>
    <tr><td><b>Miss-classified</b></td><td>{error_edge}</td><td>{error_NOTedge}</td></tr>
    <tr><td><b>Truely classified</b></td><td>{correct_edge}</td><td>{correct_NOTedge}</td></tr>
    </tbody>
    </table>
    <br/>
    <b>P-value for Chi-square test</b> = {chi2_res_pval}
    <br/>
    <b>P-value for G-test</b> = {gtest_res_pval}
    
    <hr/>
    <br/>
    <br/>
    <br/>
    
    <h4>Learning curve</h4>
    """
    pdf.write_html(html)

    # learning curve
    history_dict=FileUtility.load_obj(full_path+'history.pickle')
    plt.clf()
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    matplotlib.rcParams["axes.edgecolor"] = "black"
    matplotlib.rcParams["axes.linewidth"] = 0.6
    plt.plot(epochs, loss_values, 'ro', label='Loss for train set')
    plt.plot(epochs, val_loss_values, 'b', label='Loss for test set')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc=1, prop={'size': 8},ncol=1, edgecolor='black', facecolor='white', frameon=True)
    plt.title('Loss with respect to the number of epochs for train and test sets')
    plt.savefig(full_path + 'learning_curve'+F"{domain}_{setting}"+'.png', dpi=300)
    pdf.image(full_path + 'learning_curve'+F"{domain}_{setting}"+'.png', x = 50, y = None, w = 100, h = 0, type = '', link = '')


    pdf.output(full_path+'final_report.pdf', 'F')

    return acc_test, conf_mat, conf_mat_column_mapping, contingency_metric, chi2_res_pval, gtest_res_pval
