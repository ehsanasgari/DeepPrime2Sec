# DeepPrime2Sec

<a name="tableofcontent"/>

## Table of Content

[1. Summary](#Summary)

[2. Installation](#Installation)

[3. Running Configuration](#Configuration)

[3.1 Features](#Features)

[3.2 Training parameters](#Training)

[3.3 Model specific parameters](#Models)

[4. Output](##Output)

<hr/>

# Summary
<a name="Summary"/>

DeepPrime2Seq is developed deep learning-based prediction of protein secondary structure from the protein primary sequence.
It facilitate the function of different features in this task, including one-hot vectors, biophysical features,
protein sequence embedding (ProtVec), deep contextualized embedding (known as ELMo), and the Position Specific Scoring Matrix (PSSM).

In addition to the role of features, it allows for the evaluation of various deep learning architectures including the following models/mechanisms and
certain combinations: Bidirectional Long Short-Term Memory (BiLSTM), convolutional neural network (CNN), highway connections,
attention mechanism, recurrent neural random fields, and gated multi-scale CNN.

Our results suggest that PSSM concatenated to one-hot vectors are the most important features for the task of secondary structure prediction.
Utilizing the CNN-BiLSTM network, we achieved an accuracy of 69.9% and 70.4%  using ensemble top-k models, for 8-class of protein secondary structure on the CB513 dataset, the most challenging dataset for protein secondary structure prediction.

```
@article {Asgari705426,
	author = {Asgari, Ehsaneddin and Poerner, Nina and McHardy, Alice C. and Mofrad, Mohammad R.K.},
	title = {DeepPrime2Sec: Deep Learning for Protein Secondary Structure Prediction from the Primary Sequences},
	elocation-id = {705426},
	year = {2019},
	doi = {10.1101/705426},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2019/07/18/705426},
	eprint = {https://www.biorxiv.org/content/early/2019/07/18/705426.full.pdf},
	journal = {bioRxiv}
}
```


Through error analysis on the best performing model, we showed that the misclassification is significantly more common at positions that undergo secondary structure transitions, which is most likely due to the inaccurate assignments of the secondary structure at the boundary regions. Notably, when ignoring amino acids at secondary structure transitions in the evaluation, the accuracy increases to 90.3%. Furthermore, the best performing model mostly mistook similar structures for one another, indicating that the deep learning model inferred high-level information on the secondary structure.


DeepPrime2Sec and the used datasets are available here under the Apache 2 license.

Return to the [table of content ↑](#tableofcontent).

<a name="Installation"/>

# Installation

## Pip installation


In order to install the required libraries for running DeepPrime2Sec use the following command:

```
pip install installations/requirements.txt
```

OR you may use conda installation.

## Conda installation

In order to install the required libraries for running DeepPrime2Sec use the following conda command:

```
conda create --name deepprime2sec --file installations/deepprime2sec.yml
```

Subsequently, you need to activate the created virtual environment before running:

```
source activate deepprime2sec
```

## Download the training files


Before running the software make sure to download the traning dataset (which was too large for git) from the following file
and extract them and copy them to the `dataset` directory.

```
http://deepbio.info/proteomics/datasets/deepprime2sec/train_files.tar.gz
```


Return to the [table of content ↑](#tableofcontent).

<hr/>


<a name="Configuration"/>
# Running Configuration

### Running example

In order to run the DeepPrime2Sec, you can simply use the following command.
Every details on different deep learning models: architecture, hyper parameter, training parameters, will be provided in the yaml config file.
Here we detail how this file should be created. Examples are also provided in `sample_configs/*.yaml`.

```
python deepprime2sec.py --config sample_configs/model_a.yaml
```

<a name="Features"/>
# Features to use


We experiment on five sets of protein features to understand what are essential features for the task of protein secondary structure prediction. Although in 1999, PSSM was reported as an important feature to the secondary structure prediction (Jones et al, 1999),
this was still unclear whether recently introduced distributed representations can outperform PSSM in such a task. For a systematic comparison, the features detailed as follows are used:

<ul>
<li> <b>One-hot vector representation (length: 21) --- onehot</b>: vector representation indicating which amino acid exists at each specific position, where each index in the vector indicates the presence or absence of that amino acid.</li>
<li> <b>ProtVec embedding (length: 50) --- protvec</b>: representation trained using Skip-gram neural network on protein amino acid sequences (ProtVec). The only difference would be character-level training instead of n-gram based training. </li>
<li> <b>Contextualized embedding (length: 300) --- elmo</b>: we use the contextualized embedding of the amino acids trained in the course of language modeling, known as ELMo, as a new feature for the secondary structure task. Contextualized embedding is the concatenation of the hidden states of a deep bidirectional language model. The main difference between ProtVec embedding and ELMO embedding is that the ProtVec embedding for a given amino acid or amino acid k-mer is fixed and the representation would be the same in different sequences. However, the contextualized embedding, as it is clear from its name, is an embedding of word changing based on its context. We train ELMo embedding of amino acids using UniRef50 dataset in the dimension size of 300.</li>
<li> <b>Position Specific Scoring Matrix (PSSM) features (length: 21) --- pssm</b>: PSSM is amino acid substitution scores calculated on protein multiple sequence alignment of homolog sequences for each given position in the protein sequence.</li>
<li> <b>Biophysical features (length: 16) --- biophysical </b> For each amino acid we create a normalized vector of their biophysical properties, e.g., flexibility,  instability,  surface accessibility,  kd-hydrophobicity, hydrophilicity, and etc.</li>
</ul>

In order to use combinations of features in the software please use the following keywords for the key of `features_to_use`. `features_to_use` is part of model parameters.
The included features in the config will be concatenated as input:

```
 model_paramters:
  features_to_use:
  - onehot
  - embedding
  - elmo
  - pssm
  - biophysical
```


Return to the [table of content ↑](#tableofcontent).

<hr/>

## Training parameters
<a name="Training"/>

The following is an example of parameters for running the training and storing the results (`run_parameters`).

```
run_parameters:
  domain_name: baseline
  setting_name: baseline
  epochs: 100
  test_batch_size: 100
  train_batch_size: 64
  patience: 10
  gpu: 1
```


### `domain` and `setting_name`

The results of the model would be saved to `results` directory. The `domain` and `setting_name` parameters will be created as directy and sub-directories inside `results` to store the model weights
and results.

### `epoch` and `batch-sizes`

`epoch` refers to the number of time to iterate over the training data and `batch_size` refers to the size of data-split in each optimization step.
For a proper and faster learning we have already performed bucketing (sorting the training sequences according to their lengths), which minimizes the padding operations as well.

### `patience`

To avoid overfitting we perform early stopping, meaning that if the performance only improves over the training set and not the test set after a few epoch we stop the training.
Because then it means that the model specialized to the training data by memorizing and cannot generalize further for the test set. `patience` determine for how many epochs we should wait for an improvement on the test set.

### `gpu`

Which GPU device ID to use for training/testing the model.

Return to the [table of content ↑](#tableofcontent).

<hr/>

## How to configure input for different deep learning models
<a name="Models"/>

### Model (a) CNN + BiLSTM

For the details of CNN + BiLSTM model please refer to the paper, to specify this model for the paper use `deep_learning_model: a_cnn_bilstm`

![model_a](https://user-images.githubusercontent.com/8551117/61132550-e0457a00-a4bb-11e9-84e9-538d6455ce98.png)

`convs` refers to the convolution window sizes (in the following example we use 5 window sizes of  3, 5, 7, and 11).

`filter_size` is the size of convolutional filters.

`dense_size` is the size of feed forward layers are used before and after LSTM.

`dropout_rate` is the dropout rate.

`lstm_size` is the hidden size of bidirectional LSTM.

`lr` is the learning rate.

`features_to_use` is already covered at [3.1 Features](#Features).


Sample config file
```
deep_learning_model: a_cnn_bilstm
model_paramters:
  convs:
  - 3
  - 5
  - 7
  - 11
  - 21
  filter_size: 256
  dense_size: 1000
  dropout_rate: 0.5
  lstm_size: 1000
  lr: 0.001
  features_to_use:
  - onehot
  - pssm
```



## Model (b) CNN + BiLSTM + Highway Connection of PSSM

For the details of CNN + + Highway Connection of PSSM model please refer to the paper, to specify this model for the paper use `deep_learning_model: model_b_cnn_bilstm_highway`

![mdoel_b](https://user-images.githubusercontent.com/8551117/61133494-d91f6b80-a4bd-11e9-8999-4ce501289ec2.png)

`convs` refers to the convolution window sizes (in the following example we use 5 window sizes of  3, 5, 7, and 11).

`filter_size` is the size of convolutional filters.

`dense_size` is the size of feed forward layers are used before and after LSTM.

`dropout_rate` is the dropout rate.

`lstm_size` is the hidden size of bidirectional LSTM.

`lr` is the learning rate.

`features_to_use` is already covered at [3.1 Features](#Features).

`use_CRF` is indicate whether you would like to include a CRF layer at the end.


Sample config file
```
deep_learning_model: model_b_cnn_bilstm_highway
model_paramters:
  convs:
  - 3
  - 5
  - 7
  - 11
  - 21
  filter_size: 256
  dense_size: 1000
  dropout_rate: 0.5
  lstm_size: 1000
  lr: 0.001
  features_to_use:
  - onehot
  - pssm
  use_CRF: false
```


## Model (c) CNN + BiLSTM + Conditional Random Field Layer

For the details of CNN + BiLSTM + Conditional Random Field Layer model please refer to the paper, to specify this model for the paper use `deep_learning_model: model_c_cnn_bilstm`

![model_c](https://user-images.githubusercontent.com/8551117/61134185-54355180-a4bf-11e9-9586-d7b996f205a7.png)

`convs` refers to the convolution window sizes (in the following example we use 5 window sizes of  3, 5, 7, and 11).

`filter_size` is the size of convolutional filters.

`dense_size` is the size of feed forward layers are used before and after LSTM.

`dropout_rate` is the dropout rate.

`lstm_size` is the hidden size of bidirectional LSTM.

`lr` is the learning rate.

`features_to_use` is already covered at [3.1 Features](#Features).

`CRF_input_dim` the input dimension of CRF layer.


Sample config file
```
deep_learning_model: model_c_cnn_bilstm_crf
model_paramters:
  convs:
  - 3
  - 5
  - 7
  - 11
  - 21
  filter_size: 256
  dense_size: 1000
  dropout_rate: 0.5
  lstm_size: 1000
  lr: 0.001
  features_to_use:
  - onehot
  - pssm
  lstm_size: 1000
  CRF_input_dim: 200
```

## Model (d) CNN + BiLSTM + Attention mechanism

For the details of CNN + BiLSTM + Attention mechanism model please refer to the paper, to specify this model for the paper use `deep_learning_model: model_d_cnn_bilstm_attention`

![model_d-2](https://user-images.githubusercontent.com/8551117/61134627-4f24d200-a4c0-11e9-982b-49279a5da669.png)

`attention_type` is the attention type to be selected from `additive` or `multiplicative`.

`attention_units` is the number of attention units.

`convs` refers to the convolution window sizes (in the following example we use 5 window sizes of  3, 5, 7, and 11).

`filter_size` is the size of convolutional filters.

`dense_size` is the size of feed forward layers are used before and after LSTM.

`dropout_rate` is the dropout rate.

`lstm_size` is the hidden size of bidirectional LSTM.

`lr` is the learning rate.

`features_to_use` is already covered at [3.1 Features](#Features).

`use_CRF` is indicate whether you would like to include a CRF layer at the end.



Sample config file
```
deep_learning_model: model_d_cnn_bilstm_attention
model_paramters:
  attention_type: additive
  attention_units: 32
  convs:
  - 3
  - 5
  - 7
  - 11
  - 21
  filter_size: 256
  dense_size: 1000
  dropout_rate: 0.5
  lstm_size: 1000
  lr: 0.001
  features_to_use:
  - onehot
  - pssm
  lstm_size: 1000
  use_CRF: false
```

## Model (e) CNN

For the details of CNN model please refer to the paper, to specify this model for the paper use `deep_learning_model: model_e_cnn`

![model_e](https://user-images.githubusercontent.com/8551117/61135353-b42cf780-a4c1-11e9-87aa-fdcc13a2892f.png)

`convs` refers to the convolution window sizes (in the following example we use 5 window sizes of  3, 5, 7, and 11).

`filter_size` is the size of convolutional filters.

`dense_size` is the size of feed forward layers are after the concatenation of convlolution results.

`dropout_rate` is the dropout rate.

`lr` is the learning rate.

`features_to_use` is already covered at [3.1 Features](#Features).

`use_CRF` is indicate whether you would like to include a CRF layer at the end.

Sample config file
```
deep_learning_model: model_e_cnn
model_paramters:
  convs:
  - 3
  - 5
  - 7
  - 11
  - 21
  filter_size: 256
  dense_size: 1000
  dropout_rate: 0.5
  lstm_size: 1000
  lr: 0.001
  features_to_use:
  - onehot
  - pssm
  lstm_size: 1000
  use_CRF: false
```

## Model (f) Multiscale CNN

For the details of Multiscale CNN model please refer to the paper, to specify this model for the paper use `deep_learning_model: model_f_multiscale_cnn`

![model_f](https://user-images.githubusercontent.com/8551117/61135721-85fbe780-a4c2-11e9-8f65-3ea3ac2b17ee.png)

`multiscalecnn_layers` how many gated muliscale CNNs should be stacked.

`cnn_regularizer` regularizing parameter for the CNN.

`convs` refers to the convolution window sizes (in the following example we use 5 window sizes of  3, 5, 7, and 11).

`filter_size` is the size of convolutional filters.

`dense_size` is the size of feed forward layers are after the concatenation of convlolution results.

`dropout_rate` is the dropout rate.

`lr` is the learning rate.

`features_to_use` is already covered at [3.1 Features](#Features).

`use_CRF` is indicate whether you would like to include a CRF layer at the end.

Sample config file
```
deep_learning_model: model_f_multiscale_cnn
model_paramters:
  cnn_regularizer: 5.0e-05
  multiscalecnn_layers: 3
  convs:
  - 3
  - 5
  - 7
  - 11
  - 21
  filter_size: 256
  dense_size: 1000
  dropout_rate: 0.5
  lstm_size: 1000
  lr: 0.001
  features_to_use:
  - onehot
  - pssm
  lstm_size: 1000
  use_CRF: false
```

Return to the [table of content ↑](#tableofcontent).

<hr/>

## Your own model

Create your own model by just using the template of model_a to .._f, and test its performance against the existing methods.

Return to the [table of content ↑](#tableofcontent).


## Output
<a name="Output"/>

Finally after completion of training, DeepPrime2Seq generate a PDF of the report with the following information at `results/$domain/$setting/report.pdf`:

 - [x] The accuracy of trained model on the standard test set of the task (CB513)
 - [x] Confusion matrix of the model
 - [x] Contingency metric of the error at the edges of secondary structure changing, along with the p-value of Chi-Square and G-test tests.
 - [x] The learning curve
 - [x] The neural network weights for the best models


![Screen Shot 2019-07-12 at 5 33 30 PM](https://user-images.githubusercontent.com/8551117/61140191-45ed3280-a4cb-11e9-95f5-e12745b5de61.png)
