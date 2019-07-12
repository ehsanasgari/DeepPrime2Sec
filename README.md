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


Through error analysis on the best performing model, we showed that the misclassification is significantly more common at positions that undergo secondary structure transitions, which is most likely due to the inaccurate assignments of the secondary structure at the boundary regions. Notably, when ignoring amino acids at secondary structure transitions in the evaluation, the accuracy increases to 90.3%. Furthermore, the best performing model mostly mistook similar structures for one another, indicating that the deep learning model inferred high-level information on the secondary structure.


DeepPrime2Sec and the used datasets are available here under the Apache 2 license.

Return to the [table of content ↑](#tableofcontent).

<a name="Installation"/>
# Installation

## Conda environment set up


In order to install the required libraries for running DeepPrime2Sec use the following conda command:

```
conda create --name deepprime2sec --file installations/env_linux.txt
```

Subsequently, you need to activate the created virtual environment before running:

```
source activate deepprime2sec
```

Before running the software make sure to download the traning dataset (which was too large for git) from here.


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

<hr/>

## How to configure input for different deep learning models
<a name="Models"/>

### Model (a) CNN + BiLSTM


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
  dense_size: 1000
  dropout_rate: 0.5
  features_to_use:
  - onehot
  - pssm
  lr: 0.001
  lstm_size: 1000
run_parameters:
  domain_name: baseline
  epochs: 10
  gpu: 1
  patience: 10
  setting_name: baseline
  test_batch_size: 100
  train_batch_size: 64
```
## Model (b) CNN + BiLSTM + Highway Connection of PSSM


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
  dense_size: 1000
  dropout_rate: 0.5
  features_to_use:
  - onehot
  - pssm
  filter_size: 256
  lr: 0.001
  lstm_size: 1000
  use_CRF: false
run_parameters:
  domain_name: baseline
  epochs: 100
  gpu: 1
  patience: 10
  setting_name: baseline
  test_batch_size: 100
  train_batch_size: 64

```


## Model (c) CNN + BiLSTM + Conditional Random Field Layer

Sample config file
```
deep_learning_model: model_c_cnn_bilstm
model_paramters:
  CRF_input_dim: 200
  convs:
  - 3
  - 5
  - 7
  - 11
  - 21
  dense_size: 1000
  dropout_rate: 0.5
  features_to_use:
  - onehot
  - pssm
  filter_size: 256
  lr: 0.001
  lstm_size: 1000
run_parameters:
  domain_name: baseline
  epochs: 100
  gpu: 1
  patience: 10
  setting_name: baseline
  test_batch_size: 100
  train_batch_size: 64

```

## Model (d) CNN + BiLSTM + Attention mechanism

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
  dense_size: 1000
  dropout_rate: 0.5
  features_to_use:
  - onehot
  - pssm
  filter_size: 256
  lr: 0.001
  lstm_size: 1000
  use_CRF: false
run_parameters:
  domain_name: baseline
  epochs: 100
  gpu: 1
  patience: 10
  setting_name: baseline
  test_batch_size: 100
  train_batch_size: 64
```

## Model (e) CNN

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
  dense_size: 1000
  dropout_rate: 0.5
  features_to_use:
  - onehot
  - pssm
  filter_size: 256
  lr: 0.001
  use_CRF: false
run_parameters:
  domain_name: baseline
  epochs: 100
  gpu: 1
  patience: 10
  setting_name: baseline
  test_batch_size: 100
  train_batch_size: 64
```

## Model (f) Multiscale CNN

Sample config file
```
deep_learning_model: model_f_multiscale_cnn
model_paramters:
  cnn_regularizer: 5.0e-05
  convs:
  - 3
  - 5
  - 7
  - 11
  - 21
  dropout_rate: 0.5
  features_to_use:
  - onehot
  - pssm
  filter_size: 256
  lr: 0.001
  multiscalecnn_layers: 3
  use_CRF: false
run_parameters:
  domain_name: baseline
  epochs: 100
  gpu: 1
  patience: 10
  setting_name: baseline
  test_batch_size: 100
  train_batch_size: 64
```


Return to the [table of content ↑](#tableofcontent).

<hr/>

## Output
<a name="Output"/>

