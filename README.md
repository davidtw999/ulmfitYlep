# ULMFiT

This repository contains implementations of ULMFiT approach applied to sentiment classification for Yelp reviews. 

## Files provided

**`appendix`** contains `translationscript.ipynb` file for translating data

**`train_test_model.ipynb`** file for training ULMFiT

*   `trainData.csv` Translated Train Data with 650,000 reviews

*   `testDataTrans.csv` Translated Test Data 50000 reviews

*   `models folder` contain pre-trained models, containing `itos_wt103.pkl` and `lstm_wt103.pth`.



## Introduction

**ULMFiT**, or **U**universal **L**language **M**model **FiT**Fine-tuning, is a new transfer learning method which
surpass the state-of-the-art on a wide array of Natural Language Processing (NLP) tasks.

The idea behind this approach has three stages. The first stage is General-domain LM pretraining where it pre-trains a language model to obtain the knowledge of the semantics of language spoken by human. Then in the following stage, Target task LM fine-tuning, the model is fine tuned in a content specific language . This process is transfer learning, which plays an essential role in ULMFiT. The last stage is Target task classifier fine-tuning in which we build a classifier by further fine-tuning the model obtained above.

To be able to build such a classifier in an efficient manner,  we need to use the `fastai` framework. 

## ULMFiT with the `fastai` framework?

`fastai` is a library which simplifies the training process of neutral nets and it's based on research undertaken at [fast.ai](http://www.fast.ai). Refer to to [documentation](https://docs.fast.ai/) for details.

### System and library requirement

Python: You need to have python 3.6 or higher

Install system-wide NVIDIA's CUDA and related libraries for GPU requirement

### Testing code on Operating System:

Linux

Mac

Windows

### Installation

**NB:** `fastai v1` currently supports Linux only. **PyTorch v1** and **Python 3.6** or later are the minimal version requirements. 

`fastai-1.x` can be installed in three ways, including `conda` or `pip` package managers and also from source. Note users can't just run, since you first need to get the correct `pytorch` version and `Python` version before running *install*. 

#### Conda Install 

```bash
conda install -c pytorch -c fastai fastai
```

#### PyPI Install

```bash
pip install fastai
```


#### Developer Install

The following commands will lead to a [pip editable install](https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs). Then users can `git pull` at any time and the users' environment will automatically be updated:

```bash
git clone https://github.com/fastai/fastai
cd fastai
tools/run-after-git-clone
pip install -e ".[dev]"
```

## Yelp Review Classifier Training with ULMFiT

### General-domain LM pretraining

In this step, the model learns the language from Wikitext-103, consisting of 28,595 Wikipedia articles and 103 million words. Note the the pre-trained models are open source and can be downloaded from [here](http://files.fast.ai/models/wt103_v1/). The pretrained models contain two files, including `itos_wt103.pkl` and `lstm_wt103.pth`. 

The models will be directly used in training Target Task ML

### Target task LM fine-tuning

The key steps are as follows:

**1. Create Yelp reviews language base** Tranlated train data and translated test data are merged, while only `text` and `label` columns are kept. Note for the formatting purposes, we've manully added fake labels for test data. Then we define label column as 0 to follow format.

**2. Assemble the raw data for the language model**  Define `TextLMDataBunch` class to get the data ready to be fed into the model. Name the class as `data_lm`

**3. Initialise the language_model_learner class** 
*   **language_model_learner** Initialise a `Learner` class with a language model from DataBunch `data_lm`.
*   **AWD_LSTM** Stands for ASGD Weight-Dropped LSTM, language modelling supported with a set of regularisation and optimisaiton strategies.
*   **pretrained_fnames** Pre-trained models, including `itos_wt103` and `lstm_wt103`.
*   **drop_mult** Scale a numpy array of dropouts with the same relative ratio for each language model layer to another.
*   **fit_one_cycle** Fit a model following the 1 cycle policy. Run 5 epoches.

**4. Unfreeze the entire model** Unfreeze all the layers so that further fine tuning can be performed. Run 1 epoch.


### Target task classifier fine-tuning

**1. Assemble translated train data for the classification model:**  Define `TextClasDataBunch` class to get the data ready to be fed into the model. Name the class as `data_clas`. Meanwhile, we batchifies the data for classificaiton. 

**2. Initialise the text_classifier_learner class** Use our fine-tuned encoder to build a classifier. Run 5 epoches. 

**3. Fine-tune the lasy two layers** Unfreeze the last two layers and fine-tune them.  Run 1 epoch.

**4. Fine-tune the last three layers** Unfreeze the last three layers and fine-tune them.  Run 1 epoch.

**5. Fine-tune all layers** Unfree the entire model and fine tune the layers. Run two epoches.


## Make Prediction

After having built our classifier, we can make prediction on test data by calling function `pred_labels`. 

