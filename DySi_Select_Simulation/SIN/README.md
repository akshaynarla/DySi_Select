# Situation Identification Network (SIN) 
==============================

This repository contains the implementation of the thesis "Dynamic Situation-based selection of relevant sensor data" (MT3532). Here, methodology for the identification of the road situation based on BEV input as per the developed concept in MT3532 is provided. The implementation is not complete and consists of a part-implementation of the proposed concept.

## Content

- [Project Organization](#project-organization)
- [Installation](#installation)
- [Data](#data)
- [Preprocessing](#pre-processing)
- [Training](#training)
- [Configuration Information](#configuration-information)


## Project Organization
```
    ├── LICENSE
    ├── Makefile                       <- Makefile with commands like `make data` or `make train` (not used here)
    ├── README.md                      <- The top-level README for developers using this project.
    ├── docs                           <- A default Sphinx project; see sphinx-doc.org for details
    ├── requirements.txt               <- The requirements file for reproducing the analysis environment, e.g. generated with `pip freeze > requirements.txt` (Same as Cam2BEV requirements)
    ├── setup.py                       <- makes project pip installable (pip install -e .) so src can be imported (not used here)
    ├── src                            <- Source code for use in this project.
    │   ├── __init__.py                <- Makes src a Python module
    │   ├── data                       <- Scripts to prepare data. The default dataset location is here.
    │   │   ├── make_dataset.py        <- Script to prepare and load the dataset
    │   │   └── SIN                    <- Location to store the dataset
    │   ├── features                   <- Scripts to turn raw data into features for modeling (not used here)
    │   ├── models                     <- Scripts to define the network architecture (modified VGGNet-16 in our case)
    │   │   └── arch          
    │   │      └── vggnet.py
    │   │── utils                      <- Scripts providing helper functions for training, evaluation and use of SIN
    │   │   ├── data_utils.py
    │   │   ├── model_utils.py
    │   │   └── SIN_utils.py
    │   ├── predict_situation.py       <- Script providing interface for external usage
    │   ├── train_model.py             <- Script for training the neural network architecture
    │   ├── evaluate_model.py          <- Script for evaluating and calculating metrics (confusion matrix)
    │   ├── config.SingleCam.PredictSIN.yml
    │   └── config.MultiCam.PredictSIN.yml   (not used here)
    ├── test_environment.py                  (not used here)
    └── tox.ini                        <- tox file with settings for running tox; see tox.readthedocs.io (not used here)
```

## Installation

To use SIN, this repository can be downloaded. Similar to Cam2BEV, it is suggested to setup a **Python 3.7** virtual environment (e.g. by using _virtualenv_ or _conda_). Inside the virtual environment, users can then use _pip_ to install all package dependencies. The results of the thesis were achieved with _TensorFlow 2.5_ (_CUDA 11.2_ for GPU support). The support starts breaking with higher versions, since the _DeepLab_ model implementations do not support _TensorFlow>2.5_ due to non-trainable lambda layers.

CUDA installation requires use of conda package manager. Please follow the instructions in the [Tensorflow installation page](https://www.tensorflow.org/install/pip). 

```bash
pip install -r requirements.txt
```
## Data

To train the Situation Identification Network (SIN), a dataset is derived manually from the [Cam2BEV 2_F dataset](https://gitlab.ika.rwth-aachen.de/cam2bev/cam2bev-data/-/tree/master/2_F). The ground truth BEV images from front camera only is seperated into 5 classes _(FreeDrive, OccludedDrive, FreeIntersection, FreeDriveParkedVehicles, OccludedIntersection)_. The dataset consists of about 2500 training images, 250 validation images and 200 testing images classified into the above mentioned classes. This dataset is to be located in [src/data/SIN](src/data/SIN) in the desired camera configuration. Currently, only SingleCam dataset is provided and used in the prototype. Similar to SingleCam configuration, a MultiCam dataset can also be derived from the [Cam2BEV 1_FRLR dataset](https://gitlab.ika.rwth-aachen.de/cam2bev/cam2bev-data/-/tree/master/1_FRLR).

## Pre-processing

All necessary pre-processing of the dataset is handled in the [src/data/make_dataset.py](src/data/make_dataset.py). The dataset is resized as per the input shape of VGGNet-16 network using the preprocessing function provided for vggnet in tensorflow.

## Training

Use the scripts [src/train_model.py](src/train_model.py) and [src/evaluate_model.py](src/evaluate_model.py) to train a SIN model, evaluate it on validation data, and make predictions on a testing dataset.

Input directories, training parameters, and more can be set via CLI arguments or in a config file. Run the scripts with `--help`-flag or see one of the provided exemplary config files for reference.
- [src/config.SingleCam.PredictSIN.yml](src/config.SingleCam.PredictSIN.yml) 

The following commands will guide the training of SIN with SingleCam configuration in a Linux environment. But it can be executed in Windows environment also with correct commands in a python environment.

### Training

Start training _SIN_ by passing the provided config file [src/config.SingleCam.PredictSIN.yml](src/config.SingleCam.PredictSIN.yml). Training will automatically stop if the validation accuracy is not rising after the configured early-stopping interval.

```bash
cd src/
```
```bash
./train_model.py -c config.SingleCam.PredictSIN.yml
```

The training progress can be visualized by pointing *TensorBoard* to the output directory (`src/output/singlecam` by default). Training metrics will also be printed to `stdout`.

### Evaluation

Before evaluating the trained model, set the parameter `model-weights` to point to the `best_weights.hdf5` file in the `Checkpoints` folder of the output directory. Then run evaluation to compute a confusion matrix and related metrics. The testing dataset location can be specified in the yml file.

```bash
./evaluate_model.py -c config.SingleCam.PredictSIN.yml --model-weights output/singlecam/<YOUR-TIMESTAMP>/Checkpoints/best_weights.hdf5
```

The evaluation results will be printed at the end of evaluation and also be exported to the `Evaluation` folder in the corresponding output directory.

_Details about the interfaces will be provided in the interface documentation_

## Configuration Information

#### _I want to set different training hyperparameters_

Run the training script with `--help`-flag or have a look at one of the provided config files to see what parameters can be set.

#### _I want to add new class to the dataset_

Create a folder with the name of the new situation class, like the ones already created. No other changes needed. The training process automatically detects the added class.

#### _How easy is it to add a different modality to the model?_

It is not possible to add a different modality since the current model is trained with semantically segmented BEV images.
The data format from another modality (for ex: LiDAR) will be different and the current model is not capable of handling other data formats.
But the proposed concept is suitable for multi-modal systems also (needs a MultiMod2BEV converter and a network to identify the situation from the produced BEV format).

On the other hand, as long as a BEV image (current training model is for single camera, but can be retrained for a different configuration/setup) is provided, any number of cameras can be added.


## Acknowledgement

<p><small>Project structure based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
