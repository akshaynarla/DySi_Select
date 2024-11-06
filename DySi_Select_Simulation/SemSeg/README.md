# Semantic Segmentation Module (SemSeg) 
==============================

This repository contains part implementation of the thesis "Dynamic Situation-based selection of relevant sensor data" (MT3532). Here, a pretrained semantic segmentation network, DeepLabv3plus trained with cityscapes weights, is used for providing semantic segmentation mask for use in Cam2BEV.

## Content

- [Project Organization](#project-organization)
- [Installation](#installation)
- [Preprocessing](#pre-processing)
- [Configuration Information](#configuration-information)

## Project Organization
```
    SemSeg
    ├── examples                     <- Folder to store example test images
    ├── output                       <- Folder for storing the output for comparing RGB with produced semantic segmentation mask
    ├── README.md                    <- The top-level README for developers using this project.
    ├── pretrained                   <- Folder for storing pre-trained weights
    ├── seg_output                   <- Folder for storing the semantic segmentation image for use in Cam2BEV
    ├── deeplabv3plus.py             <- Official implementation of DeepLabv3plus in tensorflow (taken from GitHub)
    ├── sem_seg.py                   <- Python script to produce a semantically segmented image of the parsed RGB image
    └── semseg_utils.py              <- Python script with helper interfaces for sem_seg.py
```

## Installation

To use SemSeg directly, this repository can be downloaded. Similar to Cam2BEV, it is suggested to setup a **Python 3.7/8** virtual environment (e.g. by using _virtualenv_ or _conda_). Inside the virtual environment, users can then use _pip_ to install all package dependencies. The results of this module were achieved with _TensorFlow 2.5_ (_CUDA 11.2_ for GPU support). The support starts breaking with higher versions, since the _DeepLab_ model implementations do not support _TensorFlow>2.5_ due to non-trainable lambda layers. But the current modules have been tested with CUDA 11.4 and it can be assumed that Tensorflow 2.5 works with CUDA 11.4.

CUDA installation requires use of conda package manager. Please follow the instructions in the [Tensorflow installation page](https://www.tensorflow.org/install/pip). 

## Pre-processing

All necessary pre-processing of the parsed data (image) is handled in the [semseg_utils.py](semseg_utils.py).

## Configuration Information

No configuration necessary here with respect to the prototype. This is used to test the portability of the developed concept in real traffic situations. This module is only used to test the system with data from a public dataset such as KITTI or nuScenes. The RGB data is converted to semantic segmented frames using the provided interface.
Intermediate results to be used further in the prototype is stored in [seg_output](seg_output) separated by date (i.e. folder for each day). The original RGB along with the semantic mask from the network will be stored in [output](output)
All necessary information about the interfaces will be provided in the interface documentation.