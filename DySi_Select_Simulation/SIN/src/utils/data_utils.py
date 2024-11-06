# ==============================================================================
#
# Copyright 2023 IAS, Uni Stuttgart
#
# data_utils python module provides the utility interfaces for handling
# data efficiently. The utils from Cam2BEV is reused in most cases, to avoid
# rework. To ensure, modules are independent of each other, a new data_utils
# is created in SIN module.
# ==============================================================================
"""data_utils python module provides the utility interfaces for handling
data efficiently. The utils from Cam2BEV is reused in most cases, to avoid
rework. To ensure, modules are independent of each other, a new data_utils
is created in SIN module.

"""

import os
import sys
import configargparse
import numpy as np
import cv2
import tensorflow as tf


def abspath(path):
    """Interface for obtaining the absolute path of a file/folder.

    Args:
        path (str): file path to be converted to absolute file path

    Returns:
        str: absolute file path
    """
    return os.path.abspath(os.path.expanduser(path))


def listToString(s):
    """Interface for converting list to string.

    Args:
        s (list): parsed list

    Returns:
        str: converted string
    """
    # initialize an empty string
    str1 = ""
 
    # traverse in the string
    for ele in s:
        str1 += ele
 
    # return string
    return str1


def load_image(filename):
    """ Interface for loading the image.

    Args:
        filename (str): path for loading the image

    Returns:
        image: loaded image
    """
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def parse_train_configs():
    """Interface to parse the parameters and files for training the SIN neural network.

    Returns:
        loaded config file: parameters from the parsed yml file
    """
    parser = configargparse.ArgParser()
    parser.add_argument("-c", "--config", is_config_file=True, help="config file")
    parser.add_argument("-it", "--input-training",           type=str, required=True, nargs="+", help="directory/directories of input samples for training")
    parser.add_argument("-iv", "--input-validation",         type=str, required=True, nargs="+", help="directory/directories of input samples for validation")
    parser.add_argument("-a",    "--arch",                      type=str,   required=True,              help="Python file defining the backbone arch")
    parser.add_argument("-m",    "--model",                      type=str,   required=True,              help="Python file defining the neural network")
    parser.add_argument("-e",    "--epochs",                     type=int,   required=True,              help="number of epochs for training")
    parser.add_argument("-bs",   "--batch-size",                 type=int,   required=True,              help="batch size for training")
    parser.add_argument("-lr",   "--learning-rate",              type=float, default=1e-4,               help="learning rate of Adam optimizer for training")
    parser.add_argument("-lw",   "--loss-weights",               type=float, default=None,   nargs="+",  help="factors for weighting classes differently in loss function")
    parser.add_argument("-esp",  "--early-stopping-patience",    type=int,   default=10,                 help="patience for early-stopping due to converged validation mIoU")
    parser.add_argument("-si",   "--save-interval",  type=int, default=5,        help="epoch interval between exports of the model")
    parser.add_argument("-o",    "--output-dir",     type=str, required=True,    help="output dir for TensorBoard and models")
    parser.add_argument("-mw",   "--model-weights",  type=str, default=None,     help="weights file of trained model for training continuation")
    parser.add_argument("-f", "--fine_tune", type=int, default=0, help="number of pre-trained layers to be re-trained")
    parser.add_argument("-d", "--dropout", type=float, default=0.2, help="dropout config for the dense top layers")
    parser.add_argument("-cl", "--n_class", type=int, default=5, help="number of situation classes")
    conf, unknown = parser.parse_known_args()
    return conf


def parse_val_configs():
    """Interface to parse the parameters and files for validation of the SIN neural network.

    Returns:
        loaded config file: parameters from the parsed yml file
    """
    parser = configargparse.ArgParser()
    parser.add_argument("-c", "--config", is_config_file=True, help="config file")
    parser.add_argument("-iv", "--input-validation",         type=str, required=True, nargs="+", help="directory/directories of input samples for validation")
    parser.add_argument("-lv", "--label-validation",         type=str, required=True,            help="directory of label samples for validation")
    parser.add_argument("-nv", "--max-samples-validation",   type=int, default=None,             help="maximum number of validation samples")
    parser.add_argument("-is",  "--image-shape",           type=int, required=True, nargs=2, help="image dimensions (HxW) of inputs and labels for network")
    parser.add_argument("-m",    "--model",                      type=str,   required=True,              help="Python file defining the neural network")
    parser.add_argument("-e",    "--epochs",                     type=int,   required=True,              help="number of epochs for training")
    parser.add_argument("-mw",   "--model-weights",  type=str, default=None,     help="weights file of trained model for training continuation")
    conf, unknown = parser.parse_known_args()
    return conf

def parse_test_configs():
    """Interface to parse the parameters and files for testing the SIN network.

    Returns:
        loaded config file: parameters from the parsed yml file
    """
    # parse parameters from config file or CLI
    parser = configargparse.ArgParser()
    parser.add_argument("-c",    "--config", is_config_file=True, help="config file")
    parser.add_argument("-ip",   "--input-testing",          type=str, required=True, nargs="+", help="directory/directories of input samples for testing")
    parser.add_argument("-m",    "--model",                  type=str, required=True,            help="Python file defining the neural network")
    parser.add_argument("-mw",   "--model-weights",          type=str, required=True,            help="weights file of trained model")
    parser.add_argument("-o",    "--output-dir",     type=str, required=True,    help="output dir for saving results")
    parser.add_argument("-bs",   "--batch-size",                 type=int,   required=True,              help="batch size for training")
    parser.add_argument("-a",    "--arch",                      type=str,   required=True,              help="Python file defining the backbone arch")
    parser.add_argument("-f", "--fine_tune", type=int, default=0, help="number of pre-trained layers to be re-trained")
    parser.add_argument("-d", "--dropout", type=float, default=0.2, help="dropout config for the dense top layers")
    parser.add_argument("-cl", "--n_class", type=int, default=5, help="number of situation classes")
    conf, unknown = parser.parse_known_args()
    return conf