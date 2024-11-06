# ==============================================================================
#
# Copyright 2023 IAS, Uni Stuttgart
#
# model_utils python module provides the utility interfaces for handling
# the neural network efficiently. The utils from Cam2BEV is reused in most cases, 
# to avoid rework.
# ==============================================================================
"""model_utils python module provides the utility interfaces for handling
the neural network efficiently. The utils from Cam2BEV is reused in most cases, 
to avoid rework.

"""
import os
import sys
import tensorflow as tf
from importlib import util


def load_module(module_file):
    """Interface for loading a python file.

    Args:
        module_file (str): python file to be loaded

    Returns:
        python file: loaded python file
    """
    name = os.path.splitext(os.path.basename(module_file))[0]
    dir = os.path.dirname(module_file)
    sys.path.append(dir)
    spec = util.spec_from_file_location(name, module_file)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

