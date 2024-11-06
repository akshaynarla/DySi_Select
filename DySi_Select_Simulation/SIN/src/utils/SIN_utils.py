# ==============================================================================
#
# Copyright 2023 IAS, Uni Stuttgart
#
# SIN_utils python module provides the utility interfaces for handling
# data from SIN efficiently.
# ==============================================================================

"""SIN_utils python module provides the utility interfaces for handling data from SIN efficiently.

"""
from matplotlib import pyplot as plt
import numpy as np
from pycm import *
from datetime import datetime
import os


def gen_conf_matrix(ground_truth, prediction, op_dir):
    """Interface for generating the confusion matrix for the evaluation of network performance.

    Args:
        ground_truth (any): ground truth data
        prediction (any): predicted data
        op_dir (str): output directory to store results
    """
    cm = ConfusionMatrix(actual_vector=ground_truth, predict_vector=prediction)
    cm.stat(summary=True)
    cm.plot(cmap=plt.cm.Greens, number_label=True, plot_lib="matplotlib")
    cm_file = f"CMlog-{datetime.now():%Y-%m-%d %H-%m-%d}"
    cm.save_csv(os.path.join(op_dir, cm_file))


def get_label():
    """Interface for converting predicted class to label.

    Returns:
        str: class label
    """
    class_label = {
        0: "FreeDrive",
        1: "FreeDriveParkedVehicles",
        2: "FreeIntersection",
        3: "OccludedDrive",
        4: "OccludedIntersection"
    }
    return class_label