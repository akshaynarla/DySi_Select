# ==============================================================================
#
# Copyright 2023 IAS, Uni Stuttgart
#
# evaluate_model python module is used for training the neural network SIN 
# (Situation Identification Network). All parameters related to the evaluation
# of the network is available in the config files in the models directory.
# The structure is inspired by the module Cam2BEV
# ==============================================================================
"""evaluate_model python module is used for training the neural network SIN 
(Situation Identification Network). All parameters related to the evaluation
of the network is available in the config files in the models directory.
The structure is inspired by the module Cam2BEV.
"""

import os
import sys
import datetime
import tensorflow as tf
import numpy as np

from SIN.src.data import make_dataset
from SIN.src.utils import data_utils
from SIN.src.utils import model_utils
from SIN.src.utils import SIN_utils

def main():
    # Parse input parameters based on config file
    conf = data_utils.parse_test_configs()
    # Determination of absolute filepaths
    # testing directory here
    conf.model                  = data_utils.abspath(conf.model)
    conf.model_weights          = data_utils.abspath(conf.model_weights) if conf.model_weights is not None else conf.model_weights
    conf.output_dir             = data_utils.abspath(conf.output_dir)
    
    test_dir = data_utils.listToString(conf.input_testing)
    test_dir = data_utils.abspath(test_dir)
    # Load the neural network
    architecture = model_utils.load_module(conf.model)
    print("Loaded python module:", os.path.basename(conf.model))
    # Load the data pipeline for evaluation/testing
    test_data = make_dataset.prepare(test_dir, conf.batch_size, shuffl=False)
    n_val_data = test_data.samples
    true_labels = test_data.classes
    print("Testing pipeline with samples:", n_val_data)
    print("Classes in the test_data dir:", true_labels)
    
    # build the network backbone
    model = architecture.get_network(conf)
    model.load_weights(conf.model_weights)
    print(f"Reloaded model from {conf.model_weights}")
    
    # run prediction of situation after training the model 
    # or with pre-trained weights
    prediction = model.predict(test_data).squeeze()
    predicted_labels = np.argmax(prediction, axis=1)
    print("Classes predicted:", predicted_labels)
    # Create Confusion Matrix
    print("Evaluating Confusion Matrix....")
    SIN_utils.gen_conf_matrix(ground_truth=true_labels, prediction=predicted_labels, 
                              op_dir=conf.output_dir)
    print("Generated Confusion Matrix and related metrics.")
    
    
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        try:         
            sys.exit(0)
        except SystemExit:
            os._exit(0)