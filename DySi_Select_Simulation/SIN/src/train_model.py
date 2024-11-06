# ==============================================================================
#
# Copyright 2023 IAS, Uni Stuttgart
#
# train_model python module is used for training the neural network SIN 
# (Situation Identification Network). All parameters related to the training
# of the network is available in the config files in the models directory.
# The training structure is inspired by the module Cam2BEV
# ==============================================================================

"""train_model python module is used for training the neural network SIN 
(Situation Identification Network). All parameters related to the training
of the network is available in the config files in the models directory.
The training structure is inspired by the module Cam2BEV.
"""
import os
import sys
from datetime import datetime
import tensorflow as tf
import numpy as np

from SIN.src.data import make_dataset
from SIN.src.utils import data_utils
from SIN.src.utils import model_utils

def main():
    # Parse input parameters based on config file
    conf = data_utils.parse_train_configs()
    # Determination of absolute filepaths
    conf.model                  = data_utils.abspath(conf.model)
    conf.model_weights          = data_utils.abspath(conf.model_weights) if conf.model_weights is not None else conf.model_weights
    conf.output_dir             = data_utils.abspath(conf.output_dir)
    
    # Load the neural network
    architecture = model_utils.load_module(conf.model)
    print("Loaded python module:", os.path.basename(conf.model))
    train_dir = data_utils.listToString(conf.input_training)
    train_dir = data_utils.abspath(train_dir)
    val_dir = data_utils.listToString(conf.input_validation)
    val_dir = data_utils.abspath(val_dir)
    
    # # Load the data pipeline for training and validation.
    # Since the input to the network is directly from Cam2BEV module
    # preprocessing the data is not necessary. Resizing to the dimensions of
    # neural network can be done.
    train_data = make_dataset.prepare(train_dir, conf.batch_size, shuffl=True)
    n_train_data = train_data.samples
    n_train_labels = train_data.class_indices
    print("Training data pipeline with samples:", n_train_data)
    print("Classes in the training dir:", n_train_labels )
    val_data = make_dataset.prepare(val_dir, conf.batch_size, shuffl=True)
    n_val_data = val_data.samples
    n_val_labels = val_data.class_indices
    print("Validation data pipeline with samples:", n_val_data)
    print("Classes in the validation dir:", n_val_labels)
    
    # build the network backbone
    model = architecture.get_network(conf)
    
    # Load weights, if existing along with optimizer and loss
    if conf.model_weights is not None:
        model.load_weights(conf.model_weights)
    optimizer = tf.keras.optimizers.Adam(learning_rate=conf.learning_rate)
    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics = ['accuracy', tf.keras.metrics.Precision(),
               tf.keras.metrics.Recall()]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    print(f"Compiled model {os.path.basename(conf.model)}")
    
    # Creating output directories for saving model results
    model_output_dir = os.path.join(conf.output_dir, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    tensorboard_dir = os.path.join(model_output_dir, "TensorBoard")
    checkpoint_dir  = os.path.join(model_output_dir, "Checkpoints")
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Creating necessary callbacks
    n_batches_train = n_train_data // conf.batch_size
    n_batches_valid = n_val_data // conf.batch_size
    tensorboard_cb      = tf.keras.callbacks.TensorBoard(tensorboard_dir, update_freq="epoch", profile_batch=0)
    checkpoint_cb       = tf.keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_dir, "e{epoch:03d}_weights.hdf5"), save_freq=n_batches_train*conf.save_interval, save_weights_only=True)
    best_checkpoint_cb  = tf.keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_dir, "best_weights.hdf5"), save_best_only=True, monitor="val_accuracy", mode="max", save_weights_only=True)
    early_stopping_cb   = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=conf.early_stopping_patience, verbose=1)
    callbacks = [tensorboard_cb, checkpoint_cb, best_checkpoint_cb, early_stopping_cb]
    
    # Training process
    print("Starting Training now...")
    model.fit(train_data,
          epochs=conf.epochs, steps_per_epoch=n_batches_train,
          validation_data=val_data, validation_freq=1, validation_steps=n_batches_valid,
          callbacks=callbacks)
    print("Finished training.")
    
    
if __name__ == '__main__':
    try:
        tf.config.list_physical_devices('GPU')
        main()
    except KeyboardInterrupt:
        try:         
            sys.exit(0)
        except SystemExit:
            os._exit(0)