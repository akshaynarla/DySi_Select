# ==============================================================================
#
# Copyright 2023 IAS, Uni Stuttgart
#
# vggnet python module is used for loading the VGGNET neural network backbone of SIN 
# (Situation Identification Network). All parameters related to the training
# of the network is available in the config files in the models directory.
# ==============================================================================
"""vggnet python module is used for loading the VGGNET neural network backbone of SIN 
(Situation Identification Network). All parameters related to the training
of the network is available in the config files in the models directory.

"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg16 import VGG16


def get_network(conf):
    """Interface for training and fine-tuning the network 
    with the architecture defined in the config file.

    Args:
        conf (config file): loaded configuration file from yml or CLI

    Returns:
        Model: loaded SIN model (for training purposes)
    """
    # Create model based on architecture name
    if (conf.arch == 'vggnet') and (conf.config is not None):
        print("using pre-trained vggnet backbone with modified top layers.....")
        # Since dataset is already preprocessed, no further preprocessing necessary
        # Load pre-trained VGG16 Convolutional Neural Network with imagenet
        conv_base = VGG16(include_top=False,
                      weights= 'imagenet', input_shape=(224,224,3))
        # use fine_tune parameter to select layers of conv_base for training 
        if(conf.fine_tune > 0):
            for layer in conv_base.layers[:-(conf.fine_tune)]:
                layer.trainable = False
        else:
            for layer in conv_base.layers:
                layer.trainable = False
        # Top Model for SIN
        top_layer = conv_base.output
        top_layer = Flatten()(top_layer)
        top_layer = Dense(1024, activation='relu')(top_layer)
        # Dropout to avoid overfitting
        top_layer = Dropout(conf.dropout)(top_layer)
        output_layer = Dense(conf.n_class, activation='softmax')(top_layer)        
        # Combine the pre-trained and top layers
        model = Model(inputs=conv_base.input, outputs=output_layer)
        model.summary()
    else:
        print("Undefined model backbone")

    return model


def get_sin(cam_con, weights):
    """Interface for getting the neural network with already trained weights.
    The path to the best weights from the trained network must be provided

    Args:
        cam_con (str): indicates single or multi camera configuration used.
        weights (str): location of pre-trained weights

    Returns:
        Model: loaded SIN model (for external interfacing)
    """
    # neural network base of SIN
    conv_base = VGG16(include_top=False,
                      weights= None, input_shape=(224,224,3))
    # as per the fine-tune config, this has to be modified
    for layer in conv_base.layers[:-4]:
        layer.trainable = False
    # Top Model for SIN
    top_layer = conv_base.output
    top_layer = Flatten()(top_layer)
    top_layer = Dense(1024, activation='relu')(top_layer)
    # Dropout to avoid overfitting
    top_layer = Dropout(0.2)(top_layer)
    output_layer = Dense(5, activation='softmax')(top_layer)        
    # Combine the pre-trained and top layers
    model = Model(inputs=conv_base.input, outputs=output_layer)
    # load the pre-trained weights
    if cam_con =='multi':
        # placeholder for multi-cam config weights
        model.load_weights(filepath='')
    elif cam_con == 'single':
        model.load_weights(filepath=weights)
    # the network summary of reloaded SIN
    # model.summary() only for debugging
    
    return model