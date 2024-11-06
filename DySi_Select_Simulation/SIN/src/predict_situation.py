# ==============================================================================
#
# Copyright 2023 IAS, Uni Stuttgart
#
# predict_situation python module is used to provide interfaces 
# training the neural network SIN (Situation Identification Network). 
# All parameters related to the evaluation the network is available 
# in the config files in the models directory.
# ==============================================================================
"""predict_situation python module is used to provide interfaces training the neural network SIN (Situation Identification Network). 
All parameters related to the evaluation the network is available in the config files in the models directory.

"""

import tensorflow as tf
import time

from SIN.src.models.arch.vggnet import get_sin
from SIN.src.utils.SIN_utils import get_label
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input


def get_network_sin(cam_set='single', 
                    weights_dir='SIN/src/output/singlecam/stored_cp/Checkpoints/best_weights.hdf5'):
    """Interface for loading the SIN model.

    Args:
        cam_set (str, optional): camera setting used for evaluating the situation. Defaults to 'single'.
        weights_dir (str, optional): file path to the pre-trained SIN weights. Defaults to 'SIN/src/output/singlecam/stored_cp/Checkpoints/best_weights.hdf5'.

    Returns:
        Model: loaded SIN model
    """
    # Load the already trained neural network
    model = get_sin(cam_set, weights_dir)
    print(f"Reloaded already trained model")
    
    return model


def predict_situation(model, bev_img):
    """Interface for predicting the situation for a BEV image. 
    The interface can take an image from single image source, 
    from CARLA or from a frame of real-world dataset

    Args:
        model (Model): pre-trained SIN model instance
        bev_img (str): lcoation of the bev image

    Returns:
        str: predicted situation class
    """
    # preprocess the bev image as per the architecture
    # backbone used. Here vggnet is used and hence reduced 
    # to (224,224,3)
    bev_array = img_to_array(bev_img)
    bev_array_resize = tf.image.resize(bev_array, (224,224))
    bev_array_resize = tf.expand_dims(bev_array_resize, axis=0)
    preprocess_bev = preprocess_input(x=bev_array_resize)
    
    # run prediction of situation after training the model 
    # or with pre-trained weights
    start_time = time.time()
    prediction = model.predict(preprocess_bev)
    end_time = time.time()
    inf_time = end_time - start_time
    pred_label_index = tf.argmax(prediction, axis=1).numpy()[0]
    print("Prediction result:", pred_label_index)
    print("Prediction/Inference time:", inf_time)
    pred_class = get_label().get(pred_label_index, "Unknown")
    # return the predicted class
    return pred_class