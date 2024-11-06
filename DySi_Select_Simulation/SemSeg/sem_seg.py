# ==============================================================================
#
# Copyright 2023 IAS, Uni Stuttgart
#
# sem_seg python module is used to get semantic segmentation of any input frame
# and will be useful when the model is applied for real-world dataset. 
# The semantic segmented image will be used as an input to the Cam2BEV module.
# Handling of one-hot encoding is done in Cam2BEV itself.
# ==============================================================================
"""sem_seg python module is used to get semantic segmentation of any input frame
and will be useful when the model is applied for real-world dataset. 
The semantic segmented image will be used as an input to the Cam2BEV module.
Handling of one-hot encoding is done in Cam2BEV itself.

"""

import os
import time
import numpy as np
import tensorflow as tf
import cv2

from SemSeg.semseg_utils import get_network, read_image, decode_semsegmsk, cs_label_colormap, get_overlay, plot_samples, save_eval


def get_network_semseg():
    """Interface for loading the semantic segmentation model.

    Returns:
        Model: pre-trained semantic segmentation network model
    """
    # get the semantic segmentation network
    model = get_network()
    return model
    

def predict_semseg(model, original_img):
    """Interface for obtaining the semantically segmented image from a 
    SemSeg network and preprocess RGB image.

    Args:
        model (Model): pre-trained model instance of semantic segmentation network
        original_img (str): location of the input RGB image frame

    Returns:
        str: location of semantically segmented image frame
    """
    # pre-process the parsed image to a default size
    imgs = tf.io.read_file(original_img)
    inp_tensor = read_image(imgs)   
    preproc_tensor = np.expand_dims(inp_tensor, axis=0)
    
    #get the semantic segmentation prediction
    start_time = time.time()
    semseg_msk = model.predict(preproc_tensor)
    end_time = time.time()
    inf_time = end_time - start_time
    print("Prediction/Inference time for SemSeg:", inf_time)
    semseg_msk = np.squeeze(semseg_msk)
    semseg_msk = np.argmax(semseg_msk, axis=2)
    
    # decode the segmentation masks and get colored overlay images
    semseg_colormap = decode_semsegmsk(mask=semseg_msk, colormap= cs_label_colormap())
    overlay = get_overlay(image=inp_tensor, pred_mask=semseg_colormap)
    
    # save output for evaluation
    outputDir = os.path.abspath('SemSeg/seg_output')
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    op_loc = save_eval(outputDir)
    cv2.imwrite(op_loc, cv2.cvtColor(semseg_colormap, cv2.COLOR_RGB2BGR))
    
    # plot the images for evaluation and save to disk
    plot_samples([inp_tensor, overlay, semseg_colormap], figsize=(18,14))
    # returns the input necessary for Cam2BEV
    return semseg_colormap, op_loc