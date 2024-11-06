# ==============================================================================
#
# Copyright 2023 IAS, Uni Stuttgart
#
# predict_bev python module is used to predict the bev from 
# a semantically segmented input image.
# ==============================================================================

"""predict_bev python module is used to predict the bev from a semantically segmented input image.
This is a newly developed module for the functionality of DySi_Select.
This module helps in verifying the situation identification in vehicles. 

"""
import os
import time
import numpy as np
import tensorflow as tf
import cv2

from datetime import datetime
from Cam2BEV.model.utils import one_hot_decode_image, parse_convert_xml, load_module, load_image,load_image_op, resize_image_op, one_hot_encode_image_op
from Cam2BEV.get_ipm import get_homography

one_hot_palette_input = parse_convert_xml('Cam2BEV/model/one_hot_conversion/convert_10.xml')

def parse_sample(input_file):
    """Interface for parsing the input image.

    Args:
        input_file (str): location of the parsed image

    Returns:
        numpy array: pre-processed image as a numpy array
    """
    inputs=[]
    # parse and process input images
    input_file = load_image_op(input_file)
    input_file = resize_image_op(input_file, fromShape=[],toShape=[256,512], cropToPreserveAspectRatio=False ,
                                 interpolation=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # works correctly until here with correct image being loaded (256,512,3)
    input_file = np.array(input_file) 
    # color scheme extended as per the one-hot conversion file (256,512,10)
    input_file = one_hot_encode_image_op(input_file, one_hot_palette_input) 
    inputs.append(input_file)
    return inputs[0]


def save_bev_eval(outputDir):
    """Interface for saving the images for further processing and evaluation.

    Args:
        outputDir (str): location of the output directory

    Returns:
        str: file name appended to the output directory
    """
    # Get the current date
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Create the date-based folder
    date_folder = os.path.join(outputDir, current_date)
    os.makedirs(date_folder, exist_ok=True)

    # Find the highest existing file number within the run folder
    existing_files = [f for f in os.listdir(date_folder) if f.startswith("BEVLog-") and f.endswith(".png")]
    existing_file_numbers = [int(f.split("-")[1].split(".")[0]) for f in existing_files]
    highest_file_number = max(existing_file_numbers, default=0)

    # Generate the new file number
    new_file_number = highest_file_number + 1
    new_file_number_padded = str(new_file_number).zfill(4)

    # Create the filename
    file_name = f"BEVLog-{new_file_number_padded}.png"

    # Create the full file path within the new run folder
    file_path = os.path.join(date_folder, file_name)
    return file_path


def get_bev_network(backbone='unetxst', 
                    weights_dir='Cam2BEV/model/output/unetxst_singlecam/finetuned/Checkpoints/best_weights.hdf5'):
    """Interface for obtaining the bev generation neural network model.

    Args:
        backbone (str, optional): network architecture backbone to be used for bev generation.
        2 options: unetxst or deeplab. Defaults to 'unetxst'.
        weights_dir (str, optional): location of the pre-trained model weights. 
        Defaults to 'Cam2BEV/model/output/unetxst_singlecam/finetuned/Checkpoints/best_weights.hdf5'.

    Returns:
        model: the loaded model based on the parsed inputs
    """    
    # pre-processing for data
    one_hot_palette_label = parse_convert_xml('Cam2BEV/model/one_hot_conversion/convert_9+occl.xml')
    n_classes_input = len(one_hot_palette_input)
    n_classes_label = len(one_hot_palette_label)
    
    # get the Cam2BEV network with necessary parameters
    if backbone == 'deeplab':
        arch = load_module('Cam2BEV/model/architecture/deeplab_mobilenet.py')
        model = arch.get_network(input_shape=(256,512,n_classes_input), n_output_channels=n_classes_label)
        model.load_weights(weights_dir)
        print(f"Reloaded deeplab model from: {weights_dir}")
    elif backbone == 'unetxst':
        # get the Cam2BEV network with necessary parameters
        arch = load_module('Cam2BEV/model/architecture/uNetXST.py')
        homography = load_module('Cam2BEV/preprocessing/homography_converter/uNetXST_homographies/2_F.py')
        model = arch.get_network(input_shape=(256,512,n_classes_input), n_output_channels=n_classes_label,
                                n_inputs=1, thetas = homography.H)
        # model.summary() # only for debug purpose
        model.load_weights(weights_dir)
        print(f"Reloaded uNetXST model from: {weights_dir}") 
    
    return model, one_hot_palette_label


def predict_bev(model, one_hotd_label, sem_img, backbone = 'unetxst'):
    """Interface for obtaining the bev from a semantically segmented image.

    Args:
        model (Model): parsed Cam2BEV model
        one_hotd_label (list): one-hot encoded color palette for converting CityScapes color to 4 classes.
        sem_img (str): location of the input semantic segmented image
        backbone (str, optional): network backbone to be used. Ensure same backbone is 
        parsed as that of the model. Defaults to 'unetxst'.

    Returns:
        str: location of the predicted bev image
    """    
    if backbone == 'deeplab':
        # projective processing (using IPM)
        sem_img = get_homography(sem_img)

    # pre-process the parsed image to a default size
    imgs = parse_sample(sem_img)
    imgs = np.expand_dims(imgs, axis=0)  # to match the input shape of network
    
    #get the BEV prediction
    start_time = time.time()
    bev_prediction = model.predict(imgs).squeeze()
    # convert the bev prediction to the palette specified
    bev_prediction = one_hot_decode_image(bev_prediction, one_hotd_label)
    end_time = time.time()
    inf_time = end_time - start_time
    print("Prediction/Inference time for BEV Generation:", inf_time)
    
    # plot the images for evaluation and write to disk
    if backbone == 'deeplab':
        outputDir = os.path.abspath('Cam2BEV/bev_output/deeplab')
    elif backbone == 'unetxst':
        outputDir = os.path.abspath('Cam2BEV/bev_output/unetxst')
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    op_loc = save_bev_eval(outputDir)
    cv2.imwrite(op_loc, cv2.cvtColor(bev_prediction, cv2.COLOR_RGB2BGR))
    
    # returns the input necessary for SIN
    return bev_prediction, op_loc