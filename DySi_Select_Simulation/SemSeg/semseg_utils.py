# ==============================================================================
#
# Copyright 2023 IAS, Uni Stuttgart
#
# semseg_utils python module is used to define helper interfaces 
# for sem_seg module
# inspired from: https://keras.io/examples/vision/deeplabv3_plus/
# ==============================================================================
"""semseg_utils python module is used to define helper interfaces for sem_seg module.

"""

import os
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

from datetime import datetime
from SemSeg.deeplabv3plus import Deeplabv3


def get_network():
    """Interface for obtaining the DeepLabv3+ architecture for semantic segmentation with
    pre-trained weights trained on CityScapes.

    Returns:
        Model: instance of the DeepLabv3+ network
    """
    # get the network architecture DeepLabv3+
    deeplab = Deeplabv3(input_shape=(512,512,3),
                            classes=19,
                            backbone="xception",
                            weights="cityscapes",
                            activation="softmax")
    i = tf.keras.layers.Input((512,512,3))
    o = deeplab(i)

    return tf.keras.models.Model(i, o)


def read_image(image):
    """Interface for converting the input image to a preprocessed tensor.

    Args:
        image (str): location of the input image

    Returns:
        tensor: the preprocessed tensor form of parsed image
    """
    img = tf.image.decode_png(image, channels=3)
    img.set_shape([None,None,3])
    img = tf.image.resize(images=img, size=[512,512])
    img = tf.keras.applications.xception.preprocess_input(x=img)
    return img


def cs_label_colormap():
    """Interface for creating a label colormap used in Cityscapes segmentation benchmark.

    Returns:
        dict: A colormap for visualizing segmentation results
    """
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    colormap[10] = [70, 130, 180]
    colormap[11] = [220, 20, 60]
    colormap[12] = [255, 0, 0]
    colormap[13] = [0, 0, 142]
    colormap[14] = [0, 0, 70]
    colormap[15] = [0, 60, 100]
    colormap[16] = [0, 80, 100]
    colormap[17] = [0, 0, 230]
    colormap[18] = [119, 11, 32]
    return colormap


def decode_semsegmsk(mask, colormap):
    """Interface for decoding the predictions as per the Cityscapes scheme.

    Args:
        mask (array): mask to be used for semantic segmentation
        colormap (dict): color map for cityscapes palette

    Returns:
        array: A colormap for visualizing segmentation results
    """
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, 18):
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


def get_overlay(image, pred_mask):
    """Interface for getting colored overlay on the image.

    Args:
        image (array): image as numpy array
        pred_mask (array): predicted mask for overlaying on the image

    Returns:
        array: overlay array for segmenting an image
    """
    image = tf.keras.preprocessing.image.array_to_img(image)
    image = np.array(image).astype(np.uint8)
    ovlay = cv2.addWeighted(image, 0.35, pred_mask, 0.65, 0)
    return ovlay


def plot_samples(display_ls, figsize):
    """Interface for plotting the images and saving to disk.

    Args:
        display_ls (list): number of columns on the image
        figsize (array): size of the figure in the plotted image
    """
    op_dir = 'SemSeg/output'
    _, axes = plt.subplots(nrows=1, ncols=len(display_ls), figsize=figsize)
    for i in range(len(display_ls)):
        if display_ls[i].shape[-1] == 3:
            axes[i].imshow(tf.keras.preprocessing.image.array_to_img(display_ls[i]))
        else:
            axes[i].imshow(display_ls[i])
    sample_file = f"SegLog-{datetime.now():%Y-%m-%d %H%M%S}.png"
    plt.savefig(os.path.join(op_dir,sample_file), format='png')
    

def save_eval(outputDir):
    """Interface for saving the images for further processing and evaluation.

    Args:
        outputDir (str): output folder directory

    Returns:
        str: the file name is appended to output directory
    """
    # Get the current date
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Create the date-based folder
    date_folder = os.path.join(outputDir, current_date)
    os.makedirs(date_folder, exist_ok=True)

    # Find the highest existing file number within the run folder
    existing_files = [f for f in os.listdir(date_folder) if f.startswith("SemSegLog-") and f.endswith(".png")]
    existing_file_numbers = [int(f.split("-")[1].split(".")[0]) for f in existing_files]
    highest_file_number = max(existing_file_numbers, default=0)

    # Generate the new file number
    new_file_number = highest_file_number + 1
    new_file_number_padded = str(new_file_number).zfill(4)

    # Create the filename
    file_name = f"SemSegLog-{new_file_number_padded}.png"

    # Create the full file path within the new run folder
    file_path = os.path.join(date_folder, file_name)
    return file_path