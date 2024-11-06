# ==============================================================================
#
# Copyright 2023 IAS, Uni Stuttgart
#
# make_dataset python module is used for loading the SIN dataset 
# All parameters related to the training. file directory and all necessary
# detail is available in the config files in the models directory.
# ==============================================================================
"""make_dataset python module is used for loading the SIN dataset.
All parameters related to the training. file directory and all necessary 
detail is available in the config files in the models directory.

"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input


def prepare(file_dir, batch_sz, shuffl):
    """Interface for loading a dataset from directory.

    Args:
        file_dir (str): directory of the dataset
        batch_sz (int): batch size
        shuffl (bool): shuffles the data in the directory

    Returns:
        loaded dataset: preprocessed and loaded dataset
    """
    # constructor for data generator with VGG16 preprocessing
    gen_data = ImageDataGenerator(vertical_flip=True, preprocessing_function=preprocess_input)
    # create an iterator for progressively loading images
    print("File Directory being loaded:", file_dir)
    dataset_mod = gen_data.flow_from_directory(directory=file_dir, class_mode='categorical', 
                                               batch_size=batch_sz, target_size=(224,224),seed=42, shuffle=shuffl)
    # Sample output regarding number of images in each category must be displayed as per the API docs
    return dataset_mod