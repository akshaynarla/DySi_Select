# !/usr/bin/env python3
# ==============================================================================
#
# Copyright 2023 IAS, Uni Stuttgart
#
# evaluate_real_dataset python module is used to simulate the frames of real-world
# dataset to create a continuous flow of frames and test the DySi_Select 
# concept for real-world data.
# ==============================================================================
"""evaluate_real_dataset python module is used to simulate the frames of real-world
dataset to create a continuous flow of frames and test the DySi_Select 
concept for real-world data.
"""
import cv2
import os
import argparse
import time

from datetime import datetime
from SemSeg.sem_seg import get_network_semseg, predict_semseg
from SIN.src.predict_situation import predict_situation, get_network_sin
from SIN.src.utils.data_utils import load_image
from Cam2BEV.predict_bev import predict_bev, get_bev_network


def main():
    """Interface for parsing the CLI arguments and run the evaluation on public dataset
    """
    # parse argument from CLI
    argparser = argparse.ArgumentParser(
        description='DySi_Select Evaluation with a real dataset')
    argparser.add_argument(
        '-v', '--vis',
        metavar='V',
        default=True,
        help='Visualize intermediate products on a cv2 window')
    argparser.add_argument(
        '-o', '--op',
        metavar='O',
        default="EvalDatasets/Cityscapes_mini/stuttgart_00",
        type= str,
        help='Input folder path')
    argparser.add_argument(
        '-w1', '--weight1',
        metavar='W1',
        default='Cam2BEV/model/output/unetxst_singlecam/finetuned/Checkpoints/best_weights.hdf5',
        type= str,
        help='Path to parse Cam2BEV model weight')
    argparser.add_argument(
        '-b', '--backbone',
        metavar='B',
        default='unetxst',
        type= str,
        help='Selection of Cam2BEV model')
    argparser.add_argument(
        '-w2', '--weight2',
        metavar='W2',
        default='SIN/src/output/singlecam/stored_cp/Checkpoints/best_weights.hdf5',
        type= str,
        help='Path to parse SIN model weight')
    args = argparser.parse_args()
    # Directory containing the frames/images
    frame_directory = args.op
    flow_frames(frame_directory, args)


def render_window():
    """Interface for initializing the display windows
    """
    cv2.namedWindow('Video Playback', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video Playback', 400, 300)
    cv2.namedWindow('SemSeg Playback', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('SemSeg Playback', 400, 300)
    cv2.namedWindow('BEV Playback', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('BEV Playback', 400, 300)

def data_select(sitn_cls):
    """Sample interface for selecting relevant data.
    This needs additional tracking and distance calculation algorithms.

    Args:
        sitn_cls (str): identified situation class

    Returns:
        str: relevant data in a particular situation
    """
    if sitn_cls == "FreeDrive":
        tx_data = "Long range data(far away objects, if any)" 
    elif sitn_cls == "FreeDriveParkedVehicles":
        tx_data = "Long range data"
    elif sitn_cls == "FreeIntersection":
        tx_data = "Short range (objects near and in intersection)"
    elif sitn_cls == "OccludedDrive":
        tx_data = "Short range (mainly occluding objects)"
    elif sitn_cls == "OccludedIntersection":
        tx_data = "Short range (mainly occluding objects)"
    else:
        tx_data = "Send all data as per ETSI standard"    
    return tx_data


def flow_frames(frame_dir, argument):
    """Interface for obtaining continuous flow of frames to simulate a 
    real-life automotive scenario using a publicly available dataset

    Args:
        frame_dir (str): directory of the dataset with image frames
        argument (config): parsed arguments from CLI
    """
    # Get the list of image files in the directory
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png') or f.endswith('.jpg')])

    # Define font parameters
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.5
    font_color = (0, 0, 0)  # White color in BGR
    thickness = 2
    
    # Load necessary prediction model backbone
    semseg_nw = get_network_semseg()
    bev_nw, label = get_bev_network(backbone=argument.backbone,weights_dir=argument.weight1)
    sin_nw = get_network_sin(weights_dir=argument.weight2)
    
    # Initialize the display windows
    render_window()
    frame_num = 0
    inference_times = []
    # Loop through each image file and play them back as frames
    for frame_file in frame_files:
        # Read RGB frames
        frame_path = os.path.join(frame_dir, frame_file)
        frame = cv2.imread(frame_path)
        frame_num += 1
        
        # for every 10th frame, calculate the SIN
        if frame_num % 10 == 0:
            # start inference timer
            start_time = time.time()
            # parse the RGB frames to sem seg module
            frame2, sem_loc = predict_semseg(semseg_nw, frame_path)
            frame2 = cv2.imread(sem_loc)
            
            # parse the sem seg images to BEV prediction
            frame3, bev_loc = predict_bev(model=bev_nw, one_hotd_label=label,
                                        sem_img=sem_loc, backbone=argument.backbone)
            frame3 = cv2.imread(bev_loc)

            # predict the overall situation of the ego-vehicle
            bev = load_image(bev_loc)
            sit_pred = predict_situation(model= sin_nw, bev_img=bev)
        
            # end inference timer
            end_time = time.time()
            inf_time = end_time - start_time
            inference_times.append(inf_time)
            
            # DySi_Select basic Implementation call
            data_tx = data_select(sit_pred)
            
            # Put text on the image
            cv2.putText(frame3, sit_pred, (10, 25), font, font_scale, font_color, thickness)
            cv2.putText(frame3, data_tx, (10,45), font, font_scale, font_color, thickness)
        
        # save output for evaluation
        outputDir = os.path.abspath('eval_output')
        current_date = datetime.now().strftime("%Y-%m-%d")
        date_folder = os.path.join(outputDir, current_date)
        os.makedirs(date_folder, exist_ok=True)
        # process the frames and use it to visualize each step
        if frame is not None:
            cv2.imshow('Video Playback', frame)
            if frame_num % 10 == 0:
                cv2.imshow('SemSeg Playback', frame2)
                cv2.imshow('BEV Playback', frame3)
            
                # Create the filename
                file_name = f"SIN_{frame_file}"
                # Create the full file path within the new run folder
                file_path = os.path.join(date_folder, file_name)
                cv2.imwrite(file_path, frame3)
                frame_num = 0
            if cv2.waitKey(25) & 0xFF == 27:  # Exit when 'Esc' key is pressed
                break
    
    # Calculate and display the average inference time
    average_inference_time = sum(inference_times) / len(inference_times)
    print(f"Average Inference Time of the entire pipeline: {average_inference_time:.4f} seconds")   
    # Release the display window
    cv2.destroyAllWindows()
    
# ==============================================================
# Start of program
# ==============================================================
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
