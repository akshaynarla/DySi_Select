# ==============================================================================
#
# Copyright 2023 IAS, Uni Stuttgart
#
# get_ipm python module is used to convert the  semantic segmented image to 
# a BEV homography image. This is necessary if the Cam2BEV is based on 
# DeepLab-Mobilenet
# ==============================================================================

"""get_ipm python module is used to convert the semantic segmented image to 
a BEV homography image. This is necessary if the Cam2BEV is based on DeepLab-Mobilenet 
network backbone

"""

import os
import sys
import yaml
import numpy as np
import cv2

from datetime import datetime


class Camera:
  """ Camera class instantiates a camera based on the parsed configuration.
  
  Camera class is used to set the intrinsic and extrinsic parameters
  for the cameras involved.
  """
  K = np.zeros([3, 3])
  R = np.zeros([3, 3])
  t = np.zeros([3, 1])
  P = np.zeros([3, 4])

  def setK(self, fx, fy, px, py):
    self.K[0, 0] = fx
    self.K[1, 1] = fy
    self.K[0, 2] = px
    self.K[1, 2] = py
    self.K[2, 2] = 1.0

  def setR(self, y, p, r):

    Rz = np.array([[np.cos(-y), -np.sin(-y), 0.0], [np.sin(-y), np.cos(-y), 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[np.cos(-p), 0.0, np.sin(-p)], [0.0, 1.0, 0.0], [-np.sin(-p), 0.0, np.cos(-p)]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(-r), -np.sin(-r)], [0.0, np.sin(-r), np.cos(-r)]])
    Rs = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]) # switch axes (x = -y, y = -z, z = x)
    self.R = Rs.dot(Rz.dot(Ry.dot(Rx)))

  def setT(self, XCam, YCam, ZCam):
    X = np.array([XCam, YCam, ZCam])
    self.t = -self.R.dot(X)

  def updateP(self):
    Rt = np.zeros([3, 4])
    Rt[0:3, 0:3] = self.R
    Rt[0:3, 3] = self.t
    self.P = self.K.dot(Rt)

  def __init__(self, config):
    self.setK(config["fx"], config["fy"], config["px"], config["py"])
    self.setR(np.deg2rad(config["yaw"]), np.deg2rad(config["pitch"]), np.deg2rad(config["roll"]))
    self.setT(config["XCam"], config["YCam"], config["ZCam"])
    self.updateP()
    
 
def get_homography(semntc_img):
  """Interface for getting homography image for the input semantically segmented image.
  This is suitable currently for one camera only(i.e. Front).
  For adapting to multiple cameras, refer ipm.py and make changes.

  Args:
      semntc_img (str): the semantic segmented image to be transformed

  Returns:
      str: the homography image for DeepLab-Cam2BEV
  """
  # load camera configurations and drone config for BEV transformation
  with open(os.path.abspath('Cam2BEV/preprocessing/camera_configs/2_F/front.yaml')) as stream:
    cameraConfig = yaml.safe_load(stream)
  with open(os.path.abspath('Cam2BEV/preprocessing/camera_configs/2_F/drone.yaml')) as stream:
    droneConfig = yaml.safe_load(stream)
    
  # read the semantically segmented image
  input = cv2.imread(semntc_img)
  
  # init camera objects
  cam = Camera(cameraConfig)
  drone = Camera(droneConfig)
    
  # calculate output shape; adjust to match drone image
  outputRes = (int(2 * droneConfig["py"]), int(2 * droneConfig["px"]))
  dx = outputRes[1] / droneConfig["fx"] * droneConfig["ZCam"]
  dy = outputRes[0] / droneConfig["fy"] * droneConfig["ZCam"]
  pxPerM = (outputRes[0] / dy, outputRes[1] / dx)

  # setup mapping from street/top-image plane to world coords
  shift = (outputRes[0] / 2.0, outputRes[1] / 2.0)
  shift = shift[0] + droneConfig["YCam"] * pxPerM[0], shift[1] - droneConfig["XCam"] * pxPerM[1]
  M = np.array([[1.0 / pxPerM[1], 0.0, -shift[1] / pxPerM[1]], [0.0, -1.0 / pxPerM[0], shift[0] / pxPerM[0]], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
  IPM = np.linalg.inv(cam.P.dot(M))
    
  # print(f"OpenCV homography for front camera:")
  # print(IPM.tolist())
  print("Calculating new homography for the parsed image resolution....")
  newIPM, outputRes = homography_conv(IPM, input.shape)  
  # print(newIPM.tolist())
    
  mask = np.zeros((outputRes[0], outputRes[1], 3), dtype=bool)
  for i in range(outputRes[1]):
    for j in range(outputRes[0]):
      theta = np.rad2deg(np.arctan2(-j + outputRes[0] / 2 - droneConfig["YCam"] * pxPerM[0], i - outputRes[1] / 2 + droneConfig["XCam"] * pxPerM[1]))
      if abs(theta - cameraConfig["yaw"]) > 90 and abs(theta - cameraConfig["yaw"]) < 270:
        mask[j,i,:] = True
    
  interpMode = cv2.INTER_NEAREST
  warpedImage = cv2.warpPerspective(input, newIPM, (outputRes[1], outputRes[0]), flags=interpMode)
  
  birdsEyeView = np.zeros(warpedImage.shape, dtype=np.uint8)
  mask = np.any(warpedImage != (0,0,0), axis=-1)
  birdsEyeView[mask] = warpedImage[mask]
    
  outputDir = os.path.abspath('Cam2BEV/homography_output')
  if not os.path.exists(outputDir):
    os.makedirs(outputDir)
  op_loc = save_ipm_eval(outputDir)
  cv2.imwrite(op_loc, birdsEyeView)
    
  return op_loc
  

def save_ipm_eval(outputDir):
  """Interface for saving the images for further processing and evaluation.

  Args:
      outputDir (str): output directory where the images are to be saved

  Returns:
      str: name of the file to be stored appended to the parsed directory
  """
  # Get the current date
  current_date = datetime.now().strftime("%Y-%m-%d")

  # Create the date-based folder
  date_folder = os.path.join(outputDir, current_date)
  os.makedirs(date_folder, exist_ok=True)

  # Find the highest existing file number within the run folder
  existing_files = [f for f in os.listdir(date_folder) if f.startswith("IPMLog-") and f.endswith(".png")]
  existing_file_numbers = [int(f.split("-")[1].split(".")[0]) for f in existing_files]
  highest_file_number = max(existing_file_numbers, default=0)

  # Generate the new file number
  new_file_number = highest_file_number + 1
  new_file_number_padded = str(new_file_number).zfill(4)

  # Create the filename
  file_name = f"IPMLog-{new_file_number_padded}.png"

  # Create the full file path within the new run folder
  file_path = os.path.join(date_folder, file_name)
  return file_path


def homography_conv(oldIPM, imgShape):
  """Interface for converting the IPM homography to handle resolutions other than Cam2BEV dataset resolution.
  
  Cam2BEV dataset resolution: 1936x1216 --> 1936x968 (WxH format like in files)

  Args:
      oldIPM (array): old homogrpahy matrix (based on the Cam2BEV data configuration)
      imgShape (array): shape of the parsed input image

  Returns:
      array: new homography matrix
  """
  oldInputResolution = [1216,1936]
  oldOutputResolution = [968,1936]
  newInputResolution = imgShape
  newOutputResolution = [256,512]

  # read original homography
  cvH = np.array(oldIPM)

  # calculate intermediate shapes
  newInputAspectRatio = newInputResolution[0] / newInputResolution[1]
  newOutputAspectRatio = newOutputResolution[0] / newOutputResolution[1]
  isNewInputWide = newInputAspectRatio <= 1
  isNewOutputWide = newOutputAspectRatio <= 1
  if isNewInputWide:
      newInputResolutionAtOldInputAspectRatio = np.array((newInputResolution[1] / oldInputResolution[1] * oldInputResolution[0], newInputResolution[1]))
  else:
      newInputResolutionAtOldInputAspectRatio = np.array((newInputResolution[0], newInputResolution[0] / oldInputResolution[0] * oldInputResolution[1]))
  if isNewOutputWide:
      oldOutputResolutionAtNewOutputAspectRatio = np.array((newOutputAspectRatio * oldOutputResolution[1], oldOutputResolution[1]))
  else:
      oldOutputResolutionAtNewOutputAspectRatio = np.array((oldOutputResolution[0], oldOutputResolution[0] / newOutputAspectRatio))

  #=== introduce additional transformation matrices to correct for different aspect ratio

  # shift input to simulate padding to original aspect ratio
  px = (newInputResolutionAtOldInputAspectRatio[1] - newInputResolution[1]) / 2 if not isNewInputWide else 0
  py = (newInputResolutionAtOldInputAspectRatio[0] - newInputResolution[0]) / 2 if isNewInputWide else 0
  Ti = np.array([[ 1,  0, px],
              [ 0,  1, py],
              [ 0,  0,  1]], dtype=np.float32)

  # scale input to original resolution
  fx = oldInputResolution[1] / newInputResolutionAtOldInputAspectRatio[1]
  fy = oldInputResolution[0] / newInputResolutionAtOldInputAspectRatio[0]
  Ki = np.array([[fx,  0, 0],
              [ 0, fy, 0],
              [ 0,  0, 1]], dtype=np.float32)

  # crop away part of size 'oldOutputResolutionAtNewOutputAspectRatio' from original output resolution
  px = -(oldOutputResolution[1] - oldOutputResolutionAtNewOutputAspectRatio[1]) / 2
  py = -(oldOutputResolution[0] - oldOutputResolutionAtNewOutputAspectRatio[0]) / 2
  To = np.array([[ 1,  0, px],
              [ 0,  1, py],
              [ 0,  0,  1]], dtype=np.float32)

  # scale output to new resolution
  fx = newOutputResolution[1] / oldOutputResolutionAtNewOutputAspectRatio[1]
  fy = newOutputResolution[0] / oldOutputResolutionAtNewOutputAspectRatio[0]
  Ko = np.array([[fx,  0, 0],
              [ 0, fy, 0],
              [ 0,  0, 1]], dtype=np.float32)

  # assemble adjusted homography
  cvHr = Ko.dot(To.dot(cvH.dot(Ki.dot(Ti))))
  return cvHr, newOutputResolution
  
  
 
  