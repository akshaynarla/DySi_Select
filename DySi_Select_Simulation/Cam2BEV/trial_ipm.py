# ==============================================================================
#
# Copyright 2023 IAS, Uni Stuttgart
#
# get_ipm python module is used to convert the  semantic segmented image to 
# a BEV homography image. This is necessary if the Cam2BEV is based on 
# DeepLab-Mobilenet
# ==============================================================================
"""trial_ipm module can be used to test and play around homography.
"""

import os
import yaml
import numpy as np

class Camera:

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
    

def homography_conv(old):
    oldInputResolution = [1216,1936]
    oldOutputResolution = [968,1936]
    newInputResolution = [512,512]
    newOutputResolution = [256,512]

    # read original homography
    cvH = np.array(old)

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
 
    
# load camera configurations and drone config for BEV transformation
with open(os.path.abspath('preprocessing/camera_configs/2_F/front.yaml')) as stream:
    cameraConfig = yaml.safe_load(stream)
with open(os.path.abspath('preprocessing/camera_configs/2_F/drone.yaml')) as stream:
    droneConfig = yaml.safe_load(stream)
    
# init camera objects
cam = Camera(cameraConfig)
drone = Camera(droneConfig)
    
# calculate output shape; adjust to match drone image
outputRes = (int(2 * droneConfig["py"]), int(2 * droneConfig["px"]))
print("Output Resolution:", outputRes)
dx = outputRes[1] / droneConfig["fx"] * droneConfig["ZCam"]
dy = outputRes[0] / droneConfig["fy"] * droneConfig["ZCam"]
pxPerM = (outputRes[0] / dy, outputRes[1] / dx)
# setup mapping from street/top-image plane to world coords
shift = (outputRes[0] / 2.0, outputRes[1] / 2.0)
shift = shift[0] + droneConfig["YCam"] * pxPerM[0], shift[1] - droneConfig["XCam"] * pxPerM[1]
M = np.array([[1.0 / pxPerM[1], 0.0, -shift[1] / pxPerM[1]], [0.0, -1.0 / pxPerM[0], shift[0] / pxPerM[0]], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
print("Transformation Matrix M:")
print(M)
print("Camera Projection Matrix is:")
print(cam.P)
print("Camera Intrinsic is:")
print(cam.K)
print("Camera Rotation:")
print(cam.R)
print("Camera Translation:")
print(cam.t)
IPM = np.linalg.inv(cam.P.dot(M))
newIPM, opRes = homography_conv(IPM)
    
print(f"OpenCV homography for front camera:")
print(IPM.tolist())
print(f"New OpenCV homography for front camera:")
print(newIPM.tolist())
   
mask = np.zeros((opRes[0], opRes[1], 3), dtype=bool)
for i in range(opRes[1]):
    for j in range(opRes[0]):
        theta = np.rad2deg(np.arctan2(-j + opRes[0] / 2 - droneConfig["YCam"] * pxPerM[0], i - opRes[1] / 2 + droneConfig["XCam"] * pxPerM[1]))
        if abs(theta - cameraConfig["yaw"]) > 90 and abs(theta - cameraConfig["yaw"]) < 270:
            print("hits here")
            mask[j,i,:] = True
print("Theta:", theta)
print("New op shape is:", opRes)

