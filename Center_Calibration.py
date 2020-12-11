# -*- coding: utf-8 -*-
"""
##############################################################################
#
#   MODULE DESCRIPTION:
#        This module performs a calibration of the center of the platform.
#        From an image with only the platform with the laser turned on, the 
#        center of the platform is "manually" selected and its 3D coordinates
#        are estimated.
#
#   PRE-REQUISITES: 
#        Image of the platform with no object in it, laser must be ON.
#        Scanner calibration: 4x3 calibrated matrix must be available, 
#               it allows the estimation of 3D coordinates from image 
#               coordinates. 'ScannerCalibration.npz' must be in same folder.
#        Camera calibration: The intrinsic matrix of the camera and the 
#               distortion coefficients vector must be available. 
#               'CameraCalibration.npz' must be in the same folder.
#
#   OUTPUTS:
#        Center coordinates: [X,Y,Z] coordinates of the chosen center are 
#                            obtained.
#        
#   USAGE:
#         1. Change "center_file" variable with the direction of image to be
#                used in calibration.
#         2. Click in the desired center.
#         3. Note the XYZ printed values of the center point.
#    
#    ACTION:                             DATE:           NAME:
#        First implementation and tests      20/Nov/2020      David Calles
#        Code comments and last review       8/Dic/2020       David Calles           
#
##############################################################################
"""
#----------------------------------------------------------------------------#
# ----------------------- REQUIRED EXTERNAL PACKAGES ------------------------#
#----------------------------------------------------------------------------#
import cv2
import numpy as np

#----------------------------------------------------------------------------#
# ----------------------- REQUIRED OWN FUNCTIONS-----------------------------#
#----------------------------------------------------------------------------#
from Algorithms import Simple_Image_Correction
from Algorithms import Calculate_3D_Points

#----------------------------------------------------------------------------#
# ----------------------- REQUIRED VARIABLES/FUNCTIONS ----------------------#
#----------------------------------------------------------------------------#

# DEFINE image for center calibration
center_file = "Center_Calibration_Set/CenterA.png"

# LOAD 4x3 transformation matrix from previous calibration
with np.load('ScannerCalibration.npz') as file:
    LaserMtx_4x3, _, _, _, _ = [file[i] for i in (
        'Mtx_4x3','points2D','points3D','error', 'used_imgs')]

# INITIATE center values    
cx=0
cy=0

# DEFINE callback function for mouse to get x,y coordinates
def click_event(event, x, y, flags, params): 
    global cx,cy
    # CHECK for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN:   
        # DISPLAY selected coordinates
        print(x, ' ', y)
        # DRAW selected center in image
        cv2.circle(imgA, (x,y), 10, (0,0,255), -1)
        cv2.imshow('image', imgA)
        cx=x
        cy=y 
    # CHECK for right mouse clicks      
    if event==cv2.EVENT_RBUTTONDOWN:   
        # displaying the coordinates 
        # on the Shell 
        print(x, ' ', y) 
        cx=x
        cy=y

#----------------------------------------------------------------------------#
# --------------------------- CENTER CALIBRATION ----------------------------#
#----------------------------------------------------------------------------#
        
# READ  image 
img = cv2.imread(center_file) 

# UNDISTORT image
imgA = Simple_Image_Correction(img)

# DISPLAY  image 
cv2.imshow('image', imgA) 

# SET mouse-click callback function
cv2.setMouseCallback('image', click_event) 

# WAIT for key to be pressed and close window
cv2.waitKey(0) 
print("Cx,Cy:", cx,cy)
cv2.destroyAllWindows() 

# ESTIMATE 3D coordinate of selected center point
center_3D = Calculate_3D_Points(LaserMtx_4x3, np.array([[cy,cx],[0,0]]))
print("[X,Y,Z] coordinates of center point:")
print(center_3D[0,:])

# Results should be more or less similar to:
# APROXIMATE VALUE =  [48.52758569  135.39777659  578.44508265]