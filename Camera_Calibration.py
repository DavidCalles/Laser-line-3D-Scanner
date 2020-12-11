# -*- coding: utf-8 -*-
"""
##############################################################################
#
#   MODULE DESCRIPTION:
#        This module performs the camera calibration from a set of images
#        of a planar rectangular chessboard pattern with known dimensions.
#        Estimating an intrinsic camera matrix and a distortion coefficients
#        vector taking into account only radial and tangential distortions.
#
#   PRE-REQUISITES: 
#        Calibration images acquisition: A set of images of a chessboard. 
#                 Any number of iamges above 30-40 are recommended. The more
#                 the better.
#
#   OUTPUTS:
#        Point cloud: '.XYZ' file with xyz and rgb information coded in ascii.
#                     Visualization of resulting point cloud with minimum
#                     filtering.
#        
#   USAGE:
#         1. 
#
#    
#    ACTION:                             DATE:           NAME:
#        First implementation and tests      30/Sep/2020      David Calles
#        Code comments and last review       6/Dic/2020       David Calles           
#
##############################################################################
"""
#----------------------------------------------------------------------------#
# ----------------------- REQUIRED EXTERNAL PACKAGES ------------------------#
#----------------------------------------------------------------------------#

import numpy as np
import cv2
import glob
import random 

#----------------------------------------------------------------------------#
# ----------------------- FEATURE-ENABLING VARIABLES-------------------------#
#----------------------------------------------------------------------------#
SHOW_IMAGES = False
SAVE_IMAGES = False

#----------------------------------------------------------------------------#
# ----------------- CALIBRATION CONFIGURATION VARIABLES----------------------#
#----------------------------------------------------------------------------#

# PATH WHERE IMAGES ARE LOCATED
general_path = "Camera_Calibration_Image_Set/*.png"
test_path = "Camera_Calibration_Image_Set/Image_90.png" #Testing image
save_path = 'Camera_Calibration_Borders/' #Directory for saving borders imgs
name_num = len("Camera_Calibration_Image_Set/") #Len unitl image name

# WINDOW SIZE FOR EACH ITERATION (Random samples)
image_set_num = 20
# Number of random attempts
attempts = 200
# Other general variables
show_size = (1080,720)
summary_size = (1800, 500)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#----------------------------------------------------------------------------#
# ----------------- VARIABLES FOR SINGLE CAIBRATION VARIABLES----------------#
#----------------------------------------------------------------------------#

csize = [7,9]   # Chessboard size (in corners, not squares)

# PREPARE object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# assuming z=0 in all cases
objp = np.zeros((csize[0]*csize[1],3), np.float32)

sq_size = 20.0 #mm
objp[:,:2] = np.mgrid[0:csize[0],0:csize[1]].T.reshape(-1,2)

# To get measurements in milimeters knowing a square is 2cm
objp *= sq_size

# Important values to be found
error_vect = []
used_images = []
matrixes = []
new_matrixes = []
distortions = []
rotations = []
translations = []

#----------------------------------------------------------------------------#
# ----------------- PERFORM ALL ATTEMPTS OF CALIBRATION----------------------#
#----------------------------------------------------------------------------#
print("-------------------PERFORMING CAMERA CALIBRATION---------------------")

for k in range(attempts):
    print("Initiating attempt {}.".format(k))
    # INITIALIZE values
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob(general_path)
    random_images = random.sample(images, image_set_num)
    iteration = 0
    cv2.startWindowThread()
    succesfull = []
    failed = []
    success_count = 0
    
#----------------------------------------------------------------------------#
# ----------------- PERFORM EACH ATTEMPTS OF CALIBRATION---------------------#
#----------------------------------------------------------------------------#
    for fname in random_images:
        iteration += 1
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray_show = cv2.resize(gray, show_size)
        if SHOW_IMAGES:
            cv2.imshow('img', gray_show)
            cv2.waitKey(100)
    
        # FIND the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (csize[0],csize[1]),
                                                 None)

        # ADD object and image points if conerners were detected.
        if ret == True:
            objpoints.append(objp)
            # REFINE corners coordinates
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
    
            # DRAW and DISPLAY the corners
            imgBorders = cv2.drawChessboardCorners(img, (csize[0],csize[1]),
                                                   corners2,ret)
            img_show = cv2.resize(imgBorders, show_size)
            if SHOW_IMAGES:
                cv2.imshow(fname[name_num:],img_show)
                cv2.waitKey(100)
    
            BigImg = np.hstack([img, imgBorders])
            cv2.resize(BigImg, show_size)
            filename = save_path + fname[name_num:]
            
            # SAVE image with corners (if enabled)
            if SAVE_IMAGES:
                cv2.imwrite(filename, img)
            succesfull.append(fname[name_num:])
            success_count = success_count +1
        else:
            failed.append(fname[name_num:])
    
    
    cv2.destroyAllWindows()
    
    # ESTIMATE camera parameters
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                       imgpoints,
                                                       gray.shape[::-1],
                                                       None,None)
    # LOAD test image
    img = cv2.imread(test_path)
    h,  w = img.shape[:2]
    
    # ESTIMA new optimal camera matrix
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix=mtx,
                                                      distCoeffs=dist,
                                                      imageSize=(w,h),
                                                      alpha=1,
                                                      newImgSize=(w,h))
    
    # UNDISTORT test image
    out_img = img.copy()
    cv2.undistort(src=img, cameraMatrix=mtx, distCoeffs=dist,
                  dst=out_img, newCameraMatrix=newcameramtx)
    
    # CROP ROI of image
    x,y,w,h = roi
    crop_img = out_img[y:y+h, x:x+w]
    # cv2.imwrite('calibresult.png',dst)
    
    # RESIZE image
    h,  w = img.shape[:2]
    crop_img_2 = cv2.resize(crop_img, dsize=(w, h),
                            interpolation=cv2.INTER_LINEAR)
    
    # SHOW undistorting results
    bigImg = cv2.resize(np.hstack([img, out_img, crop_img_2]), summary_size)
    if SHOW_IMAGES:
        cv2.startWindowThread()
        cv2.namedWindow('Undistorting results', flags =   cv2.WINDOW_NORMAL |
                        cv2.WINDOW_FREERATIO)
        cv2.imshow('Undistorting results', bigImg)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    
    # CALCULATE reprojection error of single attempt
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i],
                                          tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error

    print("In attempt {}, {} images were proccessed."\
          .format(k,success_count))
        
    # ACCUMULATE all values from each attempt
    error_vect.append(mean_error/len(objpoints))
    used_images.append(succesfull)
    matrixes.append(mtx)
    new_matrixes.append(newcameramtx)
    distortions.append(dist)
    rotations.append(rvecs)
    translations.append(tvecs)

# SEARCH attempt with most and least error
maximum_error = np.amax(error_vect)
minimum_error = np.amin(error_vect)

location_max = np.argmax(error_vect)
location_min = np.argmin(error_vect)

# CALCULATE mean error
mean_error = np.mean(error_vect)

# VERBOSE results

print( "MAXIMUM ERROR FOUND: {} IN ITERATION {}." \
      .format(maximum_error, location_max) )
    
print( "MINIMUM ERROR FOUND: {} IN ITERATION {}." \
      .format(minimum_error, location_min) )
    
print( "MEAN ERROR FOUND: {} WITH WINDOWS OF {}" \
      .format(mean_error, image_set_num) )

print( "IMAGES TO BE USED: \n", used_images[location_min])

print( "CAMERA MATRIX TO VE USED: \n", matrixes[location_min])

print( "DISTORTION VECTOR TO VE USED: \n", distortions[location_min])

# SAVE results
np.savez('CameraCalibration.npz', 
         mtx=matrixes[location_min],
         dist=distortions[location_min],
         rvecs=rotations[location_min],
         tvecs=translations[location_min],
         newcameramtx=new_matrixes[location_min],
         mean_error=minimum_error)
print("Files succesfully saved in CameraCalibration.npz")

