#! /usr/bin/python3
"""
##############################################################################
#
#   MODULE DESCRIPTION:
#       This module can be used for "manually" acquiring calibration images
#       needed for the "Camera Calibration", "Scanner Calibration", and 
#       "Center Calibration". Image pairs are taken (lasered+laserless) 
#       without moving the motor.
#       
#
#   PRE-REQUISITES: 
#        Laser and camera must be wired properly.
#
#   OUTPUTS:
#        Image set: Image pairs are saved as Image_{i} (lasered) and 
#                   Image_{i+1} in the "save_path" directory.
#        
#   USAGE:
#         1. If pictures of a chessboard will be taken. The user can enable 
#           the borders checking by setting the "BORDER_CHECK=True". The 
#           chessborad size must be set in "csize".
#         2.To take an image press "t" key. Acquiring an image pair takes a 
#           couple of seconds as it discards by default the first 4 taken 
#           image pairs.
#         3. If the image is as wished, press "y" to save it, or press any 
#           other key to discard it. Images are saved to "save_path" folder.
#           "SAVE" must be True.
#       
#    
#    ACTION:                             DATE:           NAME:
#        First implementation and tests      17/Sep/2020      David Calles
#        Code comments and last review       10/Dic/2020       David Calles           
#
##############################################################################
"""
#----------------------------------------------------------------------------#
#------------------------ REQUIRED EXTERNAL PACKAGES ------------------------#
#----------------------------------------------------------------------------#
import numpy as np
import cv2
import RPi.GPIO as gpio
from time import sleep

#----------------------------------------------------------------------------#
#---------------------------------- SETTINGS --------------------------------#
#----------------------------------------------------------------------------#
csize = [7,9]   # Chessboard size
laser_pin = 3   # GPIO2 (3)

gpio.setwarnings(False)
gpio.setmode(gpio.BOARD)# INITIATE pin numeration as: "Board numeration"
gpio.setup (laser_pin, gpio.OUT) # SET laser pin to output

BORDER_CHECK = True # Enable chessboard checking
retC = "Disabled"
# SET termination criterio to find chessboard corners.
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

SLEEPTIME = 1
# ENABLE saving images
SAVE = True
save_path = "Calibration_Set/"

# SET camera source and image size.
cap = cv2.VideoCapture(0)# 0 es la primera camara listada en su sistema
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

"""***************************************************************************   
# NAME: TAKE_IMG_PAIR
# DESCRIPTION:  Take an image pair. If "y" key is pressed, the image is then
#               saved to save_path directory. When the function is called
#               the first 4 image pairs are discarded to avoid trash/noisy
#               images and just the 5th one is retrieved and showed.
#               
# PARAMETERS:   i: Index to save images. Lasered will be Image_{i*2}.png and 
#                  Laserless will be Image_{i*2+1}.png.
#                              
# RETURNS:      None           
#               
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  19/Aug/2020      David Calles
#       Review and documentation        9/Dec/2020       David Calles
***************************************************************************"""
def TAKE_IMG_PAIR(i):
    # TAKE frames
    for x in range(5):
        # TURN laser ON
        gpio.output(laser_pin, True)
        sleep(SLEEPTIME)
        # GET frame
        ret1, frame1 = cap.read()
        sleep(SLEEPTIME)
        # TURN laser OFF
        gpio.output(laser_pin, False)
        sleep(SLEEPTIME)
        # GET frame
        ret2, frame2 = cap.read()
        # DISCARD first four takes and take last
        if (x == 4):
            frame1_5 = frame1.copy()
            frame2_5 = frame2.copy()
    # GET gray image
    gray = cv2.cvtColor(frame2_5, cv2.COLOR_BGR2GRAY)
    # FIND the chess board corners if enabled
    if BORDER_CHECK:
        retC, corners = cv2.findChessboardCorners(gray,
                                                  (csize[0],csize[1]), None)
    # SHOW acquired frames    
    show = np.hstack((frame1_5, frame2_5))
    show_img = cv2.resize(show, None, fx=0.18, fy=0.18)
    cv2.imshow('frames',show_img)
    print("Iteration {}, Frames:{} & {}, Corners: {}".format(i, ret1,
                                                             ret2, retC))
    ok = False
    # SAVE images if "y" key is presed
    if cv2.waitKey(0) & 0xFF == ord('y'):
        if(SAVE):
            filename1 = save_path+"Image_{}.png".format(i*2)
            cv2.imwrite(filename1, frame1_5)
            filename2 = save_path+"Image_{}.png".format(i*2+1)
            cv2.imwrite(filename2, frame2_5)
            ok = True
    cv2.destroyAllWindows()
    # CHECK for corners in image if enabled  
    if not retC and BORDER_CHECK:
        print("CAREFULL BORDERS WHERE NOT FOUND!")
        print("RE-TRY PLEASE!")
    return ok


"""***************************************************************************
#                                                                            #
#                                                                            #
#                            MAIN FUNCTION                                   #
#                                                                            #
#                                                                            #
***************************************************************************"""
i=0
try: 
    while(True):
        # SHOW real-time frames
        ret, frame = cap.read()#
        showA = cv2.resize(frame, None, fx=0.3, fy=0.3)
        cv2.imshow("Frame", showA)
        # TAKE image pair if "t" key is pressed
        if cv2.waitKey(1) & 0xFF == ord('t'):
            ok = TAKE_IMG_PAIR(i)
            if ok:
                i+=1       
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

# EXIT safely      
except KeyboardInterrupt:
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


