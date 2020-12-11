
"""
##############################################################################
#
#   MODULE DESCRIPTION:
#        This module contains the algorithms to take image pairs with the 
#           raspberry pi board. Using an usb webcam, a laser-line module and
#           a motor driver connected to the GPIO of the board.
#
#   PRE-REQUISITES: 
#           Have the hardware wired properly.     
#    
#    ACTION:                             DATE:           NAME:
#        First implementation and tests      10/Aug/2020      David Calles
#        Code comments and last review       9/Dic/2020       David Calles           
#
##############################################################################
"""

#----------------------------------------------------------------------------#
#------------------------ REQUIRED EXTERNAL PACKAGES ------------------------#
#----------------------------------------------------------------------------#
import time
import RPi.GPIO as GPIO
import cv2

#----------------------------------------------------------------------------#
# ----------------------- MOTOR MOVEMENT SETTINGS ---------------------------#
#----------------------------------------------------------------------------#
dir_pin   = 11
step_pin  = 13
laser_pin = 3
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  

#----------------------------------------------------------------------------#
# ----------------------- FUNCTION DEFINITIONS ------------------------------#
#----------------------------------------------------------------------------# 

"""***************************************************************************   
# NAME: Clean_All
# DESCRIPTION:  Reset interfaces to original values
#               
# PARAMETERS:   None
#                              
# RETURNS:      None           
#               
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  19/Aug/2020      David Calles
#       Review and documentation        9/Dec/2020       David Calles
***************************************************************************"""
def Clean_All():
    GPIO.output(step_pin, 0)
    GPIO.output(dir_pin, 0)
    GPIO.output(laser_pin, 0)
    GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

"""***************************************************************************   
# NAME: Move_Motor
# DESCRIPTION:  Move motor a single time.
#               
# PARAMETERS:   steps: Amount of 1.8° steps to perform.
#               direction: Direction of movement. Clockwise-counterwise.
#               dir_pin: Pin to output of the direction signal
#               step_pin: Pin to output of the step signal.
#               one_step: Duration seconds of a single step.
#                              
# RETURNS:      None           
#               
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  19/Aug/2020      David Calles
#       Review and documentation        9/Dec/2020       David Calles
***************************************************************************""" 
def Move_Motor(steps, direction, dir_pin, step_pin, one_step):
    GPIO.output(dir_pin, direction) # 0 (clockwise)
    for i in range(steps):
        GPIO.output(step_pin, 1)
        time.sleep(one_step)
        GPIO.output(step_pin, 0)
        time.sleep(one_step)
    return 1

"""***************************************************************************   
# NAME: Take_Images_wMotor
# DESCRIPTION:  Take a certain amount of images. A motor movement is performed
#               between image pairs. Each image pair consists of a lasered
#               and a laser-less image.
#               
# PARAMETERS:   path: Directory where images should be saved.
#               num: Amount of image pairs to take.
#               save: Enable image save.
#                              
# RETURNS:      None.            
#               
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  19/Aug/2020      David Calles
#       Review and documentation        9/Dec/2020       David Calles
***************************************************************************"""
def Take_Images_wMotor(path="/home/pi/Documents/SharedImages/Scans/Scan_0/",
                       num=10, save=True):
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    SAVE = save
    save_path = path

    one_step = 0.004 #>5ms feels smooth (speed)
    steps_num = 10 # 200 = 1 turn
    SLEEPTIME = 1 #(s) time from turning laser ON to taking image
    CLOCKWISE = 0
    COUNTERWISE = 1
    MOVING_MOTOR = False

    # RASPI GPIO outputs setup
    GPIO.setup(dir_pin, GPIO.OUT)
    GPIO.setup(step_pin, GPIO.OUT)
    GPIO.setup (laser_pin, GPIO.OUT)

    iteration = 0
    trash_iter = 0
    max_iter = num
    try:
        while(iteration < max_iter):
            print("Iteration ", iteration)
            # LASER OUTPUT        
            GPIO.output(laser_pin, 1)
            time.sleep(SLEEPTIME)
            ret1, frame1 = cap.read()

            GPIO.output(laser_pin, 0)
            time.sleep(SLEEPTIME)
            ret2, frame2 = cap.read()
            # First images are discarded and motor is not moved.
            if trash_iter > 8:
                MOVING_MOTOR = True
            # COUNT images to be taken.   
            if MOVING_MOTOR:
                if iteration < max_iter:
                    if SAVE:
                        path1 = save_path + "/Image_{}.png".format(
                            iteration*2)
                        path2 = save_path + "/Image_{}.png".format(
                            (iteration*2)+1)
                        cv2.imwrite(path1, frame1)
                        cv2.imwrite(path2, frame2)
                    Move_Motor(steps_num, CLOCKWISE, dir_pin,
                               step_pin, one_step)
                    time.sleep(1)
                iteration += 1
            trash_iter +=1
        Clean_All()
    # EXIT securely by using Ctrl-C        
    except KeyboardInterrupt:
        print("System stoped manually!")
        Clean_All()
    except TypeError:
        print("Please connect usb camera!")
        Clean_All()