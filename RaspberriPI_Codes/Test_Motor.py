#! /usr/bin/python3
"""
##############################################################################
#
#   MODULE DESCRIPTION:
#        This module is used for testing the motor connection by performing
#       a movement with it. Its purpose is to check everything is correctly 
#       wired and to tune the values of step duration "one_step" and the 
#       amount of steps per movement "steps_num" to be used in scanning.
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

#----------------------------------------------------------------------------#
#------------------------- OUTPUT PINS SETTING ------------------------------#
#----------------------------------------------------------------------------#
dir_pin   = 11
step_pin  = 13
laser_pin = 3 

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.setup(dir_pin, GPIO.OUT)
GPIO.setup(step_pin, GPIO.OUT)

#----------------------------------------------------------------------------#
#------------------------- MOTOR MOVEMENT SETTINGS --------------------------#
#----------------------------------------------------------------------------#
one_step = 0.004#>5ms feels smooth (speed)
steps_num = 10# 200 = 1 turn = 360°
amount_movements = 30 # Movements to perform in test
CLOCKWISE = 0
COUNTERWISE = 1

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
# RESET GPIO to original value
def Clean_All():
    GPIO.output(step_pin, 0)
    GPIO.output(dir_pin, 0)
    GPIO.cleanup()
    
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


try:
    for i in range(amount_movements):
        
        Move_Motor(steps_num, CLOCKWISE, dir_pin, step_pin, one_step)
        time.sleep(1) # STALL for 1 sec
except KeyboardInterrupt:
        print("System stoped manually!")
        Clean_All()