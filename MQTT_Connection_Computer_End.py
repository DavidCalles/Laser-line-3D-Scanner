# -*- coding: utf-8 -*-
"""
##############################################################################
#
#   MODULE DESCRIPTION:
#       This module has the required code to stablish an MQTT connection 
#       between the raspberry-pi and the main processing unit (a desktop 
#       computer or laptop) FROM THE COMPUTER END.
#       Information sent to the embedded system include the directory where
#       the image pairs should be saved (in raspberry local filesystem) as
#       well asthe amount of image pairs that must be acquired. 
#       The algorithm then creates a local directory in the computer where 
#       (if enabled) the original image pairs will be saved. A representation
#       of the segmented laser line can also by saved (if enabled)
#        
#   USAGE:
#        For using this module, "Algorithms.py" should be in the same folder
#        as this module and all required packages must be installed in 
#        advance.
#        The atributes of the MQTT are specified with variables as following 
#        by default:
#            
#            my_host = "broker.hivemq.com"   #Broker
#            my_port = 1883                  #Port
#            my_topic = "Scanner3D_David"    #Topic
#            my_qos = 2                      #Quality of service (0,1,2)
#        
#        However, can (and should) be changed to the ones desired both in the 
#        COMPUTER END and EMBEDDED SYSTEM END codes.
#        
#       By default, retrived images are also showed and saved locally in 
#        directory "scan_name" wich is a folder with name depending of date 
#        and time of code being run. With subfolder "Originals" and "Lasered".
#       
#       To disable this, variables must be changed to "False" respectively. 
#       Please note that saving both original and lasered images can 
#       consume up to 2GB of memory per scan depending on the total amount 
#        of image pairs acquired.
#        
#            SHOW_ORIGINAL_IMAGES = True
#            SHOW_LASER_IMAGES = True
#            WRITE_LOCAL_ORIGINALS = True
#            WRITE_LOCAL_LASERS = True
#            
#        Finally, the embedded system of the scanner will take a 
#        "NUM_IMG_PAIRS" amount of image pairs. Set in high quality to 760.
#        
#        If less images are desired, the mentiones variable can be changed.
#        
#        NOTES:  Please take into account that for this code to work properly
#                a shared folder must be prepared in advance between the 
#                embedded system and the computer.
#                Also, the EMBEDDED SYSTEM END  code must be run before the 
#                COMPUTER END code.
#    
#    ACTION:                             DATE:           NAME:
#        First implementation and tests      2/Nov/2020      David Calles
#        Code comments and last review       1/Dic/2020      David Calles                
#
#
##############################################################################
"""
#----------------------------------------------------------------------------#
# ----------------------- REQUIRED EXTERNAL PACKAGES ------------------------#
#----------------------------------------------------------------------------#
import cv2
import numpy as np
import time
import os
import paho.mqtt.client as MQTT
import matplotlib.pyplot as plt
from os.path import isfile, isdir
from datetime import datetime

#----------------------------------------------------------------------------#
# ----------------------- REQUIRED OWN FUNCTIONS-----------------------------#
#----------------------------------------------------------------------------#

from Algorithms import Double_Lagrange_Aprox_Image
from Algorithms import Laser_segmentation

#----------------------------------------------------------------------------#
# ----------------------- MQTT HANDSHAKE STRUCTURE --------------------------#
#----------------------------------------------------------------------------#
#  1. Both devices are subscribed to "my_topic" in "my_host" through "my_port"
#  2. Communication through publishes in "my_topic" with strings

#       PC  --> "START"
#       RPI --> "ACK1"
#       PC  --> "FOLDER_"+scan_name
#       RPI --> "ACK2_"+scan_name
#       PC  --> "INITIATE_"+image_pairs
#       RPI --> "ACK3_"+image_pairs
#       DISCONNECT() 

#----------------------------------------------------------------------------#
# -------------- CONNECTION AND FEATURE ENABLING VARIABLES ------------------#
#----------------------------------------------------------------------------#

# ENABLE GENERAL FUNCTIONS OF CODE
SHOW_ORIGINAL_IMAGES = True
SHOW_LASER_IMAGES = True
WRITE_LOCAL_ORIGINALS = True
WRITE_LOCAL_LASERS = True
NUM_IMG_PAIRS = 760
START = False

# CURRENT DATE
now = datetime.now()
scan_name = now.strftime("%b_%d_%Y_%H_%M_%S")

# INFORMATION FOR MQTT COMUNICATION
my_host = "broker.hivemq.com"
my_port = 1883
my_topic = "Scanner3D_David"
my_qos = 2
image_pairs = str(NUM_IMG_PAIRS)

# PAYLOADS TO SEND, RECEIVE AND ACKNOWLEDGE
payload1_PC = "START"
payload2_PC = "FOLDER_" + scan_name
payload3_PC = "INITIATE_" + image_pairs
payload1_RPI = "ACK1"
payload2_RPI = "ACK2_" + scan_name
payload3_RPI = "ACK3_" + image_pairs

# PATH IN RASPBERRY TO LOCALLY SAVE IMAGES
path = '//RASPBERRYPI/Raspi/Scans/'+scan_name

#----------------------------------------------------------------------------#
#-------------------------MQTT FUNCTION DEFINITIONS -------------------------#
#----------------------------------------------------------------------------#

# CALLBACK FUNCTION FOR MQTT DISCONNECTION FROM BROKER
def on_disconnect(client, userdata, rc):
    if rc != 0:
        print("Unexpected disconnection.")
    else:
        print("Client disconnected. ")

# CALLBACK FUNCTION FOR MQTT CONNECTION TO BROKER 
def on_connect(client, userdata, flags, rc):
    print('connected (%s)' % client._client_id)
    client.subscribe(topic=my_topic, qos=my_qos)
    client.publish(my_topic, payload=payload1_PC, qos=my_qos)
 
# CALLBACK FUNCTION FOR MQTT "MESSAGE RECEIVED" 
# EVALUATE received messages
def on_message(client, userdata, message):
    msg = message.payload.decode()
    print("MESSAGE: {}".format(msg))
    if(msg == payload1_RPI):
        client.publish(my_topic, payload=payload2_PC, qos=my_qos)
    if(msg == payload2_RPI):
        client.publish(my_topic, payload=payload3_PC, qos=my_qos)
    if(msg == payload3_RPI):
        print ("CONNECTION WITH RASPBERRY STABLISHED SUCCESFULLY!")
        global START
        START = True
        client.disconnect()      

# HANDSHAKE FUNCTION
def MQTT_Handshake_PC():
    client = MQTT.Client(client_id='David-Calles-PC')
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    client.connect(host=my_host, port=my_port)
    client.loop_forever()
 

#----------------------------------------------------------------------------#
#---------------------D2D COMMUNICATION INITIALIZATION-----------------------#
#----------------------------------------------------------------------------#  

# HADSHAKE IS INITIATED 
MQTT_Handshake_PC()

# ------------- GLOBAL USE VARIABLES
image_pairs = int(image_pairs)
wait_time = 2 # Time between showing image paris
first_attempt = True
attempts = 0
max_attempts = 50 #Attempts to find an image pair
i = 0

# ------------ ENSURING THERE IS ACCESS TO SHARED FOLDER
print("WAITING for access to shared folder", end='' )
while (isdir(path) is False):
    time.sleep(20)
    print(".", end='')
print("\nACCESS OBTAINED SUCCESFULLY to ", path)

# ------------ CREATE FOLDERS TO SAVE IMAGES LOCALLY
try:
    os.mkdir(scan_name)
    os.mkdir(scan_name+"/Originals")
    os.mkdir(scan_name+"/Laser")
except OSError:
    print("Error creating local folders!")
else:
    print("Local folders created succesfully")


WARM_TIME = 22
print ("WAITING FOR SCANNER TO START...")
time.sleep(WARM_TIME)

#----------------------------------------------------------------------------#
# ------------- ITERATION TRHOUGH ALL IMAGE PAIRS ---------------------------#
#----------------------------------------------------------------------------#

while (START and (i < image_pairs)):
    # GET image pair filenames
    image_path1 = path + "/Image_{}.png".format(i*2)
    image_path2 = path + "/Image_{}.png".format((i*2)+1)
    if (isfile(image_path1) and isfile(image_path2)):
        print("RETRIEVED Images {} and {}. ".format(i*2, (i*2)+1))
        # RETRIEVE images
        imgA = cv2.imread(image_path1)
        imgB = cv2.imread(image_path2)
        cv2.destroyAllWindows()
        # SHOW retrieved images if enabled
        if SHOW_ORIGINAL_IMAGES:
            imgpair = np.hstack((imgA,imgB))
            imgpair_r = cv2.resize(imgpair, dsize=None, fx=0.4, fy=0.4)
            cv2.imshow("Images {} and {}".format(i*2, (i*2)+1), imgpair_r)
            cv2.waitKey(1)
        # SAVE retrieved images locally if enabled
        if WRITE_LOCAL_ORIGINALS:
            cv2.imwrite("{}/Originals/Image_{}.png".format(scan_name,i*2),
                        imgA)
            cv2.imwrite("{}/Originals/Image_{}.png".format(scan_name,(i*2)+1),
                        imgB)
        # UNDISTORT images (DISABLED)
        img1 = imgA.copy() #Simple_Image_Correction(imgA)
        img2 = imgB.copy() #Simple_Image_Correction(imgB)
        # GET Segmented laser using manual thresholding
        _, _, _, threshed, indxs, indxs2, opened, closed, contrast_increase= \
            Laser_segmentation(img1, img2, thresh=0.14,
                               gauss=True, strict=False, plane=False)
        # GET max values as numpy array    
        ppx = np.array(indxs)
        ppx2 = np.array(indxs2)
        # REFINE center of laser with double parabola approximation
        subpxs = Double_Lagrange_Aprox_Image(opened, ppx, ppx2, True, False)
        int_indxs = np.uint16(np.around(subpxs))
        
        # GET segmented laser image if enabled (just as preview)
        if WRITE_LOCAL_LASERS or SHOW_LASER_IMAGES:     
            subpix_laser = np.zeros((img1.shape[0],img1.shape[1]))
            for j in range(len(int_indxs)):
                subpix_laser[int_indxs[j,0],int_indxs[j,1]] = 255                   
            coloured_img = np.zeros(img1.shape,dtype=np.uint8)
            coloured_img[:, :, 2] = opened+subpix_laser
            coloured_img[:, :, 1] = subpix_laser
            coloured_img[:, :, 0] = subpix_laser 
            # SAVE laser image files if enabled 
            if(WRITE_LOCAL_LASERS):
                cv2.imwrite("{}/Laser/LaserImage_{}.png".format(
                    scan_name,i*2), coloured_img)
            # SHOW laser image files if enabled
            if(SHOW_LASER_IMAGES):
                colour_r= cv2.resize(coloured_img, dsize=None, fx=0.6, fy=0.6)
                cv2.imshow("LaserImage_{}.png".format(i*2), colour_r)
                cv2.waitKey(1)
        # some control variables
        first_attempt = True
        attempts = 0
        i+=1
        time.sleep(wait_time)  
        cv2.destroyAllWindows()
        
        # RETRY retrieving images if they are not found
    else:
        if first_attempt:
            print("WAITING for images {} and {} ".format(i*2, (i*2)+1),
                  end = '')
            first_attempt = False
            attempts +=1
        else:
            print(".", end = '')
            attempts +=1
        time.sleep(wait_time)
        if(attempts > max_attempts):
            print("\n MAXIMUM NUMBER OF ATTEMPTS REACHED, TERMINATING SCAN!")
            break

print ("A TOTAL NUMBER OF {} IMAGE PAIRS WHERE PROCESSED.".format(
                                                            image_pairs))      
cv2.destroyAllWindows()
plt.close('all')
cv2.waitKey(1) # for compatibility with all operative systems


