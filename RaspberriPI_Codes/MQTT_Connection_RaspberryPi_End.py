#! /usr/bin/python3

"""
##############################################################################
#
#   MODULE DESCRIPTION:
#       This module has the required code to stablish an MQTT connection 
#       between the raspberry-pi and the main processing unit (a desktop 
#       computer or laptop) FROM THE RASPBERRY PI END.
#       An MQTT message exchange is performed and the amount of images to be
#       taken as well as the directory to save them. 
#       Between image pairs, the motor is moved and the laser is turned ON and
#       OFF.
#        
#   USAGE:
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
#       By default, retrieved images are saved locally in the defined folder.
#        
#        NOTES:  Please take into account that for this code to work properly
#                a shared folder must be prepared in advance between the 
#                embedded system and the computer.
#                Also, the EMBEDDED SYSTEM END  code must be run before the 
#                COMPUTER END code.
#    
#    ACTION:                             DATE:           NAME:
#        First implementation and tests      2/Nov/2020      David Calles
#        Code comments and last review       9/Dic/2020      David Calles                
#
##############################################################################
"""

#----------------------------------------------------------------------------#
#------------------------ REQUIRED EXTERNAL PACKAGES ------------------------#
#----------------------------------------------------------------------------#
import paho.mqtt.client as MQTT
import os

#----------------------------------------------------------------------------#
# ----------------------- REQUIRED OWN FUNCTIONS-----------------------------#
#----------------------------------------------------------------------------#
from MotorLaserCamaraFUNC import Take_Images_wMotor

#----------------------------------------------------------------------------#
# ----------------------- MQTT HANDSHAKE STRUCTURE --------------------------#
#----------------------------------------------------------------------------#
# --------------- INITIATE HAND-SHAKE (PC-RPI) WITH MQTT ---------------------
# PC  --> "START"
# RPI --> "ACK1"
# PC  --> "FOLDER_"+scan_name
# RPI --> "ACK2_"+scan_name
# PC  --> "INITIATE_"+image_pairs
# RPI --> "ACK3_" +image_pairs
# DISCONNECT()

#----------------------------------------------------------------------------#
# -------------- CONNECTION AND FEATURE ENABLING VARIABLES ------------------#
#----------------------------------------------------------------------------#
my_host = "broker.hivemq.com"
my_port = 1883
my_topic = "Scanner3D_David"
my_qos = 2
scan_name = '0'
image_pairs = '0'

payload1_PC = "START"
payload2_PCh = "FOLDER_"
payload3_PCh = "INITIATE_"

payload1_RPI = "ACK1"
payload2_RPIh = "ACK2_"
payload3_RPIh = "ACK3_" 

#----------------------------------------------------------------------------#
#-------------------------MQTT FUNCTION DEFINITIONS -------------------------#
#----------------------------------------------------------------------------#
# CALLBACK FUNCTION FOR MQTT DISCONNECTION TO BROKER 
def on_disconnect(client, userdata, rc):
    if rc != 0:
        print("Unexpected disconnection.")
    else:
        print("Client disconnected. ")
        
# CALLBACK FUNCTION FOR MQTT CONNECTION TO BROKER 
def on_connect(client, userdata, flags, rc):
    print('connected (%s)' % client._client_id)
    client.subscribe(topic=my_topic, qos=my_qos)
    client.publish(my_topic, payload="RPI Suscribed", qos=my_qos)

# CALLBACK FUNCTION FOR MQTT  RECEIVED MESSAGE 
    # EVALUATE received messages
def on_message(client, userdata, message):
    msg = message.payload.decode()
    print("MESSAGE: ", msg)
    if(msg == payload1_PC):
        client.publish(my_topic, payload=payload1_RPI, qos=my_qos)
    if(msg[0:7] == payload2_PCh[0:7]):
        global scan_name
        scan_name = msg[7:]
        client.publish(my_topic, payload=(payload2_RPIh+scan_name),
                       qos=my_qos)
    if(msg[0:9] == payload3_PCh[0:9]):
        global image_pairs
        image_pairs = msg[9:]
        client.publish(my_topic, payload=(payload3_RPIh+image_pairs),
                       qos=my_qos)
        print ("CONNECTION WITH PC STABLISHED SUCCESFULLY!")
        client.disconnect()      

# HANDSHAKE FUNCTION
def MQTT_Handshake_RPI():
    client = MQTT.Client(client_id='My_rpi_DavidC')#my_clientid
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    client.connect(host=my_host, port=my_port)
    client.loop_forever()
try:
    # INITIATE handshake
    MQTT_Handshake_RPI()
    print ("Folder name: ", scan_name)
    print ("Image pairs amount: ", image_pairs)
    image_pairs = int(image_pairs)
    #------------ CREATE DIRECTORY FOR SCAN IMAGES -------
    path = "/home/pi/Documents/SharedImages/Scans/"+scan_name
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)
        # TAKE images moving motor.
        Take_Images_wMotor(path, image_pairs, save=True)

except:
    print("ERROR: PLEASE CHECK CONNECTION AND REBOOT DEVICE!")