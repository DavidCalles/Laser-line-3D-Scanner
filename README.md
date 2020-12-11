# Laser-Line 3D Scanner
By David Calles.

# Description
This repository contains all the code, models, image sets, and diagrams for an implementation of a 3D scanner at a prototype level. Consisting of a code for the computer-end and a code for the raspberrypi-end. 
Each step must be performed individually by running each code with its corresponding pre-requisites and using the ourputs of the block before.

# Requirements
The package requirements to run the algorithms are listed in the requirements.txt file. And can be install using:

    pip3 install -r requirements.txt
 or, if one has Anaconda installed:
 
    conda install --file requirements.txt

# Software blocks diagram
The following image illustrates the main stages of the software in the repository. 
![Blocks_Diagram](https://github.com/DavidCalles/Laser-line-3D-Scanner/blob/main/Description_Images/Sofware_Overview%20.png)

To get more information on each block as well as the fundamentals and theory of each algorithm, you are encourage to read the book located in the "Official_Documents" folder. In this document, the results using this scanner as well as a comparison with a time-of-flight sensor, and a plenoptic camera results is exposed.

# Files and folders
## Image sets
All the image sets needed to perform the different calibrations are included:

 - Camera calibration image set (100 images)
 - Scanner calibration image set (212 images = 106 image pairs)
 - Center calibration image set (4 images)

The same way, two different image sets for reconstructing its point clouds are included:

 - Volleyball ball (1500 images = 750 image pairs) 
 - Chili plant (1500 images = 750 image pairs)

## Sample point clouds

Two ascci XYZ files of the reconstructed point clouds are also shared in the "Sample point clouds" folder.

## 3D-print models
All the models for printing the car, the platform segments, the motor adapters, and the of the scanner are shared as STL files. More information can be found in the final book pdf file. 

## PCB files
All the gerber and drill files of the PCB for circuits for the motor driver and adapter to the RaspberryPi 4B are shared. More information can be found in the final book pdf file.

## Raspberry Pi-end codes
The codes that must run on the raspberry pi are shared in the "RaspberryPi_Codes" folder, including:

