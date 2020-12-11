# -*- coding: utf-8 -*-
"""
##############################################################################
#
#   MODULE DESCRIPTION:
#        This module performs the object reconstruction from image pairs. It
#        allows saving and visualizing the resulting coloured point cloud
#        with a small amount of minimum filtering.
#
#   PRE-REQUISITES: 
#        Camera-calibration: 3x3 intrinsic mtx and distoriton coefficients.
#        Scanner calibration: 4x3 transformation matrix.
#        Center calibration: 1x3 or 3x1 center coordinates.
#        Image acquisition: N Image pairs (lasered-laserless).
#
#   OUTPUTS:
#        Point cloud: '.XYZ' file with xyz and rgb information coded in ascii.
#                     Visualization of resulting point cloud with minimum
#                     filtering.
#        
#   USAGE:
#         1. All required packages should be installed in advance.
#         2. For using this module, "AlgorithmsV8.py" file must be in the same
#            directory of this file.
#         3. "FEATURE-ENABLING VARIABLES" should be set to the desired values
#            (Check each variable description). By default just the final 
#            point cloud is showed.
#         4. "path" variable should have the image pairs' folder.
#         5. The "CalibrationMtxV1.npz" file with the scanner calibration
#            4x3 matrix should be in the same directory as this file.
#         6. The "center3D" variable should be changed to the calibrated
#            center coordinates.
#         7. The rotation angle between images is automatically calculated
#            using the amount of image pairs found. This supposes that the
#            image set describes exactly 360°. If more/less degrees should be
#            taken into account, the variable "MISSING" should be changes. 
#            Its value shoould be the missing images to complete 360°. If more
#            images were taken, then the value must be negative.
#    
#    ACTION:                             DATE:           NAME:
#        First implementation and tests      23/Nov/2020      David Calles
#        Code comments and last review       6/Dic/2020       David Calles                
#
##############################################################################
"""
#----------------------------------------------------------------------------#
# ----------------------- REQUIRED EXTERNAL PACKAGES ------------------------#
#----------------------------------------------------------------------------#
import cv2
import numpy as np
import glob
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
from datetime import datetime

#----------------------------------------------------------------------------#
# ----------------------- REQUIRED OWN FUNCTIONS-----------------------------#
#----------------------------------------------------------------------------#
from Algorithms import Simple_Image_Correction
from Algorithms import Double_Lagrange_Aprox_Image
from Algorithms import Laser_segmentation

# GUARANTEE no figures/images are opened in advance
cv2.destroyAllWindows
plt.close('all')

#----------------------------------------------------------------------------#
# ----------------------- FEATURE-ENABLING VARIABLES-------------------------#
#----------------------------------------------------------------------------#

# LASER SEGMENTATION VARIABLES
NORMAL_THRESHOLDING=0
OTSU_THRESHOLDING=1

# GENERAL PURPOSE VARIABLES FOR PLOTING CONTROL
IMG_WAITKEY = 100
SHOW_IMAGES=False # Show images
PLOT=False   #Show Graphs
SAVE_IMAGES=False # Save images to files
JUST_FIRST=False # Just iterate thru first pair of images (test)
SUBPIX_GRAPH = False # Graph Subpix aproximation (1 per iamge)
ROW_TO_GRAPH = 512# Row to graph
PLOT_3D_SINGLE = False #3D points from each image
PLOT_3D_FILTER_SINGLE = False #Filtered 3D points from each image
PLOT_3D_CUMULATIVE = False #Cumuative 3D points from until current image
VERBOSE = False #Show additional information
PLOT_3D_FILTER_ALL = True #Plot filtered complete point cloud
PLOT_3D_FINAL = True #Plot final resulting point cloud

# SET Image pairs directory
# High resolution colored ball better 
#path = "C:/Users/yodav/Documents/NOVENO_SEMESTRE/TRABAJOGRADO/" + \
#    "MotorLaserCamera/Nov_25_2020_11_25_59/Originals/" #MISSING = 0

# High resolution colored plant better 
path = "C:/Users/yodav/Documents/NOVENO_SEMESTRE/TRABAJOGRADO/" + \
    "MotorLaserCamera/Nov_24_2020_01_05_26/Originals/" #MISSING = -150

# GET image filenames to be used (verbose purpose)   
images = glob.glob(path+'*.png')
cv2.startWindowThread()  
#print("Using image pairs: \n", images)
print("Found image pairs: ", len(images)/2)

# IMAGE SIZE DEFINITIONS
img_size = cv2.imread(images[0])
rwidth=img_size.shape[1] #e.g 1920
rheight=img_size.shape[0] # e.g 1080
rsize=(rwidth, rheight)
my_flag = (cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)

# LOAD Resulting transformation matrix from scanner calibration
with np.load('ScannerCalibration.npz') as file:
    LaserMtx_4x3, _, _, _, _ = [file[i] for i in (
        'Mtx_4x3','points2D','points3D','error', 'used_imgs')]

# LOAD Calibrated center point from center calibration
center3D = np.array([48.52758569, 135.39777659, 578.44508265])

# ESTIMATE angle of rotation between image pairs across vertical axis
MISSING = -150
ANGLE = 360/((len(images)+MISSING)/2)

#----------------------------------------------------------------------------#
#---------------POINT CLOUD UTILITY FUNCTION DEFINITIONS --------------------#
#----------------------------------------------------------------------------#

"""***************************************************************************   
# NAME: display_inlier_outlier
# DESCRIPTION: Display a color-coded representation of inliers and outliers
#              product of a given filtering stage.
#              Based on official open3d documentation. 
#
# PARAMETERS: cloud: Input open3d object of point cloud with xyz and rgb-
#             ind:   Indexes of the inliers obtained from filtering stage. 
#                
# RETURNS:    pcd_in: Output open3d object of point cloud inliers with 
#                        xyz and rgb.
#             pcd_out: Output open3d object of point cloud outliers with 
#                        xyz and rgb.
#               
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  21/Nov/2020      David Calles
#       Review and documentation        5/Dec/2020       David Calles
***************************************************************************"""
def display_inlier_outlier(cloud, ind):
    # SPLIT inliers from outliers
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    
    # CREATE Inliers and outliers point clouds 
    pcd_in = o3d.geometry.PointCloud()
    pcd_in.points = inlier_cloud.points
    pcd_in.colors = inlier_cloud.colors
    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = outlier_cloud.points
    pcd_out.colors = outlier_cloud.colors
    
    # VISUALIZE filter results
    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    
    return pcd_in, pcd_out

"""***************************************************************************   
# NAME: ColorEnhancer_LAB
# DESCRIPTION: Apply an adaptive histogram equalization in lab color space
#               
# PARAMETERS: img: Input 3-channel BGR image  
#             cliplimit: clip limit for adaptive histogram equalization, 
#                         default:9.0
#             tilegridsize: tuple of size of blocks for adaptive histogram 
#                         equalization, default: (4,4)
#                              
# RETURNS:    final: Output 3-channel BGR image with color enhanced 
#            
#               
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  21/Nov/2020      David Calles
#       Review and documentation        1/Dec/2020       David Calles
***************************************************************************"""
def ColorEnhancer_LAB(img, cliplimit=9.0, tilegridsize=(4,4)): 
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=tilegridsize)
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)  
    return final

"""***************************************************************************   
# NAME: Write_xyz_file
# DESCRIPTION:  Write .xyz ascii file with xyz and rgb information of a point 
#               cloud.
#               Based on official open3d documentation.
#               
# PARAMETERS:   data: xyz information as a 3-column matrix
#               rgb: color rgb information as a 3-column matrix
#               filename: string with filename, must include the ".xyz"
#                         extension.
#                              
# RETURNS:      ret: = 1 if file written correctly              
#               
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  21/Nov/2020      David Calles
#       Review and documentation        1/Dec/2020       David Calles
***************************************************************************"""
def Write_xyz_file (data, rgb, filename):
    
    fh = open(filename, "w+")
    fh.write("X Y Z R G B\n") # header
    
    for i in range(len(data)):
        fh.write("{} {} {} {} {} {}\n".format(
            str(data[i,0]), str(data[i,1]), str(data[i,2]),
            str(rgb[i,0]), str(rgb[i,1]), str(rgb[i,2])))    
    fh.close()
    
    return 1

"""***************************************************************************   
# NAME:  Filter_3D_Data_Cilinder
# DESCRIPTION: Filter data out of a defined cylinder. 
#              Cylinder is defined with height in Y azis and being X,Z
#              the axis where the circumference is defined. The center of the
#              cylinder is taken as [0,0,0]
#               
# PARAMETERS:   points3D: XYZ points of data to be filtered (3-column mtx)
#               color: RGB components of data to be filtered  (3-column mtx)
#               radius: Radius in milimeters of the cylinder. Default=700.
#               height: Lower and upper height values of the cylinder. 
#                       Default=[-100, 700].
#               verbose: Print usefull information. Default=True.
#               apply: Apply filter. Default=True.
#               plot: Plot points and cylinder using pyplot. Default=True. 
#                              
# RETURNS:      points3D_clean: filtered xyz data. (3-column mtx)
#               color_clean: Corresponding rgb data. (3-column mtx)            
#               
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  23/Nov/2020      David Calles
#       Review and documentation        6/Dec/2020       David Calles
***************************************************************************"""
def Filter_3D_Data_Cilinder(points3D, color, radius=700, height=[-100, 700],
                            verbose=True, apply=True,
                            plot=True):
    
    if apply:               
        points3D_clean = []
        color_clean = []     
        # SET Cylinder dimensions
        r_limit = radius**2 
        h_limit = np.array(height)    
        r_points = ((points3D[:,0]**2) + (points3D[:,2]**2)) 
        outliers = 0
        
        # APPLY filter
        for n in range(len(points3D)):
            if(r_points[n] <= r_limit):
                if(points3D[n,1] <= h_limit[1]):
                    if(points3D[n,1] >= h_limit[0]):
                        points3D_clean.append(points3D[n,:])
                        color_clean.append(color[n,:])
                    else:
                        outliers +=1
                else:
                    outliers +=1
            else:
                outliers +=1
                
        points3D_clean = np.array(points3D_clean)  
        color_clean = np.array(color_clean)  
        
        if(verbose):
            print(" With a radius of {}mm and height of {},{}.\n \
                  {} Outliers were found"
                  .format(radius, h_limit[0], h_limit[1], outliers))
        
        if(plot):
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            # Scatter graph
            X = points3D[:,0]
            Y = points3D[:,1]
            Z = points3D[:,2]
            ax.scatter(X, Y, Z, c=color/255)
            
            # Cylinder
            x=np.linspace(-radius, radius, 500)
            y=np.linspace(h_limit[0], h_limit[1], 500)
            Xc, Yc=np.meshgrid(x, y)
            Zc = np.sqrt(r_limit-Xc**2)
            
            # Draw parameters
            rstride = 20
            cstride = 10
            ax.plot_surface(Xc, Yc, Zc, alpha=0.2, rstride=rstride,
                            cstride=cstride)
            ax.plot_surface(Xc, -Yc, Zc, alpha=0.2, rstride=rstride,
                            cstride=cstride)
            
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            plt.show()
    else:
        points3D_clean = points3D.copy()
        color_clean = color.copy()
        
    return points3D_clean, color_clean

"""***************************************************************************   
# NAME:     Filter 3D Data
# DESCRIPTION:  Filter data far from most points (statistical sphere)
#               The average and standart deviation is calculated from the 
#               input points. All points outside "factor" times the standart 
#               deviation are eliminated
#                         
# PARAMETERS:   points3D: xyz data of pointcloud. (3-column mtx)
#               color: rgb data of pointcloud. (3-column mtx)
#               factor: Amount of standart deviations to accept. Default=3.
#               verbose: Print aditional information. Default=True.
#               apply: Enable filter. Default=True.
#               plot: Plot sphere and points using matplotlib. Default=True.
#                              
# RETURNS:      points3D_clean: filtered xyz data. (3-column mtx)
#               color_clean: Corresponding rgb data. (3-column mtx)              
#               
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  19/Oct/2020     David Calles
#       Review and documentation        6/Dec/2020      David Calles
***************************************************************************"""
def Filter_3D_Data(points3D, color, factor=3, verbose=True, apply=True,
                   plot=True):
    
    if apply:    
        avr_x = np.average(points3D[:,0]) #average in x coordinates
        std_x = np.std(points3D[:,0]) #standart deviation in x coordinates
        
        avr_y = np.average(points3D[:,1]) #average in y coordinates
        std_y = np.std(points3D[:,1]) #standart deviation in y coordinates
        
        avr_z = np.average(points3D[:,2]) #average in z coordinates
        std_z = np.std(points3D[:,2]) #standart deviation in z coordinates
        
        points3D_clean = []
        color_clean = []
        # SET limits
        xlimit = (factor*std_x) + avr_x
        ylimit = (factor*std_y) + avr_y
        zlimit = (factor*std_z) + avr_z
        
        # APPLY filter
        outliers = 0
        for n in range(len(points3D)):
            if(abs(points3D[n,0]) <= xlimit):
                if(abs(points3D[n,1]) <= ylimit):
                    if(abs(points3D[n,2]) <= zlimit):
                        points3D_clean.append(points3D[n,:])
                        color_clean.append(color[n,:])
                    else:
                        outliers +=1
                else:
                    outliers +=1
            else:
                outliers +=1
                        
        points3D_clean = np.array(points3D_clean)  
        color_clean = np.array(color_clean)  
        
        # PRINT additional information if enabled
        if(verbose):
            print("\n Data with X Average: {} and Stand. Dev: {}"
                  .format(avr_x, std_x))
            print("Data with Y Average: {} and Stand. Dev: {}"
                  .format(avr_y, std_y))
            print("Data with Z  Average: {} and Stand. Dev: {}"
                  .format(avr_z, std_z))
            print(" With a factor of {} Std. Devs. {} Outliers were found"
                  .format(factor, outliers))
        
        # PLOT graph if enabled
        if(plot):
            plt.figure()
            ax = plt.axes(projection='3d') 
            # --------------------3D POINTS and CONNECTING LINE
            zdata = points3D[:, 2]
            xdata = points3D[:, 0]
            ydata = points3D[:, 1]
            ax.plot3D(xdata, ydata, zdata, 'red')  
            ax.scatter3D(xdata, ydata, zdata, c=(color/255))             
            # --------------------Bounding sphere
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = (xlimit * np.outer(np.cos(u), np.sin(v))) + avr_x
            y = (ylimit * np.outer(np.sin(u), np.sin(v))) + avr_y
            z = (zlimit * np.outer(np.ones(np.size(u)), np.cos(v))) + avr_z
            ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b',
                            linewidth=2, alpha=0.2)      
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Z axis')
            plt.ion()
            plt.show()
    else:
        points3D_clean = points3D.copy()
        color_clean = color.copy()
    
    return points3D_clean, color_clean

"""***************************************************************************   
# NAME:     Rotate Points Manually
# DESCRIPTION: Rotate across certain axis a certain amount of degrees
#               
# PARAMETERS:  points3D: xyz point cloud data. (3-column mtx)
#              degrees: Degrees to rotate in (degrees, not radians).
#              axis: Axes to perform rotation on. Binary values. [x, y, z] 
#                                            
# RETURNS:     rotated_vec: Rotated data              
#               
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  19/Oct/2020     David Calles
#       Review and documentation        6/Dec/2020      David Calles
***************************************************************************"""
def Rotate_Points_Manually(points3D, degrees=2, axis=[0,1,0]):
    
    axis = np.array(axis)
    # ------------------- SINGLE ROTATION -------------------------------#
    # ARRAY TO BE ROTATED
    P1 = points3D.copy()
    # ANGLE TO BE ROTATED
    rotation_degrees = degrees
    # AXIS OF ROTATION (X, Y, Z)
    rotation_axis = axis
    # DEGREES to RADIANS 
    rotation_radians = np.radians(rotation_degrees)
    # ROTATION VECTOR
    rotation_vector = rotation_radians * rotation_axis
    # SCIPY ROTATION TRANSFORM
    rotation = R.from_rotvec(rotation_vector)
    # ROTATED VECTOR
    rotated_vec = rotation.apply(P1)
    
    return rotated_vec

"""***************************************************************************   
# NAME: Calculate 3D points
# DESCRIPTION:  Appy 4x3 Calculated transformation matrix. This is the actual
#               process of estimating 3D coordinates from image coordinates.
#               
# PARAMETERS:   mtx4x3: 4x3 transformation matrix.
#               points2D: points to be calculated. (row,column)(2-column mtx)
#                              
# RETURNS:      points3D2 (estimated 3D points)               
#               
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  20/Nov/2020     David Calles
#       Review and documentation        6/Dec/2020      David Calles
***************************************************************************"""
def Calculate_3D_Points(mtx4x3, points2D):    
    data = np.column_stack((points2D[:,1], points2D[:,0]))
    data_ext = np.hstack((data, np.ones((len(data),1))))  
    points3D = np.matmul(mtx4x3, data_ext.T)       
    points3D2 = (points3D/points3D[3,:]).T[:,0:3]   
    return points3D2
  

#----------------------------------------------------------------------------#
# -------------------- CUMULATIVE POINT CLOUD VARIABLES ---------------------#
#----------------------------------------------------------------------------#
 
point_cloud_points_local = []
point_cloud_points_global = []
point_cloud_color_local = []
point_cloud_color_global = []

# ------------- DEFINE if all pairs or just first pair will be used
iterations = range(0,len(images),2)
if (JUST_FIRST):
    iterations = range(0, 2, 2)

#----------------------------------------------------------------------------#
# ------------- ITERATION TRHOUGH ALL IMAGE PAIRS ---------------------------#
#----------------------------------------------------------------------------#
    
for i in iterations:
    print("Processing Image ", i)
    # READ image pair
    image_path1 = path + "Image_{}.png".format(i)
    image_path2 = path + "Image_{}.png".format(i+1)
    imgA    = cv2.imread(image_path1)
    imgB    = cv2.imread(image_path2)
    # UNDISTORT image pair
    img1 = Simple_Image_Correction(imgA)
    img2 = Simple_Image_Correction(imgB)
    # RESIZE image pair (if enabled)
    img1    = cv2.resize(img1,rsize) 
    img2    = cv2.resize(img2,rsize)    
    # SEGMENT laser line    and 
    # GET maximums (from left and right ) per row in image
    _, _, _, threshed, indxs, indxs2, opened, closed, contrast_increase= \
        Laser_segmentation(img1, img2, thresh=0.12, plane=False)    
    # TURN list of max values into a numpy array   
    ppx = np.array(indxs)
    ppx2 = np.array(indxs2)
    # REFINE center of laser line with double parabola approximation
    subpxs = Double_Lagrange_Aprox_Image(opened, ppx, ppx2, True,
                                         verbose=False)

    # CALCULATED difference between refined and non-refined laser points
    diffs2 = ppx - subpxs
    
    # ESTIMATE 3D coordinates from image coordinates using calibrated
    #             4x3 matrix transform (from scanner calibration)
    points3D = Calculate_3D_Points(LaserMtx_4x3, subpxs)  
    
    # ENHANCE colors from image in Lab color space
    img2_enhanced = ColorEnhancer_LAB(img2)
    
    # RETRIEVE color components of estimated points in rgb
    int_indxs = np.uint16(np.around(subpxs))
    bgr = img2_enhanced[int_indxs[:, 0], int_indxs[:, 1],:]
    rgb = bgr.copy()
    rgb[:, 0] = bgr[:, 2]
    rgb[:, 2] = bgr[:, 0]
    
    #-FILTER 3D points from single image pair statistically and/or using 
    #           bounding cylinder.    
    points3D_clean, rgb_clean =  Filter_3D_Data(points3D, rgb, factor=4,
                                                verbose=VERBOSE, apply=False,
                                                plot=PLOT_3D_FILTER_SINGLE) 
    """
    points3D_clean, rgb_clean = Filter_3D_Data_Cilinder(points3D, rgb,
                                                radius=700,
                                                height=[-700, 700],
                                                verbose=True, apply=True,
                                                plot=False)
    """
    # TRANSLATE 3D points from camera's coordinate frame to calibrated center   
    points3D_clean = points3D_clean - center3D
    
    # ROTATE 3D points across Y axis (vertical)
    
    rotated_3D = points3D_clean.copy()
    if i!=0:
        for r in range(int(i/2)):
            rotated_3D = Rotate_Points_Manually(rotated_3D, degrees=ANGLE,
                                                axis=[0,1,0]) # Y axis
    
    # ADD 3D points from single image pair to cumulative point cloud
    point_cloud_points_local.append(rotated_3D)
    point_cloud_points_global.append(rotated_3D)
    point_cloud_color_local.append(rgb_clean)
    point_cloud_color_global.append(rgb_clean)
    
    # PLOT desired 2D data if enabled
    if(PLOT):
              
        # --------LASER PLOTS--------------
        plt.figure()
        axes = plt.gca()
        x1 = ppx[:, 1]
        y1 = np.flip(ppx[:, 0])
        plt.scatter(x1, y1, label = "Undistorted laser max values left",
                    c='b', s=np.pi*0.1)
        x2 = ppx2[:, 1]
        y2 = np.flip(ppx2[:, 0])
        plt.scatter(x2, y2, label = "Undistorted laser max values left",
                    c='g', s=np.pi*0.1)  
        x3 = subpxs[:, 1]
        y3 = np.flip(subpxs[:, 0])
        plt.scatter(x3, y3, label = "Undistorted pixels after subpix",
                    c='r', s=np.pi*0.5)       
        axes.set_xlim([1000, 1300])
        axes.set_ylim([200, rheight+50])
        plt.xlabel('x - axis')
        plt.ylabel('y - axis')
        plt.title(' Laser segmentation evolution ')
        plt.grid()
        plt.legend()
        plt.show()
        
    # PLOT desired 3D data of single image pair if enabled
    if(PLOT_3D_SINGLE):

        # PLOT filtered points
        plt.figure()
        ax2 = plt.axes(projection='3d') 
        zdataC = points3D_clean[:, 2]
        xdataC = points3D_clean[:, 0]
        ydataC = points3D_clean[:, 1]
        ax2.scatter3D(xdataC, ydataC, zdataC, c=(rgb_clean/255), s=8)
        ax2.scatter3D(0,0,0, c='k', s=40)
        ax2.scatter3D(center3D[0],center3D[1],center3D[2], c='r', s=40)
        ax2.set_xlabel('X axis')
        ax2.set_ylabel('Y axis')
        ax2.set_zlabel('Z axis')
        plt.ion()
        plt.show()
    
    # PLOT desired 3D data from all image pairs until this ones if enabled
    if(PLOT_3D_CUMULATIVE):
        #--------------------------Turn lists into arrays--------------------
        #points
        points_plot = point_cloud_points_local[0]
        if len(point_cloud_points_local) > 1:
            for f in range(1,len(point_cloud_points_local)):
                points_plot = np.vstack((points_plot,
                                         point_cloud_points_local[f]))
        #color
        rgb_plot = point_cloud_color_local[0]
        if len(point_cloud_color_local) > 1:
            for f in range(1,len(point_cloud_color_local)):
                rgb_plot = np.vstack((rgb_plot, point_cloud_color_local[f]))
                
        #filtered points
        plt.figure()
        ax = plt.axes(projection='3d') 
        xdataC = points_plot[:, 0]
        ydataC = points_plot[:, 1]
        zdataC = points_plot[:, 2]
        ax.scatter3D(xdataC, ydataC, zdataC, c=(rgb_plot/255), s=5)       
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.ion()
        plt.show()
        
    # SHOW desired images if enabled
    if(SHOW_IMAGES):
           
        # GET thin laser image
        subpix_laser = np.zeros((rheight,rwidth),dtype=np.uint8)
        for j in range(len(int_indxs)):
            subpix_laser[int_indxs[j,0],int_indxs[j,1]] = 255
            
        coloured_img = np.zeros((rheight,rwidth,3),dtype=np.uint8)
        coloured_img[:, :, 2] = opened+subpix_laser
        coloured_img[:, :, 1] = subpix_laser
        coloured_img[:, :, 0] = subpix_laser 
        
        #General Header Title
        title = "Image pair {} and {} laser".format(i, i+1)
        # SHOW Original Image pair
        img_pair_show = cv2.resize(np.hstack((img1, img2)), None,
                                   fx=0.3, fy=0.3)
        cv2.imshow("original image pair: ",img_pair_show)

        # SHOW non-treated thresholded image
        threshed_show = cv2.resize(threshed, None, fx=0.5, fy=0.5)
        cv2.imshow(title+"TreshNormal", threshed_show)
        
        # SHOW segmented laser image
        coloured_img_show = cv2.resize(coloured_img, None, fx=0.5, fy=0.5)
        cv2.imshow(title + "Coloured", coloured_img_show)
        
        # SHOW contrast increased image from laser segmentation
        contrast_increase_show = cv2.resize(contrast_increase, None,
                                            fx=0.5, fy=0.5)
        cv2.imshow(title + "Contrasted",contrast_increase_show)
        
        opened_show = cv2.resize(opened, None, fx=0.5, fy=0.5)
        cv2.imshow(title + "After Morphology", opened_show)
        
        cv2.waitKey(IMG_WAITKEY)
        cv2.destroyAllWindows() 
        plt.close('all')

#----------------------------------------------------------------------------#
# --------------------- WHOLE POINT CLOUD FILTERING -------------------------#
#----------------------------------------------------------------------------#

print ("POINT CLOUD OBTAINED. {} IMAGE PAIRS WERE PROCESSED!"
       .format(len(images)/2))

# TURN lists into arrays

#points
point_cloudALL = point_cloud_points_global[0]
if len(point_cloud_points_global) > 1:
    for f in range(1,len(point_cloud_points_global)):
        point_cloudALL = np.vstack((point_cloudALL,
                                 point_cloud_points_global[f]))
#color
rgbALL = point_cloud_color_global[0]
if len(point_cloud_color_global) > 1:
    for f in range(1,len(point_cloud_color_global)):
        rgbALL = np.vstack((rgbALL, point_cloud_color_global[f]))

# FILTER complete point cloud, statistically or/and with a bounding cylinder
"""       
point_cloudALL_filt, rgb_ALL_filt = Filter_3D_Data(point_cloudALL, rgbALL,
                                    factor=4.0, verbose=VERBOSE, apply=True,
                                     plot=PLOT_3D_FILTER_ALL)
"""
point_cloudALL_filt, rgb_ALL_filt = Filter_3D_Data_Cilinder(point_cloudALL,
                                                rgbALL,
                                                radius=250, #mms
                                                height=[-500, 100],
                                                verbose=True, apply=True,
                                                plot=True)

# PLOT obtained point cloud using matplotlib (scatter plot)
if(PLOT_3D_FINAL):  
    plt.figure()
    ax = plt.axes(projection='3d') 
    xdataC = point_cloudALL_filt[:, 0]
    ydataC = point_cloudALL_filt[:, 1]
    zdataC = point_cloudALL_filt[:, 2]
    ax.scatter3D(xdataC, ydataC, zdataC, c=(rgb_ALL_filt/255), s=5)       
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.ion()
    plt.show()


# GET date and time  with format (dd/mm/YY H:M:S)
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

# EXPORT point cloud as ascii XYZ file	
Write_xyz_file (point_cloudALL_filt, rgb_ALL_filt,
                filename="Point_cloud_{}.xyz".format(dt_string))
print("Point cloud saved as: ", 
     "Point_cloud_{}.xyz".format(dt_string))

# IMPORT saved point cloud
point_cloud= np.loadtxt("Point_cloud_{}.xyz".format(dt_string), skiprows=1)

# LOAD data as an OPEN3D object (xyz and rgb)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud[:,:3])
pcd.colors = o3d.utility.Vector3dVector(point_cloud[:,3:6]/255)

# VISUALIZE data with OPEN3D, (using GPU is adviced for this step)
o3d.visualization.draw_geometries([pcd], point_show_normal=False)
plt.close('all')