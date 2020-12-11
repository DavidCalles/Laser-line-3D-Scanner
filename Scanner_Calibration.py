# -*- coding: utf-8 -*-
"""
##############################################################################
#
#   MODULE DESCRIPTION:
#        This module performs a scanner calibration from a set of calibration
#       image pairs. From each image, an N number of 2D-3D correspondences are
#       estimated. With correspondences from various images, an overdefined
#       system is posed and a 4x3 transformation matrix is calculated using a
#       least square estimation. To validate the results, the used correspon-
#       dence points are reprojected, and the reprojection error is found.
#       This process is performed fora given number of times, each time is 
#       called an ATTEMPT. The set of calibration image pairs is randomly
#       chosen between attempts. 
#       After configurable number of attempts, the attempt with the smallest
#       reprojection error (calculated as the L2 norm for each axis) is 
#       showed and saved. 
#       This process allows the reconstruction from image coordinates to 
#       real 3D world coordinates.
#       
#
#   PRE-REQUISITES: 
#        Calibration image pairs set: A set of images of the chessboard with 
#               a lasered and a laser-less image for each pose. At leas 40-50.
#        Camera calibration: A camera calibration must be performed in advance
#               the 'CameraCalibration.npz' must be in same folder.
#
#   OUTPUTS:
#        Scanner Calibration: 4x3 transformation matrix from attempt with
#               least error. As well as the 2D-3D correspondences na the
#               filenames of the images used.
#        
#   USAGE:
#         1. Change "fpath" variable with the direction of calibration images
#                 to be used.
#         2. Change the feature enabling variables to the desired ones. 
#                 By default a summary is showed at the end of each attemp for
#                 a couple of seconds. And the final result is saved in a 
#                 'ScannerCalibration.npz' file.                  
#         3. Change the amount of ATTEMPTS and the size of the image set per
#                 attempt. "ATTEMPTS" & "RANDOM_IMAGE_SET".
#         4. Change the amount of points to extract per image.
#                 "POINTS_PER_IMAGE"
#       
#    
#    ACTION:                             DATE:           NAME:
#        First implementation and tests      20/Nov/2020      David Calles
#        Code comments and last review       9/Dic/2020       David Calles           
#
##############################################################################
"""

#----------------------------------------------------------------------------#
# ----------------------- REQUIRED EXTERNAL PACKAGES ------------------------#
#----------------------------------------------------------------------------#
import cv2
import numpy as np
import random
import glob
from matplotlib import pyplot as plt
from scipy.optimize import leastsq
from skspatial.objects import Points, Plane
from skspatial.plotting import plot_3d
from os.path import exists

#----------------------------------------------------------------------------#
# ----------------------- REQUIRED OWN FUNCTIONS-----------------------------#
#----------------------------------------------------------------------------#
from Algorithms import Simple_Image_Correction
from Algorithms import Double_Lagrange_Aprox_Image
from Algorithms import Laser_segmentation
from Algorithms import order_points
from Algorithms import Get_Line
from Algorithms import line_intersection
from Algorithms import Draw_Quadrangle
from Algorithms import Point_projection
from Algorithms import Get_arbitrary_points_from_line
from Algorithms import rigid_transform_3D
from Algorithms import Get_Overdefined_Mtx

# CLOSE existing figures/images
plt.close('all') 
cv2.destroyAllWindows()

# SET path to calibration images
fpath = "Scanner_Calibration_Set/"

# SET images to use
img_names= glob.glob(fpath+"*.png")         
MAX_IMAGES = len(img_names)#212
USE_IMGS = len(img_names)#212 if all
IMAGE_NUM = min((MAX_IMAGES,USE_IMGS))

#----------------------------------------------------------------------------#
# ----------------------- FEATURE-ENABLING VARIABLES-------------------------#
#----------------------------------------------------------------------------#

# SHOW time and size of images
SHOW_TIME = 100 #ms For each image pair(0 to make it manual)
ITERATION_WAITKEY = 100# For each iteration(0 for wait until action )
ATTEMPT_WAITKEY = 3000# For each attempt(0 for wait until action )
ONE_IMG_SIZE = 0.7 #img show size modifier for ONE image
TWO_IMG_SIZE = 0.4 #img show size modifier for TWO image in  single

# SHOW additional information
VERBOSE_PROJECTION = False
VERBOSE_PARABOLA_APROX = False
VERBOSE_ATTEMPT = False
VERBOSE_MIN_ATTEMPT = True
SAVE = True

# ENABLE which images to show
SHOW_ORIGINALS = False
SHOW_UNDISTORTED = False
SHOW_SEGMENTED_LASER = False
SHOW_COLOURED = False
SHOW_CORNERS = False
SHOW_QUADRANGLE = False
SHOW_PROJECTION_SUMMARY = False

# ENABLE which 2-D plots to show
PLOT_ALL_POINTS = False
PLOT_LINE_APROXIMATION = False
PLOT_TRNSF_2D_LASER_SINGLE = False

# Enable which 3-D plots to show
PLOT_ORIGINAL_3D_CORNERS_SINGLE = False
PLOT_TRNSF_3D_CORNERS_SINGLE = False
PLOT_TRNSF_3D_CORNERS_ALL = False
PLOT_FILTERED_3D_POINTS = False
PLOT_REPROJECTED_CORNERS_ALL = False
PLOT_3D_ESTIMATED_COORDINATES= False
PLOT_3D_INVERTED_DATA = False

#----------------------------------------------------------------------------#
# ---------------------------- GLOBAL USE VARIABLES--------------------------#
#----------------------------------------------------------------------------#
# IMAGE SIZE
img_size = cv2.imread(img_names[0])
rwidth=img_size.shape[1] #e.g 1920
rheight=img_size.shape[0] #e.g 1080
csize = [7,9]   # Chessboard size
rsize=(rwidth, rheight)

# IMAGE SHOW FLAG
my_flag = (cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)

# TERMINATION CRITERIA
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# OBJECT POINTS
objp = np.zeros((csize[0]*csize[1],3), np.float32)
sq_size = 20.0 #mm
objp[:,:2] = np.mgrid[0:csize[0],0:csize[1]].T.reshape(-1,2)
objp *= sq_size
objp = np.column_stack((objp[:,1], objp[:,0], objp[:,2]))
y_inv = np.flip(objp[:,1])
objp2 = np.column_stack((objp[:,0], y_inv, objp[:,2]))
objp = objp2.copy()
z0 = 0

#DRAWING VARIABLES
line_type = 8
thickness_show = 5
thickness = 1
my_color  = (219,38,116) # purplish
my_color2 = (17,137,255) # orangish

# RANDOM SET SELECTION
ATTEMPTS = 300
RANDOM_IMAGE_SET = 25 #Number of image pairs to use
iter_images = range(0, IMAGE_NUM, 2)

# CORRESPONDENCES PER IMAGE
POINTS_PER_IMAGE = 15 

# CAMERA MTX
with np.load('CameraCalibration.npz') as file:
    cam_mtx, _, _, _, _, _ = [file[i] for i in (
    'mtx','dist','rvecs','tvecs', 'newcameramtx', 'mean_error')]
Cx = cam_mtx[0,2]
Cy = cam_mtx[1,2]

#----------------------------------------------------------------------------#
# --------------------- LOCAL FUNCTIONS DEFINITIONS -------------------------#
#----------------------------------------------------------------------------#

"""***************************************************************************   
# NAME: LaserPoints_Filtering_3Dplane
# DESCRIPTION: Filter a set of 3D points to approach a given plane in 3D.
#               
# PARAMETERS:  plane: Coefficients of the plane equation with the form 
#                       A*x+B*y+C*z+D=0. Coeffs=[A,B,C,D]
#              points: Points to fit. [X,Y,Z]. (3-column matrix)
#              percent: Amount of fitting to be performed. 0=None, 1=Max.
#              plot_filtered: Plot new points vs original points. 
#                               Default=True. 
#                              
# RETURNS:     filtered: Filtered 3D points as 3-column matrix.              
#               
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  17/Nov/2020      David Calles
#       Review and documentation        9/Dec/2020       David Calles
***************************************************************************"""
def LaserPoints_Filtering_3Dplane(plane, points, percent, plot_filtered=True): 
    # GET plane equation coefficients
    A = plane[0]
    B = plane[1]
    C = plane[2]
    D = plane[3]
    # GET equivalent plane points z
    z_points = -(D + (points[:,0]*A) + (points[:,1]*B))/C
    plane_points = np.column_stack((points[:,0], 
                                    points[:,1],
                                    z_points))
    # CALCULATE difference between originals and plane points
    diffs = plane_points-points
    
    # CALCULATE filtered values with a "percent" of plane fitting
    filtered = plane_points.copy()+ ((1-percent)*diffs)
    
    # PLOT results if enabled
    if plot_filtered:
        plt3d = plt.figure().gca(projection='3d') 
        ax = plt.gca() 
        X = points[:,0]
        Y = points[:,1]
        Z = points[:,2]
        ax.scatter3D(X, Y, Z, c= 'r')
        X = filtered[:,0]
        Y = filtered[:,1]
        Z = filtered[:,2]
        ax.scatter3D(X, Y, Z, c= 'g')
        ax.scatter3D(0, 0, 0, c= 'b',marker='^')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.view_init(elev=90, azim=-90)
        plt.show()
    
    return filtered
    
"""***************************************************************************   
# NAME: Fit_Plane2Data 
# DESCRIPTION:  Fit a plane to a given data using least squares. 
#               
# PARAMETERS:   data: 3D data to estimate the plane. (3-column matrix)[X,Y,Z]
#               verbose: Show additional information. Default=True.
#                              
# RETURNS:      sol: Coefficients of the equation A*x+B*y+C*z+D=0.
#                    Coeffs=[A,B,C,D].  
#               normal: Normal vector of the plane    
#               point:  A point in the plane.
#               error: Residual squared error.     
#               
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  17/Nov/2020      David Calles
#       Review and documentation        9/Dec/2020       David Calles
***************************************************************************"""
def Fit_Plane2Data(data, verbose=True):   
    # GET data to be fitted
    XYZ = data.T
    
    # SET initial guess (whichever)
    p0 = [-0.99784203156, -0.0069526644205, 0.065291197680, -60.01146044]
    
    # DEFINE function to be minimized
    def f_min(X,p):
        plane_xyz = p[0:3]
        distance = (plane_xyz*X.T).sum(axis=1) + p[3]
        return distance / np.linalg.norm(plane_xyz)
    
    # DEFINE residuals function
    def residuals(params, signal, X):
        return f_min(X, params)
    
    sol = leastsq(residuals, p0, args=(None, XYZ))[0]
    error = (f_min(XYZ, sol)**2).sum()
    
    # SHOW results if enabled
    if verbose:
        print("Solution: ", sol)
        print("Old Error: ", (f_min(XYZ, p0)**2).sum())
        print("New Error: ", error)
    normal = sol[0:3]/(np.linalg.norm(sol[0:3]))
    point = np.array([0,0,-sol[3]/sol[2]])
    
    return sol, normal, point, error

"""***************************************************************************   
# NAME: Project Laser coordinates 
# DESCRIPTION:  This function gets a set of 2D-3D correspondences from an
#               image. It uses the complete quadrangle principle, the 
#               cross-ratio of a pencil of lines, and line intersections to
#               calcualte a given number of correspondences in the laser
#               line. 
#               
# PARAMETERS:   imgBorders: image used to draw a summary of the process,
#                           the quadrangle, projection lines, laser-line, and
#                           points are drawn in this image.
#               img1: Original image used for middle step processes.
#               corners: Corners of chessboard with subpixel resolution.
#               subpxs: laser coordinates with subpixel resolution.
#               points_amount: number of correspondences to extract per image.
#               i: iteration number (left for backward compatibility)
#               draw_quadrangle: Show image with complete quadrangle. 
#                                Default=True.
#               draw_summary: Draw sommary of process. "imgBorders" is used. 
#                             Default=True.
#               verbose: Show additional information. Default=True.
#                              
# RETURNS:      p3D: 3D-correspondences. (3-column matrix)[X,Y,Z].
#               p2D: 2D-correspondences. (2-column matrix)(column,row).              
#               
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  17/Nov/2020      David Calles
#       Review and documentation        9/Dec/2020       David Calles
***************************************************************************"""
def Project_laser_coordinates(imgBorders, img1, corners, subpxs, points_amount,
                              i, draw_quadrangle=True, draw_summary=True,
                              verbose=True):
    # GET laser coordinates as integers
    int_indx = np.uint16(np.around(subpxs))
    # SET trapezoid points in chessboard
    single_obj_trapezoid = np.array([[sq_size*0, sq_size*0, 0],
                                     [sq_size*1, sq_size*6, 0],
                                     [sq_size*8, sq_size*5, 0],
                                     [sq_size*8, sq_size*0, 0]])
    
    # GET object's complete quadrangle points (2-D)
    obj_G_2d = line_intersection(
        (single_obj_trapezoid[1],single_obj_trapezoid[2]),
        (single_obj_trapezoid[0],single_obj_trapezoid[3]))
        
    obj_F_2d = line_intersection(
        (single_obj_trapezoid[0],single_obj_trapezoid[1]),
        (single_obj_trapezoid[3],single_obj_trapezoid[2]))
    
    # ASSUME Z=zinit for the whole object
    obj_G = np.append(obj_G_2d, 0)
    obj_F = np.append(obj_F_2d, 0)
    
    # GET trapezoid image points
    if((corners[0,0] < corners[62,0]) and 
       (corners[0,1] > corners[62,1])):
        img_one_trapezoid = np.array([corners[0,:], corners[13,:], 
                                      corners[61,:], corners[56,:]])
    # GET trapezoid image points (inverted)
    else:
        img_one_trapezoid = np.array([corners[62,:], corners[49,:], 
                                      corners[1,:], corners[6,:]])
    # DRAW trapezoid   
    int_img_trapezoid = np.int32(img_one_trapezoid)
    cv2.polylines(imgBorders, [int_img_trapezoid], True,
                  my_color, thickness_show, line_type)
    
    # CALCULATE complete quadrangle in image (F, G)
    F_quadrangle = line_intersection(
        (img_one_trapezoid[3], img_one_trapezoid[2]),
        (img_one_trapezoid[1], img_one_trapezoid[0]))
    
    G_quadrangle = line_intersection(
        (img_one_trapezoid[1], img_one_trapezoid[2]),
        (img_one_trapezoid[0], img_one_trapezoid[3]))
    
    # SHOW complete quadrangle if enabled
    if (draw_quadrangle):
        # Draw complete quadrangle
        quadrangle_img =  Draw_Quadrangle (img_one_trapezoid,
                                           G_quadrangle,
                                           F_quadrangle,
                                           imgBorders,
                                           thickness=15)
        quadrangle_img_show = cv2.resize(quadrangle_img, None,
                                         fx=0.1,fy=0.1)
        cv2.imshow("Quadrangle Image", quadrangle_img_show)
        cv2.waitKey(SHOW_TIME)
        
    #GET laser points in calibration lines by locating point in
    #       previously drawn 1-pixel wide trapezoid    
    thin_trapezoid = cv2.polylines(img1.copy(), [int_img_trapezoid], True,
                  my_color, thickness, line_type)
    laser_trapezoid = []
    for l in range(len(int_indx)): 
        if( np.array_equal(
                thin_trapezoid[int_indx[l,0],int_indx[l,1],:],
                my_color) ):
            laser_trapezoid.append(subpxs[l])
    upper_point = np.flip(laser_trapezoid[0])
    lower_point = np.flip(laser_trapezoid[-1])
    
    # DRAW upper laser intersection with trapezoid
    cx3 = round(upper_point[0])
    cy3 = round(upper_point[1])
    cv2.circle(imgBorders,(cx3,cy3),10,(255,255,255),-1)  
    # DRAW upper laser intersection with trapezoid 
    cx4 = round(lower_point[0])
    cy4 = round(lower_point[1])
    cv2.circle(imgBorders,(cx4,cy4),10,(255,255,255),-1) 
    # DRAW intersection line (approximation of laser)
    cv2.line(imgBorders, (cx3,cy3), (cx4,cy4),
             (20,20,255), thickness_show)
    
    # USE crossratio to project border 3D laser points
    # Upper border Point
    if verbose:
        print ("--------------PROJECTING BORDER POINTS---------------")
    upper_laser_3D, upper_cr = Point_projection(img_one_trapezoid[1],                                  
                                    upper_point,
                                    img_one_trapezoid[2],
                                    G_quadrangle, 
                                    single_obj_trapezoid[1],
                                    single_obj_trapezoid[2],
                                    obj_G,
                                    validate=False)
    
    # Lower border Point
    lower_laser_3D, lower_cr = Point_projection(img_one_trapezoid[0],                                  
                                   lower_point,
                                   img_one_trapezoid[3],
                                   G_quadrangle, 
                                   single_obj_trapezoid[0],
                                   single_obj_trapezoid[3],
                                   obj_G,
                                   validate=False) 
    # SHOW additional information if enabled
    if verbose:
        print("Upper object POINT ", upper_laser_3D)
        print("Lower object POINT ", lower_laser_3D)
        
    # GET random points in left line of trapezoid
    getpoints = True
    total_iterations = 50
    iterations = 0   
    # ITERATE if 
    while (getpoints and (iterations < total_iterations)):
        arbitrary_pts_img = Get_arbitrary_points_from_line(
            img_one_trapezoid[0], img_one_trapezoid[1], points_amount, 
            verbose = VERBOSE_PROJECTION)
        
        # ---------- Get Projections of arbitrary points ------------------
        arbitrary_pts_obj = []
        if verbose:
            print ("--------------PROJECTING ARBITRARY POINTS---------------")
        for n in range(points_amount):
            projection, _ = Point_projection(img_one_trapezoid[0],
                                             arbitrary_pts_img[n,:],
                                             img_one_trapezoid[1],
                                             F_quadrangle,
                                             single_obj_trapezoid[0],
                                             single_obj_trapezoid[1],
                                             obj_F,
                                             validate=False)
            
            arbitrary_pts_obj.append(projection)
            
        arbitrary_pts_obj = np.array(arbitrary_pts_obj)
        if verbose:
            print ("Projection from arbitrary points: \n", arbitrary_pts_obj)
        
        # GET Intersections with right border (iterate if intersections
        #                                       are not found)
        if verbose:
            print ("--------GETTING INTERSECTIONS WITH RIGHT BORDERS--------")
        right_border_img = []
        right_border_obj = []
        for n in range(points_amount):
            # 2D points from lines intersections
            intersection_2D = line_intersection(
                (img_one_trapezoid[3], img_one_trapezoid[2]),
                (arbitrary_pts_img[n,:], G_quadrangle))
            
            right_border_img.append(intersection_2D)
            
            # 3D points from projection using crossratio
            intersection_3D  = line_intersection(
                (single_obj_trapezoid[3], single_obj_trapezoid[2]),
                (arbitrary_pts_obj[n,:], obj_G))
            intersection_3D = np.append(intersection_3D, z0)
            right_border_obj.append(intersection_3D)
        
        right_border_img = np.array(right_border_img)
        right_border_obj = np.array(right_border_obj)
        if verbose:
            print ("Image right border intersections \n", right_border_img)
            print ("Object right border intersections \n", right_border_obj)
        
        # GET Intersections with LASER in image
            print ("-------GETTING INTERSECTIONS WITH LASER IN IMAGE --------")
        thin_intersections = img1.copy()
        for n in range(points_amount):
            int_arb_img = tuple(np.int16(np.around(arbitrary_pts_img[n,:])))
            int_G_img = tuple(np.int16(np.around(G_quadrangle)))
            cv2.line(thin_intersections, int_arb_img,
                     int_G_img, my_color2, thickness, line_type)
            
        laser_intersections_img = []
        for l in range(len(int_indx)): 
            if( np.array_equal(
                    thin_intersections[int_indx[l,0],int_indx[l,1],:],
                    my_color2) ):
                laser_intersections_img.append(subpxs[l])
                
        laser_intersections_img = np.flip(np.array(laser_intersections_img))
        # VERIFY obtained points
        # If our arbitrary points have no intersection with the laser
        # different points shall be used. Unless more than 
        # certain iterations have been executed 
        if verbose:
            print("Image laser_intersections: \n", laser_intersections_img)
        if (len(laser_intersections_img) != points_amount):
            getpoints = True
            iterations = iterations+1
            if (iterations > total_iterations):
                raise ValueError("RANDOM POINTS ERROR!")
        else:
            getpoints = False
            if verbose:
                print("Iterations founding usefull laser points: ", iterations)
    # GET Projections of Intersections with lasers
    if verbose:
        print ("----------PROJECTING INTERSECTIONS  WITH LASER-------------")
    laser_pts_obj = []
    
    for n in range(points_amount):
        # ESTIMATE points with crossratio projection
        """#Crossratio projection for laser in obj (too noisy)
        projection, _ = Point_projection(arbitrary_pts_img[n,:],
                                         laser_intersections_img[n,:],
                                         right_border_img[n,:],
                                         G_quadrangle,
                                         arbitrary_pts_obj[n,:],
                                         right_border_obj[n,:],
                                         obj_G,
                                         validate=False)
        laser_pts_obj.append(projection)
        """
        # ESTIMATE points with line intersections
        projection2 = line_intersection((arbitrary_pts_obj[n,:], 
                                         right_border_obj[n,:]),
                                        (upper_laser_3D,
                                         lower_laser_3D))
        projection2 = np.append(projection2, z0)
        
        laser_pts_obj.append(projection2)
        
    laser_pts_obj = np.array(laser_pts_obj)
    
    if verbose:
        print ("Projection from laser intersections: \n", laser_pts_obj)
        
    p2D = laser_intersections_img.copy()
    p3D = laser_pts_obj.copy()
    
    # DRAW and SHOW summary of process if enabled
    if draw_summary:
        # DRAW lines and intersecting points 
        for n in range(points_amount):
            int_arb_img = tuple(np.int16(np.around(
                arbitrary_pts_img[n,:])))
            int_G_img = tuple(np.int16(np.around(G_quadrangle)))
            cv2.line(imgBorders, int_arb_img,
                     int_G_img, my_color2, 3, line_type)
            cx = round(laser_intersections_img[n,0])
            cy = round(laser_intersections_img[n,1])
            cv2.circle(imgBorders,(cx,cy),5,(255,0,0),-1) 
        summary_img_show = cv2.resize(imgBorders, None,
                                         fx=ONE_IMG_SIZE,fy=ONE_IMG_SIZE)
        cv2.imshow("Summary Image Features, Images {}".format(i),
                   summary_img_show)
        cv2.waitKey(SHOW_TIME)
        
    # GET usefull points, in case upper and lower points are bad
    if ((upper_point[0] != 0) and (upper_point[1] != 0)):
        p2D = np.row_stack((p2D, upper_point))
        p3D = np.row_stack((p3D, upper_laser_3D))
    if ((lower_point[0] != 0) and (lower_point[1] != 0)):
        p2D = np.row_stack((p2D, lower_point))
        p3D = np.row_stack((p3D, lower_laser_3D))
        
    return p3D, p2D
    
"""***************************************************************************
#                                                                            #
#                                                                            #
#                            MAIN FUNCTION                                   #
#                                                                            #
#                                                                            #
***************************************************************************"""

#----------------------------------------------------------------------------#
# --------- VARIABLES TO ACCUMULATE RESULTS FROM ALL ATTEMPTS----------------#
#----------------------------------------------------------------------------#
Mtx4x3_All_Attempts = []
Points3D_All_Attempts = []
Points2D_All_Attempts = []
Used_Imgs_All_Attempts = []
Error_All_Attempts = []

#----------------------------------------------------------------------------#
# ------------------- PERFORM ALL ATTEMPTS OF CALIBRATION -------------------#
#----------------------------------------------------------------------------#

for a in range(ATTEMPTS):
    print("ATTEMPT {}:".format(a))
#----------------------------------------------------------------------------#
# ----------------- VARIABLES FOR SINGLE ATTEMPT ----------------------------#
#----------------------------------------------------------------------------#
    original_objp = []
    laser_subpixelsALL = []
    filtered_subpxALL = []
    int_laser_subpixelsALL = []
    rgbALL = []
    transf_cornersALL = []
    Laser3D_TransALL = []
    Laser2D_ALL = []
    r_vct0 = np.array([0,0])
    t_vct0 = np.array([0,0])
    SKIPPED = 0
    
    # GET random set of images for single attempt
    iter_images_rand = sorted(random.sample(iter_images, RANDOM_IMAGE_SET))
    it = 0

#----------------------------------------------------------------------------#
# ------------------ ITERATE SELECTED TRHOUGH IMAGE SET ---------------------#
#----------------------------------------------------------------------------#
    for i in iter_images_rand:
        print("\t Iteration ", it)
        # GET image pair filename
        img_filename1 = fpath+"Image_{}.png".format(i)
        img_filename2 = fpath+"Image_{}.png".format(i+1)  
        # VERIFY if image pair exists 
        if (exists(img_filename1) and exists(img_filename2)):           
            imgA    = cv2.imread(img_filename1)
            imgB    = cv2.imread(img_filename2)
            # UNDISTORT image pair
            img1 = Simple_Image_Correction(imgA)
            img2 = Simple_Image_Correction(imgB)
            # GET a single image pair per attempt
            if(it == 0):
                img_test1 = img1.copy()
                img_test2 = img2.copy()
            # SEGMENT laser line and ACQUIRE maximums
            _, _, _, threshed, indxs, indxs2, opened, closed, \
                contrast_increase=Laser_segmentation(img1, img2, thresh=0.32)
            # TURN max values into numpy array   
            ppx = np.array(indxs)
            ppx2 = np.array(indxs2)

            # REFINE laser coordinates to subpixel resolution
            subpxs = Double_Lagrange_Aprox_Image(opened, ppx, ppx2, True,
                                            verbose=VERBOSE_PARABOLA_APROX)
            int_indxs = np.uint16(np.around(subpxs))
            
            # GET colours from laser-less image
            bgr = []
            for j in range(len(int_indxs)):
                current_rgb = img2[int_indxs[j,0],int_indxs[j,1],:]
                bgr.append(current_rgb)
            bgr = np.array(bgr)
            rgb = np.column_stack((bgr[:,2], bgr[:,1], bgr[:,0]))
            
            # ACCUMULATE laser coordinates for all images
            laser_subpixelsALL.append(subpxs)
            int_laser_subpixelsALL.append(int_indxs)
            rgbALL.append(rgb)
            
            # GET grayscale image of laser-less image
            gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
            # FIND the chess board corners
            ret, corners = cv2.findChessboardCorners(gray,
                                (csize[0],csize[1]), None)
            if ret:     
                # REFINE corners coordinates to subpix resolution
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),
                                            criteria) 
                # GET extreme 4 corners of chessborad
                pts4 = np.array([corners2[0,0,:], corners2[6,0,:],
                             corners2[56,0,:], corners2[62,0,:]])
                # ORDER points clockwise
                pts4b = order_points(pts4)
                
                # APPROXIMATE laser data to a line with 95% fitting
                filtered_laser = Get_Line(subpxs, pts4b, 0.95,
                                          plot=PLOT_LINE_APROXIMATION)
                filtered_subpxALL.append(filtered_laser)

                #---------------ESTIMATE POSE OF CHESSBOARD-------------------
                #-------------- WORLD TO CAMERA COORDINATES-------------------
                
                # RESHAPE corners if chessboard upside-down
                corners3 = corners2.reshape(63,2)
                if(corners3[0,0]<corners3[62,0]): 
                    corners3 = np.flip(corners3,axis=0)
                
                # ESTIMATE world to camera coordinate system transformation
                dist_coeffs = np.zeros((5,1)) # Already in image
                (success, rvect, tvect) = \
                                    cv2.solvePnP(objp, corners3,
                                                 cam_mtx, dist_coeffs,
                                                 flags=cv2.SOLVEPNP_ITERATIVE)
                # TURN rotation vector to rotation matrix
                rmtx, jac = cv2.Rodrigues(rvect)
                
                # SET transformation matrix 4x4
                trans_mtx=np.array([[rmtx[0,0],rmtx[0,1],rmtx[0,2],tvect[0,0]]                            
                             ,[rmtx[1,0],rmtx[1,1],rmtx[1,2],tvect[1,0]],
                              [rmtx[2,0],rmtx[2,1],rmtx[2,2],tvect[2,0]],
                              [0,0,0,1]])
                
                # TRANSFORM original chessboard corners 3D coordinates             
                objp_ext = np.hstack((objp, np.ones((63,1))))    
                objpTrans = np.matmul(trans_mtx, objp_ext.T)       
                objpTrans2 = objpTrans[0:3,:].T    
                
                # ACQUIRE 2D-3D correspondences using projections
                laser3D, laser2D = Project_laser_coordinates(img1.copy(),
                                    img1.copy(), corners3, filtered_laser,
                                    points_amount=POINTS_PER_IMAGE, i=i, 
                                    draw_quadrangle=SHOW_QUADRANGLE,
                                    draw_summary=SHOW_PROJECTION_SUMMARY,
                                    verbose=VERBOSE_PROJECTION)
                    
                # TRANSFORM laser 3D coordinates              
                laser3D_ext = np.hstack((laser3D, np.ones((len(laser3D),1))))
                Laser3D_Trans = np.matmul(trans_mtx, laser3D_ext.T) 
                Laser3D_Trans2 = Laser3D_Trans[0:3,:].T
                      
                # GET centroid of chessboard
                centroid = np.average(objpTrans2, axis=0)
                
                # RECTIFY respect to centroid
                Laser3D_Trans2[:,0] = centroid[0] - (Laser3D_Trans2[:,0]- \
                                                     centroid[0])
                objpTrans2[:,0] = centroid[0] - (objpTrans2[:,0]-centroid[0])
                
                # ENSURE order respect to Y value (rows)
                Laser3D_Trans2 = Laser3D_Trans2[np.argsort(
                                                        Laser3D_Trans2[:, 1])]
                laser2D = laser2D[np.argsort(laser2D[:, 1])]
                
                # Laser 2D points
                Laser2D_ALL.append(laser2D)
                # Transformed chessboard
                transf_cornersALL.append(objpTrans2)
                # Transformed laser points
                Laser3D_TransALL.append(Laser3D_Trans2)
                # Original object corners
                original_objp.append(objp)
                
            # SHOW plots of single images if enabled
                
            if PLOT_ORIGINAL_3D_CORNERS_SINGLE:               
                plt3d = plt.figure().gca(projection='3d') 
                ax = plt.gca()
                X = objp[:,0]
                Y = objp[:,1]
                Z = objp[:,2] 
                ax.scatter3D(X, Y, Z, c= 'b')
                X = laser3D[:,0]
                Y = laser3D[:,1]
                Z = laser3D[:,2]
                ax.scatter3D(X, Y, Z, c= 'r')
                ax.scatter3D(0, 0, 0, c= 'g')
                # Ensure that the next plot doesn't overwrite the first plot
                ax.set_xlabel('X axis')
                ax.set_ylabel('Y axis')
                ax.set_zlabel('Z axis')
                ax.view_init(elev=90, azim=-90)
                #ax.set_zlim(-20, 20)
                plt.ion()
                plt.show() 
            if PLOT_TRNSF_3D_CORNERS_SINGLE:
                
                plt3d = plt.figure().gca(projection='3d') 
                ax = plt.gca()
                X2 = objpTrans2[:,0]
                Y2 = objpTrans2[:,1]
                Z2 = objpTrans2[:,2]
                ax.scatter3D(X2, Y2, Z2, c= 'k')
                X2 = Laser3D_Trans2[:,0]
                Y2 = Laser3D_Trans2[:,1]
                Z2 = Laser3D_Trans2[:,2]
                ax.scatter3D(X2, Y2, Z2, c= 'r')
                X2 = Laser3D_Trans2[0,0]
                Y2 = Laser3D_Trans2[0,1]
                Z2 = Laser3D_Trans2[0,2]
                ax.scatter3D(X2, Y2, Z2, s=40, c= 'g')
                X2 = Laser3D_Trans2[1,0]
                Y2 = Laser3D_Trans2[1,1]
                Z2 = Laser3D_Trans2[1,2]
                ax.scatter3D(X2, Y2, Z2, s=40, c= 'b')
                ax.scatter3D(0, 0, 0, c= 'b',marker='^')
                ax.scatter3D(centroid[0], centroid[1], centroid[2],s=50, c= 'b')
                # Ensure that the next plot doesn't overwrite the first plot
                ax.set_xlabel('X axis')
                ax.set_ylabel('Y axis')
                ax.set_zlabel('Z axis')
                ax.view_init(elev=90, azim=-90)
                plt.show()  
            if PLOT_TRNSF_2D_LASER_SINGLE:
                
                xdata2 = laser2D[:,0]
                ydata2 = laser2D[:,1]       
                xdata3 = laser2D[0,0]
                ydata3 = laser2D[0,1]   
                xdata4 = laser2D[1,0]
                ydata4 = laser2D[1,1] 
                # Plot
                area = np.pi*1
                fig = plt.figure()
                plt.gca().invert_yaxis()
                plt.scatter(xdata2, ydata2, s=area, c='r')
                plt.scatter(xdata3, ydata3, s=8*area, c= 'k')
                plt.scatter(xdata4, ydata4, s=8*area, c= 'b')
                plt.title('2D points obtained')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.grid()
                plt.show()
            
            if SHOW_ORIGINALS:
                originals = cv2.resize(np.hstack((imgA, imgB)),None,
                                       fx=TWO_IMG_SIZE, fy=TWO_IMG_SIZE)
                cv2.imshow("ORIGINAL images", originals)
                cv2.waitKey(SHOW_TIME)
            if SHOW_UNDISTORTED:
                undistorteds = cv2.resize(np.hstack((imgA, img1)),None,
                                          fx=TWO_IMG_SIZE, fy=TWO_IMG_SIZE)
                cv2.imshow("UNDISTORTED images", undistorteds)
                cv2.waitKey(SHOW_TIME)
            if SHOW_SEGMENTED_LASER:
                opened_show = cv2.resize(opened,None,
                                         fx=ONE_IMG_SIZE, fy=ONE_IMG_SIZE)
                cv2.imshow("PROCESSED image", opened_show)
                cv2.waitKey(SHOW_TIME)
            if SHOW_COLOURED:
                subpix_laser = np.zeros((rheight,rwidth),dtype=np.uint8)
                for j in range(len(int_indxs)):
                    subpix_laser[int_indxs[j,0],int_indxs[j,1]] = 255
                            
                coloured_img = np.zeros((rheight,rwidth,3),dtype=np.uint8)
                coloured_img[:, :, 2] = opened+subpix_laser
                coloured_img[:, :, 1] = subpix_laser
                coloured_img[:, :, 0] = subpix_laser 
                coloured_show = cv2.resize(coloured_img,None,
                                           fx=ONE_IMG_SIZE, fy=ONE_IMG_SIZE)
                cv2.imshow("COLOURED image", coloured_show)
                cv2.waitKey(SHOW_TIME)
            if SHOW_CORNERS:
                # Draw and display the corners
                imgBorders = cv2.drawChessboardCorners(
                    img1.copy(), (csize[0],csize[1]), corners2,ret)
                # Pintar las 4 esquinas del cuadrilatero
                cv2.circle(imgBorders, tuple(pts4b[0,:]), 16,(0,0,255),-1)
                cv2.circle(imgBorders, tuple(pts4b[1,:]), 16,(255,0,0),-1)
                cv2.circle(imgBorders, tuple(pts4b[2,:]), 16,(0,255,0),-1)
                cv2.circle(imgBorders, tuple(pts4b[3,:]), 16,(0,0,0),-1)
        
                imgBorders_show = cv2.resize(imgBorders,None,
                                         fx=ONE_IMG_SIZE, fy=ONE_IMG_SIZE)
                cv2.imshow("CORNERS DETECTION image", imgBorders_show)
                cv2.waitKey(SHOW_TIME)
        else:
            print("Image pair {}-{} not found. Skipping...".format(i, i+1))
            SKIPPED += 2
        it+=1
        cv2.waitKey(ITERATION_WAITKEY)
        cv2.destroyAllWindows()
        plt.close('all') 
        
#----------------------------------------------------------------------------#
# --------- CALIBRATE SCANNER WITH CORRESPONDENCES FROM SINGLE ATTEMPT ------#
#----------------------------------------------------------------------------# 
        
    USE_IMGSB = RANDOM_IMAGE_SET-SKIPPED
    # RESHAPE arrays containing data from all images
    
    # Laser coordinates in image (all)
    print("\t COLLECTING DATA")
    laser_subpixelsALL2 = laser_subpixelsALL[0]
    for f in range(1,len(laser_subpixelsALL)):
        laser_subpixelsALL2 = np.vstack((laser_subpixelsALL2,
                                         laser_subpixelsALL[f]))
        
    # Integer laser coordinates in image (all)
    int_laser_subpixelsALL2 = int_laser_subpixelsALL[0]
    for f in range(1,len(int_laser_subpixelsALL)):
        int_laser_subpixelsALL2 = np.vstack((int_laser_subpixelsALL2,
                                             int_laser_subpixelsALL[f]))
    # RGB data from lasered area in image    
    rgbALL2 = rgbALL[0]
    for f in range(1,len(rgbALL)):
        rgbALL2 = np.vstack((rgbALL2, rgbALL[f]))
    
    # Laser 2D-correspondences 
    Laser2D_ALL2 = Laser2D_ALL[0]
    for f in range(1,len(Laser2D_ALL)):
        Laser2D_ALL2 = np.vstack((Laser2D_ALL2,Laser2D_ALL[f]))
    
    # Filtered Laser coordinates in image (all)(fitted to line at 95%)
    filtered_subpxALL2 = filtered_subpxALL[0]
    for f in range(1,len(filtered_subpxALL)):
        filtered_subpxALL2 = np.vstack((filtered_subpxALL2,
                                        filtered_subpxALL[f]))
        
    # Transformed 3D chessboard corners (world frame)
    transf_cornersALL2 = transf_cornersALL[0]
    for f in range(1,len(transf_cornersALL)):
        transf_cornersALL2 = np.vstack((transf_cornersALL2,
                                        transf_cornersALL[f]))
    
    # Laser 3D-correspondences (world frame)
    Laser3D_TransALL2 = Laser3D_TransALL[0]
    for f in range(1,len(Laser3D_TransALL)):
        Laser3D_TransALL2 = np.vstack((Laser3D_TransALL2,
                                       Laser3D_TransALL[f]))
    
    original_objp = np.array(original_objp)
    
    print("\t FILTERING DATA ( PLANE FIT )")
    # FIT plane to transformed data using numpy
    planeA, normalA, pointA, errorA = Fit_Plane2Data(Laser3D_TransALL2.copy(),
                                                     verbose=False)   
    planeAobj = Plane(pointA,normalA)
    # PLOT fitted plane using scipy
    points = Points(Laser3D_TransALL2.copy())
    plot_3d(
        points.plotter(c='k', s=50, depthshade=False),
        planeAobj.plotter(alpha=0.3, lims_x=(-60, 30), lims_y=(-60, 60)),
    )
    
    # SMOOTH (filter) points corresponding to laser plane at a 99%
    filtered_3dlaser = LaserPoints_Filtering_3Dplane(planeA,
                                        Laser3D_TransALL2,
                                        percent = 0.99,
                                        plot_filtered=PLOT_FILTERED_3D_POINTS)
    
    # ----------- RETURN DATA INTO WORLD COORDINATE SYSTEM -------------------
    dataA = transf_cornersALL2[0:63,:]
    dataB = original_objp[0,:,:]
    # ESTIMATE rotation matrix and translation vector
    r_mtx0, t_vct0= rigid_transform_3D(dataA.T, dataB.T)
    dist_coeffs = np.zeros((5,1)) # Already in image
    # SET transformation matrix
    trans_mtx2 = np.array([[r_mtx0[0,0],r_mtx0[0,1],r_mtx0[0,2],t_vct0[0,0]],
                           [r_mtx0[1,0],r_mtx0[1,1],r_mtx0[1,2],t_vct0[1,0]],
                           [r_mtx0[2,0],r_mtx0[2,1],r_mtx0[2,2],t_vct0[2,0]],
                           [0,0,0,1]])
    # RETURN corners to original world coordinate frame
    data = transf_cornersALL2[:,:]
    corners_extW = np.hstack((data, np.ones((len(data),1))))  
    corners_transW = np.matmul(trans_mtx2, corners_extW.T)       
    corners_transW2 = corners_transW[0:3,:].T
    
    # RETURN laser coordinates to original world coordinate frame
    data2 = filtered_3dlaser[:,:]
    laser_extW2 = np.hstack((data2, np.ones((len(data2),1))))  
    laserTransWB = np.matmul(trans_mtx2, laser_extW2.T)       
    laserTransW2B = laserTransWB[0:3,:].T
    
    # RETURN USABLE LASER coordinates to original world coordinate frame
    data3 = filtered_3dlaser[:,:]
    laser_extW3 = np.hstack((data3, np.ones((len(data3),1))))  
    laserTransWC = np.matmul(trans_mtx2, laser_extW3.T)       
    laserTransW2C = laserTransWC[0:3,:].T
    
    # FIT corresponding plane
    planeB, normalB, pointB, errorB = Fit_Plane2Data(laserTransW2B.copy(),
                                                     verbose=False)   
    planeBobj = Plane(pointB,normalB)
    # PLOT using scipy
    pointsB = Points(laserTransW2B.copy())
    plot_3d(
        pointsB.plotter(c='b', s=50, depthshade=False),
        planeBobj.plotter(alpha=0.5, lims_x=(-60, 30), lims_y=(-60, 60)),
    )
    
    # ---------- PREPARE DATA -------------------------------------------------
    # Laser
    Laser3D_TransALL2_inv = np.column_stack((filtered_3dlaser[:,0],
                                             filtered_3dlaser[:,1],
                                             filtered_3dlaser[:,2]))
    # Corners
    transf_cornersALL2_inv = np.column_stack((transf_cornersALL2[:,0],
                                              transf_cornersALL2[:,1],
                                              transf_cornersALL2[:,2]))
    # Img points
    Laser2D_ALL2_inv = np.column_stack((Laser2D_ALL2[:,0],
                                        Laser2D_ALL2[:,1]))
    
    image_laser_rep = img_test1.copy()
    idxs = Laser2D_ALL2_inv.astype(int)
    color = np.array([0,255,0])
    for f in range(len(idxs)):
        image_laser_rep[idxs[f,1],idxs[f,0], :] = color
        image_laser_rep[idxs[f,1],idxs[f,0]+1, :] = color
        image_laser_rep[idxs[f,1],idxs[f,0]+2, :] = color
        image_laser_rep[idxs[f,1]+1,idxs[f,0], :] = color
        image_laser_rep[idxs[f,1]+1,idxs[f,0]+1, :] = color
        image_laser_rep[idxs[f,1]+1,idxs[f,0]+2, :] = color
    
    image_laser_rep_show = cv2.resize(image_laser_rep, None,
                                      fx=ONE_IMG_SIZE+0.1,
                                      fy=ONE_IMG_SIZE+0.1)
    cv2.namedWindow("USING COORDINATES ...", flags=my_flag)
    cv2.imshow("USING COORDINATES ...", image_laser_rep)
    cv2.waitKey(ATTEMPT_WAITKEY)
    # ---------- CREATE overdefined matrix ----------------------------------
    mtxA_total, vct1_total = Get_Overdefined_Mtx(Laser2D_ALL2_inv,
                                                 Laser3D_TransALL2_inv)
    
    # ---------- SOLVE general overdetermined mtx from lstsq method ---------
    print (" \t LSTSQ ALGORITHM ")
    vct_T_gen, res_gen, rank_gen, s_gen = np.linalg.lstsq(mtxA_total.copy(),
                                                          vct1_total.copy(),
                                                          rcond=-1)
    # SHOW lstsq solution if enabled
    if VERBOSE_ATTEMPT:
        print("GENERAL Solution: \n", vct_T_gen)
        print("GENERAL Residuals: \n", res_gen)
        print("GENERAL Rank: \n", rank_gen)
        print("GENERAL S value: \n", s_gen)
            
    Mtx4x3_Total = np.append(vct_T_gen, 1).reshape(4,-1)
    # SHOW transformation matrix if enabled
    if VERBOSE_ATTEMPT:
        print("\nGENERAL 4x3 transformation mtx: \n", Mtx4x3_Total)
    # VALIDATE transformation matrix with single point
        print("Validation Point :\n")
    img_point1 = np.append(Laser2D_ALL2_inv[5,:], 1)
    obj_point1 = Mtx4x3_Total.dot(img_point1)
    obj_point1 = (obj_point1/obj_point1[3])[:3]
    if VERBOSE_ATTEMPT:
        print(obj_point1)
        print(Laser3D_TransALL2_inv[5,:])
        
    # REPROJECT points
    print (" \t REPROJECTION ERROR")
    img_points_rep = np.hstack((Laser2D_ALL2_inv,
                                np.ones((len(Laser2D_ALL2_inv),1))))
    results1 = np.matmul(Mtx4x3_Total, img_points_rep.T)
    results2 = results1.T
    results3 = results2.copy()[:,0:3]
    for f in range(len(results2)):
        results3[f,:] = results3[f,:]/results2[f,3]
        
    # CALCULATE reprojection average error
    mean_error = np.average(np.abs(Laser3D_TransALL2_inv-results3),axis=0)
    if VERBOSE_ATTEMPT:
        print ("MEAN REPROJECTION ERROR (X,Y,Z): ", mean_error)
    
    # SHOW plots from each attempt 
        
    if PLOT_3D_INVERTED_DATA:
        plt3d = plt.figure().gca(projection='3d') 
        ax = plt.gca() 
        XALL = transf_cornersALL2_inv[:,0]
        YALL = transf_cornersALL2_inv[:,1]
        ZALL = transf_cornersALL2_inv[:,2]
        ax.scatter3D(XALL, YALL, ZALL, s=8, c= 'b')
        XALL = Laser3D_TransALL2_inv[:,0]
        YALL = Laser3D_TransALL2_inv[:,1]
        ZALL = Laser3D_TransALL2_inv[:,2]
        ax.scatter3D(XALL, YALL, ZALL, s=8, c= 'r')
        firsts = np.array([0,0,0])
        for f in range(int(USE_IMGSB/2)):
            firsts = np.vstack((firsts, transf_cornersALL2_inv[int(f*63),:]))
        XALL = firsts[:,0]
        YALL = firsts[:,1]
        ZALL = firsts[:,2]
        ax.scatter3D(XALL, YALL, ZALL, s=20, c= 'k')
        firsts = np.array([0,0,0])
        for f in range(int(USE_IMGSB/2)):
            firsts = np.vstack((firsts,
                                Laser3D_TransALL2_inv[int(f*POINTS_PER_IMAGE+2),:]))
        XALL = firsts[:,0]
        YALL = firsts[:,1]
        ZALL = firsts[:,2]
        ax.scatter3D(XALL, YALL, ZALL, s=35, c= 'k')
        ax.scatter3D(transf_cornersALL2_inv[0,0], transf_cornersALL2_inv[0,1],
                     transf_cornersALL2_inv[0,2], s=50, c= 'g', marker='^') 
        ax.scatter3D(0, 0, 0, c= 'k',s=50, marker='^')
        # Ensure that the next plot doesn't overwrite the first plot
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.view_init(elev=90, azim=-90)
        plt.show()
        
    if PLOT_3D_ESTIMATED_COORDINATES:
        plt3d = plt.figure().gca(projection='3d') 
        ax = plt.gca() 
        XALL = Laser3D_TransALL2_inv[:,0]
        YALL = Laser3D_TransALL2_inv[:,1]
        ZALL = Laser3D_TransALL2_inv[:,2]
        ax.scatter3D(XALL, YALL, ZALL, c= 'g', s=1)
        XALL = results3[:,0]
        YALL = results3[:,1]
        ZALL = results3[:,2]
        ax.scatter3D(XALL, YALL, ZALL, c= 'r', s=1)
        #First points in laser original
        firsts = np.array([0,0,0])
        for f in range(int(USE_IMGSB/2)):
            firsts = np.vstack((firsts,
                            Laser3D_TransALL2_inv[int(f*POINTS_PER_IMAGE+2),:]))
        XALL = firsts[:,0]
        YALL = firsts[:,1]
        ZALL = firsts[:,2]
        ax.scatter3D(XALL, YALL, ZALL, s=60, c= 'k') 
        #First points in laser estimated
        firsts = np.array([0,0,0])
        for f in range(int(USE_IMGSB/2)):
            firsts = np.vstack((firsts, results3[int(f*POINTS_PER_IMAGE+2),:]))
        XALL = firsts[:,0]
        YALL = firsts[:,1]
        ZALL = firsts[:,2]
        ax.scatter3D(XALL, YALL, ZALL, s=60, c= 'b') 
        ax.set_xlim([-200,200])
        ax.set_ylim([-200,200]) 
        ax.set_zlim([-10,700]) 
        ax.scatter3D(0, 0, 0, c= 'b',marker='^')
        # Ensure that the next plot doesn't overwrite the first plot
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.view_init(elev=90, azim=-90)
        plt.show()
    
    if PLOT_ALL_POINTS:
        #xdata = laser_subpixelsALL2[:,1]
        #ydata = laser_subpixelsALL2[:,0]
        
        xdata2 = filtered_subpxALL2[:,1]
        ydata2 = filtered_subpxALL2[:,0]
        
        xdata3 = Laser2D_ALL2_inv[:,0]
        ydata3 = Laser2D_ALL2_inv[:,1]
        
        firsts = np.array([1000,0])
        for f in range(int(USE_IMGSB/2)):
            firsts = np.vstack((firsts,
                                Laser2D_ALL2_inv[int(f*POINTS_PER_IMAGE+2),:]))
        XALL = firsts[:,0]
        YALL = firsts[:,1]     
        
        # Plot
        area = np.pi*1
        fig = plt.figure()
        plt.gca().invert_yaxis()
        #plt.scatter(xdata, ydata, s=area, c=rgbALL2/255)
        plt.scatter(xdata2, ydata2, s=area, c='r')
        plt.scatter(xdata3, ydata3, s=area, c='g')
        plt.scatter(XALL, YALL, s=8*area, c= 'k')
        plt.title('2D points obtained')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid()
        plt.show()
        
    if PLOT_REPROJECTED_CORNERS_ALL:
        
        plt3d = plt.figure().gca(projection='3d') 
        ax = plt.gca() 
        
        XALL = corners_transW2[:,0]
        YALL = corners_transW2[:,1]
        ZALL = corners_transW2[:,2]
        ax.scatter3D(XALL, YALL, ZALL,s=8, c= 'b')
        XALL = laserTransW2C[:,0]
        YALL = laserTransW2C[:,1]
        ZALL = laserTransW2C[:,2]  
        ax.scatter3D(XALL, YALL, ZALL,s=8, c= 'k')
        firsts = np.array([0,0,0])
        for f in range(int(USE_IMGSB/2)):
            firsts = np.vstack((firsts, corners_transW2[int(f*63),:]))
        XALL = firsts[:,0]
        YALL = firsts[:,1]
        ZALL = firsts[:,2]
        ax.scatter3D(XALL, YALL, ZALL, s=20, c= 'k') 
        ax.scatter3D(0, 0, 0, c= 'b',marker='^')
        # Ensure that the next plot doesn't overwrite the first plot
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.view_init(elev=90, azim=-90)
        plt.show()
        plt.show()
        
    if PLOT_TRNSF_3D_CORNERS_ALL:
        
        plt3d = plt.figure().gca(projection='3d') 
        ax = plt.gca() 
        XALL = transf_cornersALL2[:,0]
        YALL = transf_cornersALL2[:,1]
        ZALL = transf_cornersALL2[:,2]
        ax.scatter3D(XALL, YALL, ZALL, s=8, c= 'b')
        XALL = Laser3D_TransALL2[:,0]
        YALL = Laser3D_TransALL2[:,1]
        ZALL = Laser3D_TransALL2[:,2]
        ax.scatter3D(XALL, YALL, ZALL, s=8, c= 'r')
        firsts = np.array([0,0,0])
        for f in range(int(USE_IMGSB/2)):
            firsts = np.vstack((firsts, transf_cornersALL2[int(f*63),:]))
        XALL = firsts[:,0]
        YALL = firsts[:,1]
        ZALL = firsts[:,2]
        ax.scatter3D(XALL, YALL, ZALL, s=20, c= 'k')   
        ax.scatter3D(transf_cornersALL2[0,0], transf_cornersALL2[0,1],
                     transf_cornersALL2[0,2], s=50, c= 'g', marker='^') 
        ax.scatter3D(0, 0, 0, c= 'k',s=50, marker='^')
        # Ensure that the next plot doesn't overwrite the first plot
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.view_init(elev=90, azim=-90)
        plt.show()
            
        cv2.destroyAllWindows()    
    plt.close('all') 
    cv2.destroyAllWindows()            
    
    # ACCUMULATE data from each attempt
    Mtx4x3_All_Attempts.append(Mtx4x3_Total)
    Points3D_All_Attempts.append(Laser3D_TransALL2_inv)
    Points2D_All_Attempts.append(Laser2D_ALL2_inv)
    Used_Imgs_All_Attempts.append(iter_images_rand)
    Error_All_Attempts.append(mean_error)

#----------------------------------------------------------------------------#
# ----------------- SEARCH FOR ATTEMPT WITH MINIMUM ERROR -------------------#
#----------------------------------------------------------------------------# 
    
error_magn = []
for err in range(len(Error_All_Attempts)):
    error_magn.append(np.linalg.norm(Error_All_Attempts[err]))
error_magn = np.array(error_magn)

minimum_error = np.amin(error_magn)
min_idx = np.argmin(error_magn)

# SHOW result with minimum error if enabled
if VERBOSE_MIN_ATTEMPT: 
    print("\n \n TOTAL ATTEMPTS {} OF {} IMAGE PAIRS EACH \n"
          .format(ATTEMPTS,RANDOM_IMAGE_SET))
    print("ATTEMPT {} had smaller error.".format(min_idx))
    print("With Magnitude of {}".format(minimum_error))
    print("[X,Y,Z] error: \n", Error_All_Attempts[min_idx])
    print("IMAGE PAIRS USED: \n", Used_Imgs_All_Attempts[min_idx])
    print("Transformation matrix: \n", Mtx4x3_All_Attempts[min_idx])

# SAVE results with minimum error if enabled
if SAVE:
    # Saving Variables
    np.savez('ScannerCalibration.npz', 
             Mtx_4x3 = Mtx4x3_All_Attempts[min_idx],
             points2D = Points2D_All_Attempts[min_idx],
             points3D = Points3D_All_Attempts[min_idx],
             error = Error_All_Attempts[min_idx], 
             used_imgs = Used_Imgs_All_Attempts[min_idx])
    print("Files succesfully saved in ScannerCalibration.npz")
    



