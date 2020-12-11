# -*- coding: utf-8 -*-
"""
##############################################################################
#
#   MODULE DESCRIPTION:
#        This module contains most of the algorithms used in the calibration 
#       and reconstruction process, such as laser segmentation, 
#       laser refining, point projection, over-defined system formation, and 
#       many ohers.
#
#   PRE-REQUISITES: 
#        Camera calibration: Intrinsic matrix and distortion coefficients 
#                           vector(used in Simple_Image_Correction function).
#                           'CameraCalibration.npz' must be in same folder.       
#    
#    ACTION:                             DATE:           NAME:
#        First implementation and tests      10/Aug/2020      David Calles
#        Code comments and last review       9/Dic/2020       David Calles           
#
##############################################################################
"""

#----------------------------------------------------------------------------#
# ----------------------- REQUIRED EXTERNAL PACKAGES ------------------------#
#----------------------------------------------------------------------------#
import numpy as np
import cv2
import math as mt
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize
from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import lagrange
import random

#----------------------------------------------------------------------------#
# ----------------------- FEATURES ENABLING VARIABLES------------------------#
#----------------------------------------------------------------------------#

# SET variables for thresholding type in laser segmentation
NORMAL_THRESHOLDING=0
OTSU_THRESHOLDING=1

# ENABLE features (show images, show plots)
SHOW_IMAGES=True # Show images
PLOT=True   #SHow Graphs
SAVE_IMAGES=True # Save images to files
JUST_BALL=True # Just iterate thru first pair of images (test)
SUBPIX_GRAPH = False # Graph Subpix aproximation (1 per iamge)
ROW_TO_GRAPH = 509# Row to graph
PLOT_3D = True

# SET image size 
rwidth=mt.floor(1920)
rheight=mt.floor(1080)
rsize=(rwidth, rheight)
my_flag = (cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)

# SET variables for drawing in image
line_type = 8
thickness_show = 5
thickness = 1
my_color  = (219,38,116) # purplish
my_color2 = (17,137,255) # orangish

#----------------------------------------------------------------------------#
# ----------------------- FUNCTION DEFINITIONS ------------------------------#
#----------------------------------------------------------------------------#

"""***************************************************************************   
# NAME: Calculate_3D_Points
# DESCRIPTION:  Appy 4x3 Calculated matrix to estimate 3D coordinates from
#               image coordinates.
#               
# PARAMETERS:   mtx4x3: 4x3 transformation matrix
#               points2D: Image points for estimation. (2-column mtx)
#                         (column,row)
#                              
# RETURNS:      points3D2: Estimated 3D coordinates of image points.
#                           (3-column matrix)            
#               
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  19/Oct/2020      David Calles
#       Review and documentation        9/Dec/2020       David Calles
***************************************************************************"""
def Calculate_3D_Points(mtx4x3, points2D):   
    # TURN  cartesian coordinates into homogeneus
    data = np.column_stack((points2D[:,1], points2D[:,0]))
    data_ext = np.hstack((data, np.ones((len(data),1))))  
    # MULTIPLY by calibrated transformation amtrix
    points3D = np.matmul(mtx4x3, data_ext.T) 
    # TURN homogeneus into cartesian coordinates      
    points3D2 = (points3D/points3D[3,:]).T[:,0:3]   
    return points3D2

"""***************************************************************************   
# NAME: rigid_transform_3D
# DESCRIPTION: Calculate a transformation from one set of 3D points 
#              to another. As a rotation matrix and translation vector.
#               Taken from: http://nghiaho.com/?page_id=671
#
# PARAMETERS: A: First set of 3D points (3-row matrix)
#             B: Second set of 3D points (3-row matrix)
#                
# RETURNS:    XYZ: xyz data of axis points. (3-column array)
#             rgb: color data of axis points. (3-column array)
#               
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  27/Nov/2020      David Calles
#       Review and documentation        7/Dec/2020       David Calles
***************************************************************************"""
def rigid_transform_3D(A, B):
    # VERIFY input sizes
    assert A.shape == B.shape
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception("matrix A is not 3xN, it is {}x{}"
                        .format(num_rows,num_cols))
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception("matrix B is not 3xN, it is {}x{}"
                        .format(num_rows,num_cols))

    # FIND mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ENSURE centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # SUBSTRACT mean
    Am = A - centroid_A
    Bm = B - centroid_B
    H = Am @ np.transpose(Bm)

    # FIND rotation transform
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # EVALUATE special reflection case
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = Vt.T @ U.T
        
    # FIND translation vector
    t = -R@centroid_A + centroid_B

    return R, t

"""***************************************************************************   
# NAME: Get_Line
# DESCRIPTION: Approximate a set of points to a line by adjusting its X 
#                values (column). Just the values inside the chessboard
#                are taken.
#
# PARAMETERS:  subpxs: Coordinates in image of the points to approximate.
#                        (2-column matrix).
#              corners: Corners of chessboard to calcualte the limits.
#              percent: Proportion of amount of fitting to be performed to
#                        points. (0-1).
#              plot: Plot result of line fitting.
#                
#                
# RETURNS:    fitted: adjusted points. (2-column matrix).
#               
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  27/Nov/2020      David Calles
#       Review and documentation        8/Dec/2020       David Calles
***************************************************************************"""
def Get_Line(subpxs, corners, percent, plot=True):
    
    # GET reference points of max and minimum corners
    sup_max = max((corners[0,1], corners[1,1]))  
    bot_max = max((corners[2,1], corners[3,1]))     
    
    # GET useable points
    usable_subpxs = []
    for n in range(len(subpxs)):
        if(subpxs[n,0]>=sup_max and subpxs[n,0]<=bot_max):
            usable_subpxs.append(subpxs[n,:])
    usable_subpxs = np.array(usable_subpxs)
    
    # FIT line to points
    m, b = np.polyfit(usable_subpxs[:,0], usable_subpxs[:,1], 1)
    x_data = usable_subpxs[:,0]
    y_data = usable_subpxs[:,0]*m + b
    diff = y_data - usable_subpxs[:,1]
    
    # SET amount of fitting
    filtered = y_data + ((1-percent)*diff)
    
    # PLOT if enabled
    if plot:
        area = np.pi*2
        fig = plt.figure()
        axes = plt.gca()
        axes.invert_yaxis()
        plt.scatter(usable_subpxs[:,1], x_data, s=area, c='r')
        plt.scatter(filtered, x_data, s=area, c='g')
        plt.plot(y_data, x_data) 
        plt.title('Fitted line')
        mean=np.average(filtered)
        axes.set_xlim([mean-100,mean+100])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid()
        plt.show()
       
    return np.column_stack((x_data, filtered))


"""***************************************************************************   
# NAME: order_points
# DESCRIPTION: Order 4 points clockwise. Taken from: Course "Procesamiento
                de imágenes y visión. Francisco Calderón".
#
# PARAMETERS:  pts: Points to be ordered.               
#                
# RETURNS:    rect: Ordered points.
#               
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  27/Sep/2020      Francisco Calderón
#       Review and documentation        8/Dec/2020       David Calles
***************************************************************************"""
def order_points(pts):
    
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect

"""***************************************************************************   
# NAME: Create overdefined matrix
# DESCRIPTION:  Create matrix of overdefined system. As a 11 column system 
#               with corresponding solution vector. In the "A*x=b" system 
#               the A matrix and the b vector are returned. The overdefined
#               system is calculated from a set of 2D-3D correspondences.
#               
# PARAMETERS:   p2D: 2D-points (3-column matrix)
#               p3D: 3D-points (3-column matrix)
#               verbose: Show additional information. Default=True
#                              
# RETURNS:      mtx_A1: rank-11 A matrix. (11-column matrix).
#               vct_1: solution vector b.               
#               
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  10/Nov/2020      David Calles
#       Review and documentation        8/Dec/2020       David Calles
***************************************************************************"""
def Get_Overdefined_Mtx(p2D, p3D, verbose=False):
    # INITIATE matrix and vector
    vct_0 = np.zeros((len(p2D)*3)) # ZERO RESULT     
    mtx_A = np.zeros((1, 12)) # BIG MTX
    
    # FILL matrix        
    for m in range(len(p2D)):
                                
        row1 = np.array([ p2D[m,0], p2D[m,1], 1, 0, 0, 0, 0, 0, 0, 
                         (-p2D[m,0]*p3D[m,0]),
                         (-p2D[m,1]*p3D[m,0]),
                         (-p3D[m,0]) ]) 
        row2 = np.array([ 0, 0, 0, p2D[m,0], p2D[m,1], 1, 0, 0, 0,
                         (-p2D[m,0]*p3D[m,1]),
                         (-p2D[m,1]*p3D[m,1]),
                         (-p3D[m,1]) ])
        row3 = np.array([ 0, 0, 0, 0, 0, 0, p2D[m,0], p2D[m,1], 1, 
                         (-p2D[m,0]*p3D[m,2]),
                         (-p2D[m,1]*p3D[m,2]),
                         (-p3D[m,2]) ])
        # CONCATENATE 3 rows in system    
        mtx_A = np.row_stack((mtx_A, row1, row2, row3))           
        mtx_A = mtx_A[1:,:] 
        
        # ---------- Last coefficient = 1 -----------------------------------
        # TURN rank-12 matrix into rank-11 matrix
        mtx_A1 = mtx_A[:,0:-1]
        vct_1 = -mtx_A[:,-1]
    if verbose:
        print(" Dimensions of original Mtx and 0_Vect: ",
                  mtx_A.shape, vct_0.shape)
    return mtx_A1, vct_1

"""***************************************************************************   
# NAME: Get_arbitrary_points_from_line
# DESCRIPTION:  Get a random "amount" of points in a line. Line is defined by
#               two points. All values are integer.
#               
# PARAMETERS:   p1, p2: points to which define the line.
#               amount: amount of points to get. Points are obtained without
#                       repetition. Value must be lower than the total x 
#                       values in line. 
#               verbose: Show additional information. Default=True.
#                              
# RETURNS:      points: "amount" number of points from defined line without
#               duplicates. (2-colum array) [row,column].              
#               
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  12/Oct/2020      David Calles
#       Review and documentation        8/Dec/2020       David Calles
***************************************************************************"""
def Get_arbitrary_points_from_line(p1, p2, amount, verbose=True):

    # CALCULATE the coefficients. This line answers the initial question. 
    x = [p1[0], p2[0]]
    y = [p1[1], p2[1]]
    coeffs = np.polyfit(x, y, 1)  
    # CALCULATE line equation
    polynomial = np.poly1d(coeffs)    
    # GET random x values to be evaluated 
    x_vals = random.sample(range(round(min(x)+1),round(max(x))), amount)
    x_vals = np.array(x_vals)   
    y_vals = polynomial(x_vals)
    points = np.column_stack([x_vals, y_vals])
    if verbose:
        print("Arbitrary points: \n", points)
    return points

"""***************************************************************************   
# NAME: Point4_3points_crossratio
# DESCRIPTION:  Calculate a fourth point from 3 points and its crossratio.
#               It is assumed that these four points form a pencil of lines
#               between them. 
#               
# PARAMETERS:   p1,p3,p4: Three points from the pencil of lines.
#               cr: Cross-ratio of the pencil of lines
#                              
# RETURNS:      p2: Missing point from pencil of lines.              
#               
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  5/Oct/2020       David Calles
#       Review and documentation        8/Dec/2020       David Calles
***************************************************************************"""
def Point4_3points_crossratio (p1, p3, p4, cr):
    
    lam2 = np.linalg.norm(p3 - p1) # AB
    lam3 = np.linalg.norm(p4 - p1) # AD
    
    # SOLVE for lambda1 from crossratio definition
    lam1 = (((cr-1)*lam2*lam3)/((cr*lam3)-lam2)) #AC
    # GET C from line aproximation using other 2 points
    
    C_A = np.linalg.norm(p3 - p1) #DA norm
    p2 = (lam1*((p3 - p1)/C_A)) + p1
    
    return p2

"""***************************************************************************   
# NAME: Crossratio_4points
# DESCRIPTION:  Calculate the crossratio of a pencil of lines given 4 points.
#               
# PARAMETERS:   p1,p2,p3,p4: 4 points defining the pencil of lines.
#                              
# RETURNS:      cr: Cross-ratio of the pencil of lines.              
#               
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  5/Oct/2020       David Calles
#       Review and documentation        8/Dec/2020       David Calles
***************************************************************************"""
def Crossratio_4points (p1, p2, p3, p4):
    
    lam1 = np.linalg.norm(p2 - p1) # AB
    lam2 = np.linalg.norm(p3 - p1) # AC
    lam3 = np.linalg.norm(p4 - p1) # AD
    # CALCULATE cross-ratio
    cr = ((lam2*(lam3-lam1))/(lam3*(lam2-lam1)))
    
    return cr

"""***************************************************************************   
# NAME: Point projection
# DESCRIPTION:  Calculates corssratio from first 4 points and then 
#               projects missing point. Uses Crossratio_4points() and
#               Point4_3points_crossratio(), already defined functions.
#               
# PARAMETERS:   p1,p2,p3,p4: Points to get crossratio. Can be N-dimensional.
#               p1_obj, p3_obj, p4_obj: Points to get projection. Can be N-
#                                       dimensional.
#               validate: Perform validation of the obtained cross-ratio and
#                           "verbose" it.
#                              
# RETURNS:      p2_obj: projected point in the [p1_obj, p2_obj, p3_obj,
#                       p4_obj] pencil of lines.
#               cr: crossratio of new pencil of lines. (must be the same for
#                   for both pencils).            
#               
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  12/Oct/2020      David Calles
#       Review and documentation        8/Dec/2020       David Calles
***************************************************************************"""
def Point_projection (p1, p2, p3, p4, p1_obj, p3_obj, p4_obj, validate):
    
    cr = Crossratio_4points (p1, p2, p3, p4)
    p2_obj = Point4_3points_crossratio (p1_obj, p3_obj, p4_obj, cr)
    
    if(validate):
        cr_v =  Crossratio_4points(p1_obj, p2_obj, p3_obj, p4_obj)
        print ("CR Validation: ", round(abs(cr/cr_v),5))
    
    return p2_obj, cr

"""***************************************************************************   
# NAME: Plot_Plane
# DESCRIPTION:  Plot a plane from 3 points using pyplot. In addition, plot 
#               given points as a scatter in the same graph. 
#               
# PARAMETERS:   p1,p2,p3: Points defining the plane.
#               xdata,ydata,zdata: Data to create the scatter plot.
#               color: RGB data of the data to be scattered.
#               verbose: Show calculated equation of plane. Default=False.
#                              
# RETURNS:      (1). And creates the resulting plot.              
#               
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  4/Oct/2020       David Calles
#       Review and documentation        8/Dec/2020       David Calles
***************************************************************************"""
def Plot_Plane (p1, p2, p3, xdata, ydata, zdata, color, verbose=False):
    
    # CALCULATE two vector in the plane
    v1 = p3 - p1
    v2 = p2 - p1
    
    # CALCULATE the cross product of vectors. (Normal vector to the plane).
    cp = np.cross(v1, v2)
    a, b, c = cp
    
    # EVALUATE a * x3 + b * y3 + c * z3. (Which equals)
    d = np.dot(cp, p3)
    
    if (verbose):
       print('The equation is {0}x + {1}y + {2}z = {3}'.format(a, b, c, d))
    
    # CREATE points for plane
    X, Y = np.meshgrid(np.linspace(-50,500),np.linspace(-50,500))
    Z = (d - a * X - b * Y) / c
    
    plt3d = plt.figure(6).gca(projection='3d') 
    ax = plt.gca()
    # PLOT plane   
    plt3d.plot_surface(X, Y, Z, alpha=0.5)
    # PLOT points as scatter
    ax.scatter3D(xdata, ydata, zdata, color=color)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.view_init(elev=90, azim=-90)
    plt.ion()
    #plt.show()
    
    return 1

"""***************************************************************************   
# NAME: Plot_Just_Plane
# DESCRIPTION:  Plot a plane from 3 points using pyplot.  
#               
# PARAMETERS:   p1,p2,p3: Points defining the plane.
#               verbose: Show calculated equation of plane. Default=False.
#                              
# RETURNS:      (1). And creates the resulting plot.              
#               
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  4/Oct/2020       David Calles
#       Review and documentation        8/Dec/2020       David Calles
***************************************************************************"""
def Plot_Just_Plane (p1, p2, p3, verbose=False):
    
    # CALCULATE two vector in the plane
    v1 = p3 - p1
    v2 = p2 - p1
    
    # CALCULATE the cross product of vectors. (Normal vector to the plane).
    cp = np.cross(v1, v2)
    a, b, c = cp
    
    # EVALUATE a * x3 + b * y3 + c * z3. (Which equals)
    d = np.dot(cp, p3)
    
    if (verbose):
       print('The equation is {0}x + {1}y + {2}z = {3}'.format(a, b, c, d))
    
    # CREATE points for plane
    X, Y = np.meshgrid(np.linspace(-50,500),np.linspace(-50,500))
    Z = (d - a * X - b * Y) / c
    
    plt3d = plt.figure(6).gca(projection='3d') 
    ax = plt.gca()
    # PLOT plane   
    plt3d.plot_surface(X, Y, Z, alpha=0.5)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.view_init(elev=90, azim=-90)
    ax.set_zlim(-20, 20)
    plt.ion()
    plt.show()   
    
    return 1

"""***************************************************************************   
# NAME: Draw_Quadrangle
# DESCRIPTION: Draw the complete quadrangle of a trapezoid in an image. This 
#               function assumes the complete quadrangle extends to the right 
#               of the image.  
#               
# PARAMETERS:  trapezoid: 4 points defining the trapezoid. (2-column matrix) 
#              G,F: Coordinates of the points in the complete quadrangle.
#              imgBorders: Image to draw the complete quadrangle in. Image
#                           will be modified.
#              thickness: Thickness of lines. Default=thickness_show.
#                   
# RETURNS:     quadrangle_img: Image with complete quandrangle in it.               
#               
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  4/Oct/2020       David Calles
#       Review and documentation        8/Dec/2020       David Calles
***************************************************************************"""
def Draw_Quadrangle (trapezoid, G, F, imgBorders, thickness=thickness_show):
    # GET new sizes of resulting image.
    if G[0] < 0:
        G_modifier_w = imgBorders.shape[1] - G[0]
    else: 
        G_modifier_w = G[0]
        
    if F[1] < 0:
        F_modifier_h = imgBorders.shape[0] - F[1]
    else: 
        F_modifier_h = F[1]
        
    # SET canvas size
    img_q_w = round(max([imgBorders.shape[1], G_modifier_w]))+100
    img_q_h = round(max([imgBorders.shape[0], F_modifier_h]))+100
    quadrangle_img = np.zeros((img_q_h, img_q_w, 3), dtype=np.uint8) 
        
    vshift = quadrangle_img.shape[0]-imgBorders.shape[0]
    
    # PUT original image in canvas
    quadrangle_img[vshift:,:imgBorders.shape[1],:] = imgBorders
        
    # DRAW G value
    int_img_G = np.int32([
        (trapezoid[0][0], trapezoid[0][1]+vshift),
        (trapezoid[1][0], trapezoid[1][1]+vshift),
        (G[0], G[1]+vshift) ])
    cv2.polylines(quadrangle_img, [int_img_G], True,
                  (255, 0, 0), thickness, line_type)
        
    # DRAW F value
    int_img_F = np.int32([
        (trapezoid[2][0], trapezoid[2][1]+vshift),
        (trapezoid[1][0], trapezoid[1][1]+vshift),
        (F[0], F[1]+vshift) ])
    cv2.polylines(quadrangle_img, [int_img_F], True,
                  (255, 0, 0), thickness, line_type)  

    # DRAW all points 
    # From trapezoid
    int_trapezoid = np.int32(trapezoid)
    for j in range(len(trapezoid)):
        cv2.circle(quadrangle_img,
                   (int_trapezoid[j][0], int_trapezoid[j][1]+vshift),
                   8,(255,255,255),-1)
    # From quadrangle    
    cv2.circle(quadrangle_img,
               (int_img_G[2][0], int_img_G[2][1]),
               8,(255,255,255),-1)
    cv2.circle(quadrangle_img,
               (int_img_F[2][0], int_img_F[2][1]),
               8,(255,255,255),-1)
    
    return quadrangle_img

"""***************************************************************************   
# NAME: line_intersection
# DESCRIPTION:  Compute the intersection of two lines. Lines must be 2D. 
#               If lines are of greater dimension, only first 2 dimensions
#               will be considered.
#               
# PARAMETERS:   line1,line2: Definitions of lines are made using 2 points for
#                           each line. Each row is a point.
#                              
# RETURNS:      intersect: intersecting point from lines. If lines do not
#                           intersect, return=False,False.              
#               
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  4/Oct/2020       David Calles
#       Review and documentation        8/Dec/2020       David Calles
***************************************************************************"""
def line_intersection(line1, line2):
    # CALCULATE distances in X and Y.
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    
    # DEFINE determinant
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]
    
    # VERIFY intersection exists
    div = det(xdiff, ydiff)
    if div == 0:
       return False, False
    # CALCULATE intersections
    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    
    return np.array([x, y])

"""***************************************************************************   
# NAME: Double_Lagrange_Aprox_Image
# DESCRIPTION:  Fit parabola into laser detection to improve resolution
#               in laser segmentation from pixel to subpix.
#               Parabola is fitted on the assumption that one single 
#               masurement must be taken per row.
#               The data used for the parabola approximation is x,y
#               being x: the column (3 are taken, max +-1 cols)
#                     y: original brightness in single channel.
#               Current implementated using Lagrange polinomials.        
#               
# PARAMETERS:   img: Gray scale thresholded image
#               idx1: Max values for each row. Left values.
#               idx2: Max values for each row. Right values.
#               two: Apply double parabola approximation.
#               verbose: Show additional information.
#               
# RETURNS:      subpxs: Estimated center of laser using double parabola 
#                       approximation.
#               
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  20/Sep/2020      David Calles
#       Review and documentation        9/Dec/2020       David Calles
***************************************************************************"""
def Double_Lagrange_Aprox_Image (img, idx1, idx2, two, verbose):
    fit = []
    count = 0
    count2 = 0
    for i in range(idx1.shape[0]):
        
        # -------------------------------------------------------------------   
        # FIRST PARABOLA: ALWAYS 
        # -------------------------------------------------------------------
        row1 = round(idx1[i,0])
        col1 = round(idx1[i,1])
        # Column numbers
        x_points = np.array([(idx1[i,1]-1), idx1[i,1], (idx1[i,1]+1)]) 
        # Respective values
        y_points = np.array([img[row1,col1-1],
                             img[row1,col1],
                             img[row1,col1+1]])
        # OBTAIN Lagrange Polynomial and coefficients
        poly = lagrange(x_points, y_points)
        coeffs = Polynomial(poly).coef
        if (len(coeffs) >= 2):
            max_val_x = -coeffs[1]/(2*coeffs[0])
            count = count+1
        else:
            max_val_x = idx1[i,1]
        # -------------------------------------------------------------------   
        # SECOND PARABOLA: If right and left maximums are different 
        # -------------------------------------------------------------------
        if((two) and (idx1[i,1] != idx2[i,1])): 
            row2 = round(idx2[i,0])
            col2 = round(idx2[i,1])
            #Care with last column
            if(col2 >= (img.shape[1]-1)):
                col2 = img.shape[1]-2
            # Column numbers
            x_points2 = np.array([(idx2[i,1]-1), idx2[i,1], (idx2[i,1]+1)]) 
            # Respective values
            y_points2 = np.array([img[row2,col2-1],
                                 img[row2,col2],
                                 img[row2,col2+1]])
            # OBTAIN Lagrange Polynomial and coefficients
            poly2 = lagrange(x_points2, y_points2)
            coeffs2 = Polynomial(poly2).coef
            if (len(coeffs2) >= 2):
                max_val_x2 = -coeffs2[1]/(2*coeffs2[0])
                count2 = count2+1
            else:
                max_val_x2 = idx2[i,1]
            max_val_mean = (max_val_x + max_val_x2)/2
            fit.append(max_val_mean) # Mean
        else:
            fit.append(max_val_x)
        # -------------------------------------------------------------------   
        # GRAPHS: IF ENABLED
        # -------------------------------------------------------------------
        if (SUBPIX_GRAPH and (ROW_TO_GRAPH == i)): 
            # -----------------PLOT ORIGINAL DATA
            plt.figure()
            x_original = np.arange(max_val_x-30, max_val_x+30)
            y_original = img[row1,col1-30:col1+30] 
            plt.plot(x_original, y_original, 'b',  linewidth=5) # Original data
            
            # -----------------PLOT FIRST PARABOLA DATA and MAX POINT
            # GRAPH Parabola aproximation
            x_parabola = x_original
            y_parabola = poly(x_parabola)           
            plt.plot(x_parabola, y_parabola, 'r') # Fitted Parabola
            plt.scatter(x_points,y_points)        # Points Used
         
            # GET Max value in Y to graph if Lagrange aprox was made
            if (len(coeffs) >= 3):
                max_val_y = (coeffs[0]*(max_val_x**2)  
                             + (coeffs[1]*(max_val_x))  
                             + (coeffs[2]))
            else:
                max_val_y = 255
            # PLOT max value
            plt.scatter(max_val_x,max_val_y)      # Max value found
            
            if((two) and (idx1[i,1] != idx2[i,1])): 
                # -----------------PLOT SECOND PARABOLA DATA and MAX POINT
                # GRAPH Parabola aproximation
                x_parabola2 = np.arange(max_val_x2-30, max_val_x2+30)
                y_parabola2 = poly2(x_parabola2)           
                plt.plot(x_parabola2, y_parabola2, 'g') # Fitted Parabola
                plt.scatter(x_points2,y_points2)        # Points Used
             
                # GET Max value in Y to graph if Lagrange aprox was made
                if (len(coeffs2) >= 3):
                    max_val_y2 = (coeffs2[0]*(max_val_x2**2)  
                                 + (coeffs2[1]*(max_val_x2))  
                                 + (coeffs2[2]))
                else:
                    max_val_y2 = 255
                # PLOT max value
                plt.scatter(max_val_x2,max_val_y2)  # Max value found
                # -----------------PLOT MEAN MAX VALUE
                max_val_ymean = round((max_val_y2+max_val_y)/2)
                plt.scatter(max_val_mean,max_val_ymean)  # Max value found
                
                # PLOT MAX VAL LINE
                liney = np.arange(0, 255)
                linex = np.ones(len(liney))*max_val_mean
                plt.plot(linex, liney, 'purple')
            else:
                liney = np.arange(0, 255)
                linex = np.ones(len(liney))*max_val_x
                plt.plot(linex, liney, 'purple')
                
            plt.grid()
            plt.title("Parabola aproximation")
            plt.xlabel("Columns")
            plt.ylabel("Brightness")
            axes = plt.gca()
            axes.set_xlim([max_val_x-40, max_val_x+40])
            axes.set_ylim([0, 256])
            plt.show()
      
    # print how many aproximations where made
    if verbose:    
        print ("Succesfull First Lagrange Aproximations = {}/{}".format(
            count,
            idx1.shape[0]))   
        if (two):
            print ("Succesfull Second Lagrange Aproximations = {}/{}".format(
                count2,
                idx2.shape[0])) 
    # Make sure all values are in bounds for all columns       
    subpix_idx = np.clip(np.array(fit),0,rwidth) 
    subpix_idx = subpix_idx.reshape(-1,1) # column array
    
    rows = idx1[:,0].reshape(-1,1)
    return np.concatenate((rows,subpix_idx),axis=1)

"""***************************************************************************   
# NAME: Simple_Image_Correction
# DESCRIPTION: This function handles the distortion correction using the 
#              OpenCv package. It estimates an optimal new camera matrix 
#              from the original calibrate matrix and the resolution values
#              of the images. It then computes the undistorted image.
#               
# PARAMETERS:  img: Original image to be undistorted.          
#               
# RETURNS:     crop_img: Undistorted and croped image (to ROI).              
#               
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  20/Sep/2020      David Calles
#       Review and documentation        9/Dec/2020       David Calles
***************************************************************************"""
def Simple_Image_Correction (img):
    # GET shape of image
    shape = (img.shape[1]+1, img.shape[0]+1)
    # LOAD intrinsic camera matrix and distortion coefficients vector.
    with np.load('CameraCalibration.npz') as file:
        mtx, dist, _, _, _, _ = [file[i] for i in (
            'mtx','dist','rvecs','tvecs', 'newcameramtx', 'mean_error')]
        # ESTIMATE new camera matrix
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            cameraMatrix = mtx,
            distCoeffs = dist,
            imageSize = shape,
            alpha = 0,
            newImgSize = shape
        )
        
        # UNDISTORT image
        out_img = np.zeros(img.shape,np.uint8)
        cv2.undistort(
            src = img,
            cameraMatrix = mtx,
            distCoeffs = dist,
            dst = out_img,
            newCameraMatrix = newcameramtx
        )
        # CROP image to ROI
        x,y,w,h = roi
        crop_img = out_img[y : y+h, x : x+w]
    return crop_img

"""***************************************************************************   
# NAME: Skeletonization
# DESCRIPTION:  Using skimage library, a skeletonization is performed, 
#               the zhan-suen fast parallel algorithm is used here. Note that
#               no good performance was showed with this function.             
#               
# PARAMETERS:   image: Thresholded grayscale image.              
#               
# RETURNS:      skeletonb: skeleton of image              
#               
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  24/Aug/2020      David Calles
#       Review and documentation        9/Dec/2020       David Calles
***************************************************************************"""
def Skeletonization (image) : 
    # TURN image into bolean
    img_bool = image > 127
    # PERFORM skeletonization
    skeleton = skeletonize(img_bool)
    skeletonb = np.uint8 (np.array(skeleton) * 255)
    return skeletonb 
    
"""***************************************************************************   
# NAME: LASER SEGMENTATION FUNCTION
# DESCRIPTION:  Separates the laser pixels in an image by taking its red 
#               component and then applying one of many possible thresholds.
#               Then it performs some morphological operations. Other middle
#               step features can be enabled/disabled to improve performance.
#
# PARAMETERS:   imA: Image with laser in it
#               imB: Image without laser
#               thresh: Threshold to be applied. Default=0.15.
#               method: NORMAL_THRESHOLDING=0 or OTSU_THRESHOLDING=1.
#                       Default=0.
#               gauss: Apply gaussian 3x3 filtering before thresholding.
#                       Default=True.
#               strict: Set the HSV filter values to be very strict. It can 
#                       improve performace if the laser in image has a vivid
#                       red color. Default=False.(disabled).
#               hsv_on: Enable/Disable the HSV filtering. If laser color is
#                       not highlighted enough. It should remain disabled.
#                       Default=False.(disabled).
#               contrast_on: Enable/Disable the contrast increase feature.
#                            If the red component of laser is to weak, 
#                            enabling this feature can improve performance.
#                            Default=False.(disabled).
#               plane: Some morphological operations are performed in the
#                       to reduce noice and "try to" merge points of laser.
#                       If this feature is enabled, the operations will 
#                       do a greater effort at merging white points. If the
#                       shape of the object is known to be a plane, then the 
#                       laser is also knwon to be a line and this feature 
#                       will make the algorithm perform better. 
#                       If the illuminated object is other than a plane, this
#                       feature must be turned off. Default=True.(enabled).
#
# RETURNS:      thin_laser_bw: B/W image with max value from left.
#               thin_laser_color: BGR image with max value from left.
#               skeletonA: Skeleton of lasered image (not used).
#               thr_R: thresholded image.
#               indexes_izq: max values of each row from lefto to right.
#               indexes_der: max values of each row from right to left.
#               opened: Image after closing and opening morfology operations.
#               closed: Image after closing morfology operation
#               ci: Image with contrast increased. If enabled.
#                
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  23/Aug/2020       David Calles
#       HSV filter implementation       15/Nov/2020       David Calles
#       Review and documentation        9/Dec/2020        David Calles
***************************************************************************"""
def Laser_segmentation  (imA, imB, thresh=0.15, method=0, gauss=True,
                         strict=False, hsv_on=False, contrast_on=False,
                         plane=True):
    # SET HSV filter values
    if strict:
        lower_redA = np.array([0/2,50,75])
        upper_redA = np.array([35/2,255,255])
        lower_redB = np.array([340/2,50,75])
        upper_redB = np.array([360/2,255,255])
    else:
        lower_redA = np.array([0/2,30,30])
        upper_redA = np.array([32/2,255,255])
        lower_redB = np.array([335/2,30,30])
        upper_redB = np.array([360/2,255,255])
    
    # SET structuring element in hsv filter    
    struct_elem_mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,15))
    # GET red component of images
    red1 = imA[:, :, 2]
    red2 = imB[:, :, 2]
    # SUBSTRACT red components
    red_substract = cv2.absdiff(red1, red2)
    # GET max and min values in image
    biggest_r = np.amax(red_substract)
    smallest_r = np.amin(red_substract)
    # APPLY gaussian filter if enabled
    if(gauss):
        red_substract = cv2.GaussianBlur(red_substract, (3, 3), 0)
    # APPLY threholding 
    if (method):
        # ------------------------ otsus threshholding keeping original value
        _,thr_R = cv2.threshold(
            red_substract,
            smallest_r,
            biggest_r,
            cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
    else:
        # ----------------------- normal threshholding keeping original value
        _,thr_R = cv2.threshold(
            red_substract,
            thresh*biggest_r,
            biggest_r,
            cv2.THRESH_TOZERO)
    # FILTER: Using HSV palette
    if hsv_on:
        hsv = cv2.cvtColor(imA, cv2.COLOR_BGR2HSV)
        # THRESHOLS: HSV image to get only red colors
        red_maskA = cv2.inRange(hsv.copy(), lower_redA, upper_redA) 
        red_maskB = cv2.inRange(hsv.copy(), lower_redB, upper_redB)
        # APPLY: MASKS
        red_mask = cv2.bitwise_or(red_maskA, red_maskB)
        closing_mask = cv2.morphologyEx(red_mask.copy(), cv2.MORPH_CLOSE,
                                        struct_elem_mask)
        opening_mask = cv2.morphologyEx(closing_mask, cv2.MORPH_OPEN,
                                        struct_elem_mask)
        filtered = cv2.bitwise_and(thr_R,thr_R, mask=opening_mask)
    else: 
        filtered = thr_R
    if contrast_on:
        # CONTRAST INCREASE
        ci = cv2.convertScaleAbs(filtered, alpha=1.2, beta=0)
    else: 
        ci = filtered
    # MORPHOLOGICAL OPERATIONS
    if plane:
        elem_size1 = (7,140)
    else:
        elem_size1 = (6,4)
    # CLOSE: attempts to unify the laser line with a vertical rectangle
    struct_elem = cv2.getStructuringElement(cv2.MORPH_RECT, elem_size1) 
    closed = cv2.morphologyEx(ci, cv2.MORPH_CLOSE, struct_elem)
    # OPEN: aids to supress random white points all over the image
    struct_elem2 = cv2.getStructuringElement(cv2.MORPH_RECT, (6,4))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, struct_elem2)
    
    # OPTION 1: HIGHEST VALUE SKELETIZATION
    # --------------------------- initiate black pictures
    thin_laser_bw = np.zeros((rheight,rwidth), np.uint8)
    thin_laser_color = np.zeros((rheight,rwidth,3), np.uint8) 
    #indexes creation
    indexes_izq = []
    indexes_der = []
    #Inverted image
    opened2 = np.flip(opened,1)
    # Thinerization with highest value
    for j in range(rheight):
        # MAX FROM LEFT TO RIGHT
        max_index = np.argmax(opened[j,:])
        # Avoiding rows full of zeros
        if(opened[j,max_index] != 0):
            indexes_izq.append((j, max_index))
            thin_laser_bw.itemset((j, max_index), 255)
            thin_laser_color[j, max_index, :] = imB[j, max_index, :]
        # MAX FROM RIGHT TO LEFT    
        max_index2 = np.argmax(opened2[j,:])
        # Avoiding rows full of zeros
        if(opened2[j,max_index2] != 0):
            corrected_max = rwidth - max_index2 -1 
            indexes_der.append((j, corrected_max))
    # OPTION 2: SKIMAGE
    # --------------------------- initiate black pictures
    skeletonA = Skeletonization(thr_R)
    
    return thin_laser_bw, thin_laser_color, skeletonA, \
             thr_R, indexes_izq, indexes_der, opened, closed, ci



