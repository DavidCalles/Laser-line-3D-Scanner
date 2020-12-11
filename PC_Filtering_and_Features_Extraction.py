# -*- coding: utf-8 -*-
"""
##############################################################################
#
#   MODULE DESCRIPTION:
#        This module performs various steps of more-restrictive filtering in
#        a point cloud. It also estimates width, depth, height, and volume of 
#        the point cloud using a bounding box and the convex hull of the 
#        point cloud. Here several translations and a scaling of the point
#        are performed. Please note that most of the filtering steps must be
#        tuned depending on the characteristics of each point cloud.
#
#   PRE-REQUISITES: 
#        A point cloud with its respective vertices colors in an .xyz format.
#
#   OUTPUTS:
#        Filtered point cloud
#        Estimated features of the point cloud.
#        
#   USAGE:
#         1. Change "dataname" variable with the direction of the point cloud
#                file.
#         2. Enable different fitlers by changing variables
#                ENABLE_CYLINDER_FILTER,
#                ENABLE_STATISTIC_FILTER,
#                ENABLE_NEIGHBORS_FILTER.
#         3. Tune each filter by changing its corresponding values.
#
#    
#    ACTION:                             DATE:           NAME:
#        First implementation and tests      23/Nov/2020      David Calles
#        Code comments and last review       8/Dic/2020       David Calles           
#
##############################################################################
"""
#----------------------------------------------------------------------------#
# ----------------------- REQUIRED EXTERNAL PACKAGES ------------------------#
#----------------------------------------------------------------------------#

import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull

#----------------------------------------------------------------------------#
# ----------------------- SCALE CORRECTION VARIABLES-------------------------#
#----------------------------------------------------------------------------#

#With Black Ball
X_modifier = 0.72+0.1
Y_modifier = 0.74+0.1
Z_modifier = 0.72+0.1

#----------------------------------------------------------------------------#
# ----------------------- FUNCTION DEFINITIONS ------------------------------#
#----------------------------------------------------------------------------#

"""***************************************************************************   
# NAME: Create_Axis
# DESCRIPTION: Create the point cloud of an axis centered in "center".
#              This aids the visualization of the reference point when 
#              graphing a resultant point cloud. [X,Y,Z] = [B,G,R].
#
# PARAMETERS: center: [x0,y0,z0] coordinates of axis center. Default:[0,0,0].
#                
# RETURNS:    XYZ: xyz data of axis points. (3-column array)
#             rgb: color data of axis points. (3-column array)
#               
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  27/Nov/2020      David Calles
#       Review and documentation        7/Dec/2020       David Calles
***************************************************************************"""
def Create_Axis(center=[0,0,0]):
    # CREATE X axis points
    X = np.linspace(center[0], center[0]+300, 300)
    ones = np.ones(len(X))
    X2 = np.column_stack((X,ones*center[1],ones*center[2]))
    # CREATE X axis color data
    X_rgb = np.ones((len(X),3), dtype=np.uint8)*np.array([255,0,0])
    
    # CREATE Y axis points
    Y = np.linspace(center[1], center[1]+300, 300)
    ones = np.ones(len(Y))
    Y2 = np.column_stack((ones*center[0],Y,ones*center[2]))
    # CREATE Y axis color data
    Y_rgb = np.ones((len(Y),3), dtype=np.uint8)*np.array([0,255,0])
    
    # CREATE Z axis points
    Z = np.linspace(center[2], center[2]+300, 300)
    ones = np.ones(len(Z))
    Z2 = np.column_stack((ones*center[0],ones*center[1],Z))
    # CREATE Z axis color data
    Z_rgb = np.ones((len(Z),3), dtype=np.uint8)*np.array([0,0,255])
    
    # APPEND all axis data
    XYZ = np.vstack((X2,Y2,Z2))
    rgb = np.vstack((X_rgb,Y_rgb,Z_rgb))
    
    return XYZ, rgb

"""***************************************************************************   
# NAME: convex_hull_volume_bis
# DESCRIPTION: Estimate volume of a point cloud from its convex hull. Which
#              is a watertight enclosed mesh.
#
# PARAMETERS: pts: xyz data of the input point cloud. (3-column array).
#                
# RETURNS:    vol: Estimated volume of point cloud. The units are the same
#                  from the point cloud. (scalar).
#               
#       ACTION:                         DATE:       NAME:
#       First implementation and tests  27/Nov/2020      David Calles
#       Review and documentation        7/Dec/2020       David Calles
***************************************************************************"""
def convex_hull_volume_bis(pts):
    # CALCULATE convex hull of point cloud
    ch = ConvexHull(pts)
    
    # ESTIMATE volume
    simplices = np.column_stack((np.repeat(ch.vertices[0], ch.nsimplex),
                                 ch.simplices))
    tets = ch.points[simplices]
    cross = np.cross(tets[:, 1]-tets[:, 3], tets[:, 2]-tets[:, 3])
    tet_vol = np.abs(np.einsum('ij,ij->i', tets[:, 0]-tets[:, 3], cross))/6   
    vol = np.sum(tet_vol)
    
    return vol

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
#       Review and documentation        7/Dec/2020       David Calles
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
        r_limit = radius**2 
        h_limit = np.array(height)    
        r_points = ((points3D[:,0]**2) + (points3D[:,2]**2)) 
        outliers = 0
        
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


#----------------------------------------------------------------------------#
# ----------------------- ALGORITHM SETTINGS --------------------------------#
#----------------------------------------------------------------------------#
# POINT CLOUD FILENAME
# Chili plant  
#dataname = "Sample_Point_Clouds/Point_cloud_Chili.xyz"
            #Tuned values of filter:           
                # For Whole plant:
                #radius=175, height=[-200, 125]
                #nb_points=50, radius=20            
                # For Biomass:
                #radius=175, height=[-200, -55]
                #nb_points=60, radius=10
# Volleyball ball
dataname = "Sample_Point_Clouds/Point_cloud_Ball.xyz"
            #Tuned values of filter:           
               # radius=150, height=[-130, 85]
                #nb_points=100, radius=20 

# ENABLE features
ENABLE_CYLINDER_FILTER = True
ENABLE_STATISTIC_FILTER = False
ENABLE_NEIGHBORS_FILTER = True

# SET cylinder deimensions for filtering (if enabled)
cyl_radius  = 175
cyl_heights = [-200, 125]

# SET statistical filter values (if enabled)
stat_nb_points = 30
stat_std_ratio = 1.0

# SET neighbor-based filter values (if enabled)
neig_nb_points = 50
neig_radius = 20
#----------------------------------------------------------------------------#
# --------------------------- LOAD RAW POINT CLOUD --------------------------#
#----------------------------------------------------------------------------#

# READ .xyz file
point_cloud= np.loadtxt(dataname, skiprows=1)

# LOAD data as OPEN3D object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud[:,:3]) #xyz
pcd.colors = o3d.utility.Vector3dVector(point_cloud[:,3:6]/255) #rgb
o3d.visualization.draw_geometries([pcd])

# CALCULATE initial bounding box
bbox = pcd.get_axis_aligned_bounding_box()

# SHOW raw point cloud and bounding box
o3d.visualization.draw_geometries([pcd, bbox])

# GET point cloud data as numpy arrays
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

#----------------------------------------------------------------------------#
# ---------------------- FILTER POINT CLOUD ---------------------------------#
#----------------------------------------------------------------------------#

# TRANSLATE data to have center in [0,0,0] (Needed for cylinder filtering)
avr = np.average(points,axis=0)
points = points - avr

# FILTER point cloud with a defined axis cylinder across Y axis (if enabled)
if ENABLE_CYLINDER_FILTER:
    points2, colors2 = Filter_3D_Data_Cilinder(points, colors*255, radius=150,
                                               height=[-130, 85],
                                               verbose=True, apply=True,
                                               plot=False)
else:
    points2 = points.copy()
    colors2 = colors.copy()*255

pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(points2)
pcd2.colors = o3d.utility.Vector3dVector(colors2/255)

# FILTER point cloud using statistical outliers removal (if enabled)
if ENABLE_STATISTIC_FILTER:
    cl, ind = pcd2.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.0)
    inlierA, outlierA = display_inlier_outlier(pcd2, ind)
else:
    inlierA = pcd2
    
# FILTER point cloud using neighbors based outliers removal (if enabled)
if ENABLE_NEIGHBORS_FILTER:    
    cl2, ind2 = inlierA.remove_radius_outlier(nb_points=60, radius=10)
    inlier, outlier = display_inlier_outlier(inlierA, ind2)
else:
    inlier = inlierA
  
# SHOW cleaned point cloud (with color) 
o3d.visualization.draw_geometries([inlier])

# CALCULATE bounding box from cleaned point cloud
bbox2 = inlier.get_axis_aligned_bounding_box()
bounding_box_points = np.asarray(bbox2.get_box_points())
# GET height max and min values
y_min = np.min(bounding_box_points[:,1])
y_max = np.max(bounding_box_points[:,1])
# SHOW bounding box
o3d.visualization.draw_geometries([inlier, bbox2])

# GET point cloud as numpy array
points_modif= np.asarray(inlier.points)
colors_modif = np.asarray(inlier.colors)

#----------------------------------------------------------------------------#
# ---------------- TRANSLATE,ROTATE AND SCALE POINT CLOUD -------------------#
#----------------------------------------------------------------------------#

# ESTIMATE center of point cloud from geometry
center_Y = y_min+(abs(y_max-y_min)/2)
avr2 = np.average(points_modif,axis=0)
avr3 = np.array([avr2[0], center_Y, avr2[2]])

# CREATE axis and visualize with point cloud (informative)
axis0, axis0rgb = Create_Axis(np.array([0,0,0]))
centerPC = o3d.geometry.PointCloud()
centerPC.points = o3d.utility.Vector3dVector(axis0)
centerPC.colors = o3d.utility.Vector3dVector(axis0rgb)
o3d.visualization.draw_geometries([inlier, centerPC])

# CORRECT center of point cloud with the estimated one from geometry
points_modif2 = points_modif.copy() - avr3
pcd3A = o3d.geometry.PointCloud()
pcd3A.points = o3d.utility.Vector3dVector(points_modif2)
pcd3A.colors = o3d.utility.Vector3dVector(colors_modif)
o3d.visualization.draw_geometries([pcd3A, centerPC])

# SCALE axes by calibrated value to obtain best results
points_modif3 = np.array(points_modif2)* np.array([X_modifier,
                                                   Y_modifier,
                                                   Z_modifier])

# SHOW scaled point cloud with axis
pcd3 = o3d.geometry.PointCloud()
pcd3.points = o3d.utility.Vector3dVector(points_modif3)
pcd3.colors = o3d.utility.Vector3dVector(colors_modif)
o3d.visualization.draw_geometries([pcd3, centerPC])

# GET filtered and scaled point cloud as numpy arrays
points_axis_change= np.asarray(pcd3.points)
colors_axis_change = np.asarray(pcd3.colors)

# CHANGE axis to have Z as height and X,Y as horizontal plane
points_axis_change = np.column_stack((points_axis_change[:,0], 
                                      points_axis_change[:,2],
                                      points_axis_change[:,1]))
pcd4 = o3d.geometry.PointCloud()
pcd4.points = o3d.utility.Vector3dVector(points_axis_change)
pcd4.colors = o3d.utility.Vector3dVector(colors_axis_change)
# SHOW processed point cloud
o3d.visualization.draw_geometries([pcd4])

#----------------------------------------------------------------------------#
# ---------------------- ESTIMATE POINT CLOUD'S FEATURES --------------------#
#----------------------------------------------------------------------------#

# CALCULATE usable bounding box of processed point cloud
bbox4 = pcd4.get_axis_aligned_bounding_box()
bounding_box_points4 = np.asarray(bbox4.get_box_points())

# CALCULATE convex hull
hull4, _ = pcd4.compute_convex_hull()

# SHOW convex hull with bounding box and point cloud as sets of lines
hull_ls4 = o3d.geometry.LineSet.create_from_triangle_mesh(hull4)
hull_ls4.paint_uniform_color((0, 1, 0))
o3d.visualization.draw_geometries([pcd4, bbox4, hull_ls4])

# GET height, width and depth from bounding box
x_minf = np.min(bounding_box_points4[:,0])
x_maxf = np.max(bounding_box_points4[:,0])
y_minf = np.min(bounding_box_points4[:,1])
y_maxf = np.max(bounding_box_points4[:,1])
z_minf = np.min(bounding_box_points4[:,2])
z_maxf = np.max(bounding_box_points4[:,2])
widthf = x_maxf - x_minf
depthf = y_maxf - y_minf
heightf = z_maxf - z_minf

print ("Width, Depth, Height: \n{},\n{},\n{}".format(widthf,
                                                     depthf,
                                                     heightf))

pc_volumef = convex_hull_volume_bis(points_axis_change)
print ("Volume of PC: \n{}".format(pc_volumef))
