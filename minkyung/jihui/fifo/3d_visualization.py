import numpy as np
import open3d as o3d

ver = "ver0"
a = np.load("visualization_"+ver+"_output3d.npy")

# initialize pcd
pcd_cw = o3d.geometry.PointCloud()
pcd_sf = o3d.geometry.PointCloud()
pcd_rf = o3d.geometry.PointCloud()

pcd_cw.points = o3d.utility.Vector3dVector(a[:1000,:])
pcd_sf.points = o3d.utility.Vector3dVector(a[1000:2000,:])
pcd_rf.points = o3d.utility.Vector3dVector(a[2000:,:])

pcd_cw.paint_uniform_color([1,0.7,0])
pcd_sf.paint_uniform_color([0,1,0.7])
pcd_rf.paint_uniform_color([0.7,0,1])

o3d.visualization.draw_geometries([pcd_cw,pcd_sf,pcd_rf],
                                  zoom=0.7,
                                  front=[0.5439, -0.2333, -0.8060],
                                  lookat=[2.4615, 2.1331, 1.338],
                                  up=[-0.1781, -0.9708, 0.1608])