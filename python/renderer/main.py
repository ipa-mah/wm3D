import os
import cv2
import torch
import argparse
import torch.nn as nn
import numpy as np
from common import Config
from texture_mesh_renderer import TextureMeshRenderer
import matplotlib.pyplot as plt
from skimage.io import imread

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes

# Data structures and functions for rendering
from pytorch3d.structures import Meshes, Textures
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    TexturedSoftPhongShader
)

"""
config_file = '/home/hah/wm3D/python/config/rendering_config.yaml'
if __name__ == "__main__":
    config = Config(config_file)
    view_control = ViewControl()
    intrins = np.eye(3,dtype=np.float32)
    intrins[0,0] = config.fx
    intrins[1,1] = config.fy
    intrins[0,2] = config.cx
    intrins[1,2] = config.cy
    view_control.set_config(config.horizontal_views,config.vertical_views,intrins,config.distance)
    print(config.horizontal_views)
    view_control.generate_lool_at_matrix()
"""
obj_file = '/home/hah/wm3D/python/data/cow_mesh/cow.obj'
config_file = '/home/hah/wm3D/python/config/rendering_config.yaml'
output_path = '/home/hah/wm3D/data/'

if __name__ == '__main__':
    cfg = Config(config_file)
    renderer = TextureMeshRenderer('cuda:0')
    #renderer.set_phong_render([0.0,0.0,-3.0])
    renderer.load_obj_file(obj_file)
    
    #renderer.render_with_batch_size(cfg.batch_size,cfg.distance,[0.0,0.0,-3.0],output_path)
    elev = 0
    azim = 180

#    renderer.render(cfg.distance,[0.0,0.0,-3.0],azim,elev,output_path+"render-image.png")
    renderer.render_with_different_azim_elev_size(cfg.distance,
                                        cfg.vertical_views,
                                        cfg.horizontal_views,
                                        [0.0,0.0,-3.0],output_path)
    renderer.saveExtrinsics(output_path+"world_to_cam_poses.yaml")
