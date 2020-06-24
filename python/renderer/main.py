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


if __name__ == '__main__':
    cfg = Config(config_file)
    renderer = TextureMeshRenderer('cuda:0',cfg.distance,cfg.batch_size)
    #renderer.set_phong_render([0.0,0.0,-3.0])
    renderer.load_obj_file(obj_file)
    renderer.render([0.0,0.0,-3.0])

"""
# Setup
device = torch.device("cuda:0")
torch.cuda.set_device(device)

mesh = load_objs_as_meshes([obj_file],device)
textures = mesh.textures.maps_padded()
lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
lights.location = torch.tensor([0.0, 0.0, +1.0], device=device)[None]
R, T = look_at_view_transform(2.7, 0, 180) 
cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)

# Define the settings for rasterization and shading. Here we set the output image to be of size
# 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
# and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
# the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
# explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
# the difference between naive and coarse-to-fine rasterization. 
raster_settings = RasterizationSettings(
    image_size=512, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
)
# Create a phong renderer by composing a rasterizer and a shader. The textured phong shader will 
# interpolate the texture uv coordinates for each vertex, sample from a texture image and 
# apply the Phong lighting model
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=TexturedSoftPhongShader(
        device=device, 
        cameras=cameras,
        lights=lights
    )
)

batch_size = 4
meshes = mesh.extend(batch_size)
# Get a batch of viewing angles. 
elev = torch.linspace(0, 180, batch_size)
azim = torch.linspace(-180, 180, batch_size)
R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
lights.location = torch.tensor([[0.0, 0.0, -3.0]], device=device)
images = renderer(meshes, cameras=cameras, lights=lights)

plt.figure(figsize=(10, 10))
plt.imshow(images[2][0, ..., :3].cpu().numpy())
plt.grid("off");
plt.axis("off");
plt.show()
#images = renderer(mesh, lights=lights)
#plt.figure(figsize=(10, 10))
#plt.imshow(images[0, ..., :3].cpu().numpy())
#plt.grid("off");
#plt.axis("off");
#plt.show()
"""