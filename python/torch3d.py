import os
import torch
import matplotlib.pyplot as plt
from skimage.io import imread

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

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

# add path for demo utils functions 
import sys
import os
sys.path.append(os.path.abspath(''))
# Setup
device = torch.device("cuda:0")
torch.cuda.set_device(device)
obj_file = '/home/hah/wm3D/python/data/cow_mesh/cow.obj'

mesh = load_objs_as_meshes([obj_file], device=device)
raster_settings = RasterizationSettings(
    image_size=512, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
)
batch_size = 6
meshes = mesh.extend(batch_size )
# Get a batch of viewing angles. 
elev = torch.linspace(0, 180, batch_size)
azim = torch.linspace(-180, 180, batch_size)
R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
#lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
lights = PointLights(device=device)
lights.location = torch.tensor([0.0, 0.0, -3.0], device=device)

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

images = renderer(meshes, cameras=cameras, lights=lights)
plt.figure(figsize=(10, 10))
plt.imshow(images[3, ..., :3].cpu().numpy())
plt.grid("off");
plt.axis("off");
plt.show()