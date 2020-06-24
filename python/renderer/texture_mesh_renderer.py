import os, torch

# Util function for loading meshes
from pytorch3d.io import load_obj, load_ply, load_objs_as_meshes

# Data structures and functions for rendering
from pytorch3d.structures import Meshes, Textures
from pytorch3d.ops import GraphConv, sample_points_from_meshes, vert_align
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    PointLights, HardPhongShader,
    RasterizationSettings,
    MeshRenderer, MeshRasterizer,
    TexturedSoftPhongShader, BlendParams
)
import numpy as np
import matplotlib.pyplot as plt

class TextureMeshRenderer(object):
    def __init__(self, device = 'cuda:0', dist = 0.5, batch_size = 5):
        torch.cuda.set_device(device)
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.elev = torch.linspace(0,180,self.batch_size)
        self.azim = torch.linspace(-180,180,self.batch_size)
        self.R, self.T = look_at_view_transform(dist=dist, elev=self.elev, azim= self.azim) 
        self.cameras = OpenGLPerspectiveCameras(device= self.device, R= self.R, T = self.T)
        self.light_location = [0.0,0.0,-3.0]
        self.raster_settings = RasterizationSettings(
            image_size=512, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )
        self.lights = PointLights(device=self.device)

    def load_obj_file(self,obj_file):
        mesh = load_objs_as_meshes([obj_file],self.device)
        self.meshes = mesh.extend(self.batch_size)
    def render(self,light_location):
        self.lights.location = torch.tensor([0.0,0.0,-3.0],self.device)
        renderer = MeshRenderer (
            rasterizer = MeshRasterizer (
            cameras = self.cameras,
            raster_settings = self.raster_settings
            ),
            shader= TexturedSoftPhongShader(
                device= self.device,
                cameras= self.cameras,
                lights= self.lights
            )
        ) 


        images = self.phong_renderer(self.meshes, cameras=self.cameras)
        plt.figure(figsize=(10, 10))
        plt.imshow(images[1, ..., :3].cpu().numpy())
        plt.grid("off");
        plt.axis("off");
        plt.show()


    def get_look_at_view_transform(self):
        return self.R, self.T
    def get_light_location(selft):
        return self.light_location
    def change_light(self, light_location):
        self.set_phong_render(light_location=light_location)

