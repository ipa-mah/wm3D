import os, torch
import cv2
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

class TextureMeshRenderer(object):
    def __init__(self, device = 'cuda:0'):
        torch.cuda.set_device(device)
        self.batch_size = 0
        self.device = torch.device(device)
        self.R = torch.empty(1,3,3)
        self.T = torch.empty(1,3)
        self.cameras = []
        self.light_location = [0.0,0.0,0]
        self.raster_settings = RasterizationSettings(
            image_size=512, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )
    # load texture mesh 
    def load_obj_file(self,obj_file):
        self.meshes = load_objs_as_meshes([obj_file],self.device)
    def render_with_batch_size(self,batch_size,dist,light_location,output_path):
        self.meshes = self.meshes.extend(batch_size)
        self.batch_size = batch_size
        elev = torch.linspace(0,180,batch_size)
        azim = torch.linspace(-180,180,batch_size)
        self.R, self.T = look_at_view_transform(dist=dist, elev=elev, azim= azim) 
        self.cameras = OpenGLPerspectiveCameras(device= self.device, R= self.R, T = self.T)
        #set light locatioin
        self.light_location = light_location
        lights = PointLights(device=self.device,location=[self.light_location])        
        # call pytorch3d mesh renderer with shong shader
        renderer = MeshRenderer (
            rasterizer = MeshRasterizer (
            cameras = self.cameras,
            raster_settings = self.raster_settings
            ),
            shader= TexturedSoftPhongShader(
                device= self.device,
                cameras= self.cameras,
                lights= lights
            )
        ) 
        images = renderer(self.meshes, cameras=self.cameras, lights= lights)
        for i in range(self.batch_size) :
            img = images[i, ..., :3].cpu().numpy()*255
            img = img.astype('uint8')
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            cv2.imwrite(output_path + 'render-image-'+ str(i)+'.png',img)
    def render(self,dist,light_location,azim, elev,image_file):
        self.R, self.T = look_at_view_transform(dist=dist,elev=elev,azim=azim)
        self.cameras = OpenGLPerspectiveCameras(device=self.device, R=self.R, T= self.T)
        self.light_location = light_location
        lights = PointLights(device=self.device,location=[self.light_location])
        # call pytorch3d mesh renderer with shong shader
        renderer = MeshRenderer (
            rasterizer = MeshRasterizer (
            cameras = self.cameras,
            raster_settings = self.raster_settings
            ),
            shader= TexturedSoftPhongShader(
                device= self.device,
                cameras= self.cameras,
                lights= lights
            )
        ) 
        images = renderer(self.meshes, cameras=self.cameras, lights= lights)
        img = images[0, ..., :3].cpu().numpy()*255
        img = img.astype('uint8')
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        cv2.imwrite(image_file,img)

    def render_with_different_azim_elev_size(self,dist,elev_size,azim_size,light_location,output_path):
        azims = torch.linspace(-180,180,azim_size)
        elevs = torch.linspace(0,180,elev_size)
        self.light_location = light_location
        lights = PointLights(device=self.device,location=[self.light_location])
        index = 0
        tensor_r = torch.empty(1,3,3)
        for elev in elevs :
            for azim in azims :
                R, T = look_at_view_transform(dist=dist,elev=elev,azim=azim)
                self.cameras = OpenGLPerspectiveCameras(device=self.device, R=R, T= T)
                renderer = MeshRenderer (
                    rasterizer = MeshRasterizer (
                    cameras = self.cameras,
                    raster_settings = self.raster_settings
                    ),
                    shader= TexturedSoftPhongShader(
                    device= self.device,
                    cameras= self.cameras,
                    lights= lights
                    )
                )
                images = renderer(self.meshes, cameras=self.cameras, lights= lights)   
                img = images[0, ..., :3].cpu().numpy()*255     
                img = img.astype('uint8')
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                cv2.imwrite(output_path + 'render-image-'+ str(index)+'.png',img)
                index += 1  
                self.R = torch.cat((self.R,R))
                self.T = torch.cat((self.T,T))

    def saveExtrinsics(self,file_name):
        for i in range(self.R.shape[0]):
            print(self.R[i])
            print(self.T[i])

    def get_look_at_view_transform(self):
        return self.R, self.T
    def get_light_location(selft):
        return self.light_location
    

