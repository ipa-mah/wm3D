#!/usr/bin/env python3
import os
import yaml
import numpy as np
from easydict import EasyDict 

class Config:
    def __init__(self,config_file=''):
        self.fx = 0
        self.fy = 0
        self.cx = 0
        self.cy = 0
        self.image_height = 0
        self.image_width = 0
        self.horizontal_views = 5
        self.vertical_views = 5
        self.distance = 0.5
        with open(config_file) as file:
            params = yaml.full_load(file)
            if 'fx' in params:
                self.fx = params['fx']
            if 'fy' in params:
                self.fy = params['fy']
            if 'cx' in params:
                self.cx = params['cx']
            if 'cy' in params:
                self.cy = params['cy']
            if 'image_width' in params:
                self.image_width = params['image_width']
            if 'image_height' in params:
                self.image_height = params['image_height']
            if 'horizontal_views' in params:
                self.horizontal_views = params['horizontal_views']
            if 'vertical_views' in params:
                self.vertical_views = params['vertical_views']
            if 'distance' in params:
                self.distance = params['distance']
            if 'batch_size' in params:
                self.batch_size = params['batch_size']    

