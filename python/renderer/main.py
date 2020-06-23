import os
import tqdm
import cv2
import torch
import argparse
import torch.nn as nn
import numpy as np
from common import Config,ViewControl

config_file = '/home/hah/wm3D/python/config/rendering_config.yaml'
if __name__ == "__main__":
    config = Config(config_file)
    view_control = ViewControl(config)