import torch
from ultralytics import YOLO
import cv2
import config
from utils import load_all, transform_image_to_tensor
from torchvision import transforms
import numpy as np
# 这里后面跟的
model = YOLO(f"cfg/yolov8{config.yolo_model_version}").to(torch.device('cuda'))

print(model)
