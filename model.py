import torch
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss

import config


image = torch.rand((1, 3, 640, 640)).to('cuda')
model = YOLO(f"cfg/yolov8{config.yolo_model_version}.yaml").to(torch.device('cuda'))
output = model.predict(image)
# loss=v8DetectionLoss(output)
# print(loss)
print(type(output[0]))
print(output[0])

