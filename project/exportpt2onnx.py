from ultralytics import YOLO

model = YOLO('./best.pt')
model.export(format='onnx', nms=True, dynamic=True, simplify=True, int8=True)
