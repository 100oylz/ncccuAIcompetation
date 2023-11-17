from ultralytics import YOLO
from multiprocessing import freeze_support
if __name__ == '__main__':
    freeze_support()
    model = YOLO('cfg/yolov8n.yaml').load('preTrainedModel/yolov8n.pt')
    results = model.train(data='dataset/train_data.yaml', epochs=100, imgsz=640, batch=4, device=0,resume=True)
