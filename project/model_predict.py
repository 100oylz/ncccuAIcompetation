from ultralytics import YOLO
import torch
modelpath = './best.pt'


class predict():
    def __init__(self):
        self.model = YOLO(modelpath)

    def detect_image(self, image_path):
        results = self.model(image_path, stream=True, device='cpu')
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
        results = []
        for cls, conf, xyxy in zip(boxes.cls, boxes.conf, boxes.xyxy):
            results.append([xyxy[1].item(), xyxy[0].item(), xyxy[3].item(), xyxy[2].item(), conf.item(), cls.item()])

        return results


if __name__ == '__main__':

    model = predict()
    print(model.detect_image(
        r"D:\ProgramProject\PycharmProject\ObjectDetection_YOLO\dataset\images\val\0b1e7df6cc0e00d37ca7d16f14529304.jpg"))
