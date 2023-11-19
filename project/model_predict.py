from ultralytics import YOLO
import torch

modelpath = './best.onnx'


class predict():
    def __init__(self):
        self.model = YOLO(modelpath, task='detect')

    def detect_image(self, image_path):
        results = self.model(image_path, stream=True, device='cpu')
        boxresults = []
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            for cls, conf, xyxy in zip(boxes.cls, boxes.conf, boxes.xyxy):
                boxresults.append(
                    [xyxy[1].item(), xyxy[0].item(), xyxy[3].item(), xyxy[2].item(), conf.item(), cls.item()])

        return boxresults


if __name__ == '__main__':
    model = predict()
    print(model.detect_image(
        r"D:\ProgramProject\PycharmProject\ObjectDetection_YOLO\dataset\images\val\0c842b1611a496b84c93716362f20482.jpg"))
