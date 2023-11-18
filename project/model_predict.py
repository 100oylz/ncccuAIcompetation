from ultralytics import YOLO

modelpath = './best.pt'


class predict():
    def __init__(self):
        self.model = YOLO(modelpath)

    def detect_image(self, image_path):
        results = self.model(image_path, stream=True, device='cpu')
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs

        cls = boxes.cls.numpy().tolist()
        conf = boxes.conf.numpy().tolist()
        xmin, ymin, xmax, ymax = boxes.xyxy[:, 0].numpy().tolist(), boxes.xyxy[:, 1].numpy().tolist(), boxes.xyxy[:,
                                                                                     2].numpy().tolist(), boxes.xyxy[:,
                                                                                                 3].numpy().tolist()

        return [ymin, xmin, ymax, xmax, conf, cls]


if __name__ == '__main__':
    model = predict()
    print(model.detect_image(
        r"D:\ProgramProject\PycharmProject\ObjectDetection_YOLO\dataset\images\val\0b1e7df6cc0e00d37ca7d16f14529304.jpg"))
