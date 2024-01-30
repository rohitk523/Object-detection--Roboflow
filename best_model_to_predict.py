# Load pre-trained model

from ultralytics import YOLO

model_path = '/home/rohit/GithubRepo/ObjectDetection-YOLOv8/runs/detect/train37/weights/best.pt'
source_path = '/home/rohit/GithubRepo/Object-detection--Roboflow/HandGesture/test/images'


model = YOLO(model= model_path)
result = model.predict(source= source_path, save = True)