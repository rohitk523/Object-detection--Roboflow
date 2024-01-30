from ultralytics import YOLO
import os

# Load pre-trained model
model_path = '/home/rohit/GithubRepo/ObjectDetection-YOLOv8/runs/detect/train37/weights/best.pt'
source_path = 'https://youtu.be/HQIibrdTJcY?si=rHJFJf9giu-mc3pH'

model = YOLO('yolov8m.pt')
results = model.track(source=source_path, show=True)