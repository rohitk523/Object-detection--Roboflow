from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.track(source='https://youtu.be/gqUg-_JJAEg?si=XR9UeP9_QNQ7o7CR', show=True)