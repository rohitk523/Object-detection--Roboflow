from ultralytics import YOLO


source_path = '/home/rohit/GithubRepo/Object-detection--Roboflow/footballPlayers/test/images'
# Create a new YOLO model from scratch
model = YOLO('yolov8n.yaml')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data='footballPlayers/data.yaml', epochs=20,)

# Evaluate the model's performance on the validation set
results = model.val()

model.predict(source= source_path, save= True)