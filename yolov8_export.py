from ultralytics import YOLO

# Load a model
model = YOLO('runs/detect/train38/weights/best.pt')

# Export the model
path = model.export(format="tflite", int8=True)