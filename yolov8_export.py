from ultralytics import YOLO

# Load a model
model = YOLO('runs/detect/train16/weights/best.pt')

# Export the model
path = model.export(format="onnx")