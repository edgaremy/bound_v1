from ultralytics import YOLO

# Load a model
model = YOLO('runs/detect/train40/weights/best.pt')

# Export the model
# path = model.export(format="tflite", int8=True)

# Export in onnx format
path = model.export(format="onnx")#, data="/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT18/dataset/Arthropods_wave18.yaml")