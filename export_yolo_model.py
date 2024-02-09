from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from ultralytics import YOLO
import os
from collections import Counter

# Load a model
model = YOLO('runs/detect/train16/weights/best.pt') # load best.pt or last.pt of local model

# Export the model
# WARNING: This may need tensorflow<=2.13.1 to work
model.export(format="tflite", optimize=True)  # export to TensorFlow Lite