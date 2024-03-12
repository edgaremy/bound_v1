"""
This script is used to test the model on live webcam feed.
"""
from ultralytics import YOLO
import cv2
import numpy as np
import time
from utils import draw_bboxes


MODEL_PATH = 'best.onnx'
WINDOW_NAME = 'Live test'


# Load model.
model = YOLO(MODEL_PATH, task='detect')

# Open webcam.
cap = cv2.VideoCapture(0)
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

print('Press "q" to quit.')

# Start time.
cur_time = time.time()
dt_list = []
nb_dt = 100

while True:
    # Calculate FPS.
    new_time = time.time()
    dt = new_time - cur_time
    cur_time = new_time
    dt_list.append(dt)
    if len(dt_list) > nb_dt:
        dt_list.pop(0)
    fps = 1 / np.mean(dt_list)
    print(f'\x1b[2K\rFPS: {fps:.2f}', end='')

    # Read frame.
    ret, img = cap.read()
    if not ret:
        break

    # Detect.
    results = model.track(
        source=img,
        stream=True,
        agnostic_nms=True,
        verbose=False,
        persist=True,
    )

    # Draw boxes.
    for r in results:
        boxes = r.boxes
        image = r.orig_img
        draw_bboxes(
            image, boxes.data,
            track_ids=boxes.id,
            labels=r.names,
            score=True,
            conf=0.5,
        )

    # Show image.
    cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow(WINDOW_NAME, img)
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources.
cap.release()
cv2.destroyAllWindows()