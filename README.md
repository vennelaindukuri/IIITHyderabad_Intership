# IIITHyderabad_Intership
IIIT internship 2025 
# 🧠 Internship Task 2 – Object Detection and Segmentation using YOLOv8

## 📌 Overview
This project demonstrates **object detection and segmentation** using the **YOLOv8** model from the **Ultralytics** library.  
The goal was to detect and segment objects in three categories of images — **animals**, **fruits**, and **traffic** — and evaluate the model’s performance metrics.

---
CODE 
from ultralytics import YOLO
import os

model = YOLO("yolov8n-seg.pt")

image_paths = [
    "images/fruits.jpeg",
    "images/traffic.jpeg",
    "images/animals.jpeg"
]

for img in image_paths:
    if os.path.exists(img):
        results = model.predict(img, save=True)
        print(f"✅ Processed: {img}")
    else:
        print(f"⚠️ Image not found: {img}")

print("🎯 Done! Check 'runs/segment/predict' for output images.")
