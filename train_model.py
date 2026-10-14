
from ultralytics import YOLO

# Load a small model that we can fine-tune
model = YOLO("yolo11n.pt")

# Train it on our railway crack dataset
model.train(
    data="railway_dataset/data.yaml",
    epochs=30,
    imgsz=640,
    batch=4,
    workers=0,
    patience=8,
    project="runs",
    name="railway_cracks"
)
