
import cv2
from ultralytics import YOLO

# Load our trained railway crack detector

model = YOLO("runs/detect/runs/railway_cracks/weights/best.pt")

# Open the laptop's built-in camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not be opened.")
    print("Close other camera apps and try again.")
    raise SystemExit

print("Railway crack detection is running.")
print("Press Q to close the camera window.")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read a camera frame.")
        break

    # Analyze the current frame
    results = model.predict(
        source=frame,
        imgsz=640,
        conf=0.15,
        verbose=False
    )

    # Draw the predicted boxes and labels
    annotated_frame = results[0].plot()

    cv2.imshow(
        "Railway Crack Detection - Press Q to exit",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

