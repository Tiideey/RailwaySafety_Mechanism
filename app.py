
import os
import threading

import cv2
from flask import Flask, Response, jsonify, render_template
from ultralytics import YOLO

app = Flask(__name__)

MODEL_PATH = os.path.join(
    "runs", "detect", "runs", "railway_cracks",
    "weights", "best.pt"
)

model = YOLO(MODEL_PATH)

camera = None
camera_lock = threading.Lock()
state = {
    "running": False,
    "status": "READY",
    "detections": 0,
    "confidence": 0.0,
    "message": "Press Start Camera to begin."
}


def release_camera():
    global camera

    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None

    state["running"] = False


def generate_frames():
    global camera

    while state["running"]:
        with camera_lock:
            current_camera = camera
            if current_camera is None:
                break
            success, frame = current_camera.read()

        if not success:
            state["message"] = "Could not read camera."
            break

        results = model.predict(
            source=frame,
            imgsz=640,
            conf=0.15,
            verbose=False
        )

        result = results[0]
        count = len(result.boxes) if result.boxes is not None else 0

        state["detections"] = count

        if count:
            confidences = result.boxes.conf.tolist()
            state["confidence"] = max(confidences) * 100
            state["status"] = "CANDIDATE DETECTED"
            state["message"] = (
                f"{count} candidate detection(s). "
                "Inspect the image to verify."
            )
        else:
            state["confidence"] = 0.0
            state["status"] = "NO DETECTION"
            state["message"] = (
                "No crack candidate detected in this frame."
            )

        
        # Draw readable class names and confidence percentages
        annotated = frame.copy()

        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )

                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = str(
                    result.names.get(class_id, "Unknown")
                ).replace("_", " ").upper()

                label = f"{class_name} - {confidence * 100:.1f}%"

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (0, 220, 100),
                    2
                )

                # Keep the label within the image
                label_y = max(y1 - 10, 25)

                cv2.putText(
                    annotated,
                    label,
                    (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 220, 100),
                    2,
                    cv2.LINE_AA
                )

        ok, buffer = cv2.imencode(".jpg", annotated)


        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )

    release_camera()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start():
    global camera

    if state["running"]:
        return jsonify({"ok": True, "message": "Camera is already running."})

    with camera_lock:
        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            camera.release()
            camera = None
            state["status"] = "CAMERA ERROR"
            state["message"] = "Could not open webcam."
            return jsonify({"ok": False, "message": state["message"]}), 500

    state["running"] = True
    state["status"] = "STARTING"
    state["message"] = "Camera started. Analyzing frames."

    return jsonify({"ok": True, "message": state["message"]})


@app.route("/stop", methods=["POST"])
def stop():
    release_camera()
    state["status"] = "STOPPED"
    state["detections"] = 0
    state["confidence"] = 0.0
    state["message"] = "Camera stopped."
    return jsonify({"ok": True, "message": state["message"]})


@app.route("/video_feed")
def video_feed():
    if not state["running"]:
        return "Camera is stopped.", 400

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/status")
def get_status():
    return jsonify(state)


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )
