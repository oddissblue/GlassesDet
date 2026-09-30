"""
GlassesDet — Real-Time Webcam Inference

Runs the exported YOLO12m GlassesDet ONNX model on a live webcam feed.

Classes:
    0 - not_wearing_glasses
    1 - wearing_glasses

Controls:
    Q - Quit
"""

from pathlib import Path
import time

import cv2
from ultralytics import YOLO


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = Path(
    r"./exported_wights/Yolo12m_GlassesDetector_V1.onnx"
)

CAMERA_ID = 0
CONFIDENCE_THRESHOLD = 0.30
IMAGE_SIZE = 640

WINDOW_NAME = "GlassesDet - YOLO12m ONNX Live Inference"


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Validate model path
    # --------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"ONNX model not found:\n{MODEL_PATH}"
        )

    print("=" * 70)
    print("GLASSESDET — REAL-TIME YOLO12M ONNX INFERENCE")
    print("=" * 70)

    print(f"Model      : {MODEL_PATH}")
    print(f"Backend    : ONNX")
    print(f"Camera     : {CAMERA_ID}")
    print(f"Confidence : {CONFIDENCE_THRESHOLD}")
    print(f"Image size : {IMAGE_SIZE}")

    # --------------------------------------------------------
    # Load exported ONNX model
    # --------------------------------------------------------

    print("\nLoading ONNX model...")

    model = YOLO(str(MODEL_PATH), task="detect")

    print("ONNX model loaded successfully.")

    print("\nClasses:")

    for class_id, class_name in model.names.items():
        print(f"  {class_id}: {class_name}")

    # --------------------------------------------------------
    # Open webcam
    # --------------------------------------------------------

    print("\nOpening webcam...")

    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {CAMERA_ID}."
        )

    print("Webcam opened successfully.")
    print("\nPress Q to quit.")
    print("=" * 70)

    # --------------------------------------------------------
    # Live inference
    # --------------------------------------------------------

    previous_time = time.perf_counter()

    try:

        while True:

            success, frame = cap.read()

            if not success:
                print("Failed to capture webcam frame.")
                break

            # ------------------------------------------------
            # ONNX inference
            # ------------------------------------------------

            results = model.predict(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                imgsz=IMAGE_SIZE,
                device=0,
                verbose=False,
            )

            # Draw detections
            annotated_frame = results[0].plot()

            # ------------------------------------------------
            # Calculate FPS
            # ------------------------------------------------

            current_time = time.perf_counter()

            elapsed = current_time - previous_time

            fps = 1.0 / elapsed if elapsed > 0 else 0.0

            previous_time = current_time

            # ------------------------------------------------
            # Draw FPS
            # ------------------------------------------------

            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # ------------------------------------------------
            # Display
            # ------------------------------------------------

            cv2.imshow(
                WINDOW_NAME,
                annotated_frame,
            )

            # Press Q to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:

        cap.release()
        cv2.destroyAllWindows()

        print("\nWebcam released.")
        print("Live inference stopped.")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()