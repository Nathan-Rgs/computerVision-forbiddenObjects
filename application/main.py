import os
import glob
import cv2
from application.core.camera import Camera
from application.core.detector import Detector
from application.core.logger import Logger

# --- ROI global state ---
roi = [0, 0, 0, 0]
is_drawing = False
roi_defined = False

#draw roi
def draw_roi_callback(event, x, y, flags, param):
    """
    Callback function to handle mouse events for drawing the ROI.
    """
    global roi, is_drawing, roi_defined

    if event == cv2.EVENT_LBUTTONDOWN:
        is_drawing = True
        roi = [x, y, x, y]
        roi_defined = False

    elif event == cv2.EVENT_MOUSEMOVE:
        if is_drawing:
            roi[2] = x
            roi[3] = y

    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing = False
        # Ensure top-left corner is stored first
        x_start, x_end = min(roi[0], x), max(roi[0], x)
        y_start, y_end = min(roi[1], y), max(roi[1], y)
        roi = [x_start, y_start, x_end, y_end]
        
        # A very small ROI is likely a misclick, ignore it.
        if (roi[2] - roi[0]) > 10 and (roi[3] - roi[1]) > 10:
            roi_defined = True
            print(f"ROI defined at: {roi}")
        else:
            # If the area is too small, reset the ROI
            roi_defined = False
            roi = [0, 0, 0, 0]


def is_center_in_roi(box_center, current_roi):
    """
    Checks if the center of a bounding box is within the ROI.
    """
    x, y = box_center
    x1, y1, x2, y2 = current_roi
    return x1 < x < x2 and y1 < y < y2


def get_video_source():
    source_choice = input("Enter 'c' for camera or 'v' for video file: ").lower()
    if source_choice == 'c':
        return 0
    elif source_choice == 'v':
        video_path = input("Enter the path to the video file: ")
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at '{video_path}'")
            return get_video_source()
        return video_path
    else:
        print("Invalid choice. Please enter 'c' or 'v'.")
        return get_video_source()


def get_model_weights():
    models = glob.glob("jupyter/Benchmark_Visao/**/weights/best.pt", recursive=True)
    #nao deu para subir o peso da faster rcnn no github
    # models += glob.glob("jupyter/**/best_faster_rcnn_amp.pth", recursive=True)
    if not models:
        print("Error: No models found.")
        return None

    print("Available models:")
    for i, model_path in enumerate(models):
        print(f"{i + 1}: {model_path}")

    model_choice = input(f"Select a model (1-{len(models)}): ")
    try:
        model_index = int(model_choice) - 1
        if 0 <= model_index < len(models):
            return models[model_index]
        else:
            print("Invalid choice. Please select a valid model.")
            return get_model_weights()
    except ValueError:
        print("Invalid input. Please enter a number.")
        return get_model_weights()


def main():
    video_source = get_video_source()
    model_weights = get_model_weights()

    if model_weights is None:
        return

    camera = Camera(video_source)
    detector = Detector(model_weights)
    logger = Logger("data/forbidden_objects_log.csv")
    
    window_name = "Forbidden Object Detection"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_roi_callback)

    print("\n--- Instructions ---")
    print("Drag the mouse on the window to draw a Region of Interest (ROI).")
    print("Only objects detected inside the ROI will be logged.")
    print("Press 'q' to quit.")
    print("--------------------\n")

    while camera.is_opened():
        ok, frame = camera.get_frame()
        if not ok:
            if camera.is_video_file():
                camera.reset()
                continue
            else:
                break

        # --- ROI and Detection Logic ---
        results = detector.detect(frame)
        res = results[0]
        names = res.names

        # Draw the current ROI rectangle
        if is_drawing:
            cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 255), 2)  # Yellow while drawing
        elif roi_defined:
            cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)    # Green when defined

        if res.boxes is not None and len(res.boxes) > 0:
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                cls_name = names.get(cls_id, str(cls_id))
                conf = float(box.conf[0].item())

                # If an ROI is defined, only process boxes inside it
                if roi_defined:
                    box_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    if is_center_in_roi(box_center, roi):
                        # This box is inside the ROI, so log and draw it
                        logger.log_event(cls_name, conf)
                        detector.draw_box(frame, [x1, y1, x2, y2], f"{cls_name} {conf:.2f}")
                    # else: box is outside ROI, do nothing with it
                
                else: 
                    # If no ROI is defined, just draw all boxes (no logging)
                    detector.draw_box(frame, [x1, y1, x2, y2], f"{cls_name} {conf:.2f}")

        camera.show_frame(frame, window_name)

        if camera.wait_key(1) & 0xFF == ord("q"):
            break

    camera.release()
    camera.destroy_all_windows()


if __name__ == "__main__":
    main()
