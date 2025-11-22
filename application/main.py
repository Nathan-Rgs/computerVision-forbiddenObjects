import os
import glob
from application.core.camera import Camera
from application.core.detector import Detector
from application.core.logger import Logger


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

    while camera.is_opened():
        ok, frame = camera.get_frame()
        if not ok:
            break

        results = detector.detect(frame)
        res = results[0]
        names = res.names

        if res.boxes is not None and len(res.boxes) > 0:
            for box in res.boxes:
                cls_id = int(box.cls[0].item())
                cls_name = names.get(cls_id, str(cls_id))
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                logger.log_event(cls_name, conf)
                detector.draw_box(frame, [x1, y1, x2, y2], f"{cls_name} {conf:.2f}")

        camera.show_frame(frame, "Forbidden Object Detection")

        if camera.wait_key(1) & 0xFF == ord("q"):
            break

    camera.release()
    camera.destroy_all_windows()


if __name__ == "__main__":
    main()