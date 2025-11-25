import cv2


class Camera:
    def __init__(self, source):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

    def get_frame(self):
        ok, frame = self.cap.read()
        return ok, frame

    def is_video_file(self):
        return isinstance(self.source, str)

    def reset(self):
        if self.is_video_file():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def show_frame(self, frame, title="Frame"):
        cv2.imshow(title, frame)

    def release(self):
        self.cap.release()

    def is_opened(self):
        return self.cap.isOpened()

    @staticmethod
    def wait_key(delay=1):
        return cv2.waitKey(delay)

    @staticmethod
    def destroy_all_windows():
        cv2.destroyAllWindows()
