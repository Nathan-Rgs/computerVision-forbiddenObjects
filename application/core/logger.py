import csv
import os
from datetime import datetime


class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        self._ensure_log_header()

    def _ensure_log_header(self):
        exists = os.path.exists(self.log_file)
        if not exists:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["timestamp", "object_class", "confidence"]
                )

    def log_event(self, object_class, confidence):
        ts = datetime.now().isoformat(timespec="seconds")
        with open(self.log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [ts, object_class, f"{confidence:.2f}"]
            )
