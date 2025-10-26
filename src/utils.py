import time
from datetime import datetime
from pathlib import Path
import json


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_time(message, path="output/runtime_log.txt"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(f"[{now()}] {message}\n")


class Timer:
    def __init__(self, name, path="output/runtime_log.txt"):
        self.name = name
        self.path = path

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self.t0
        log_time(f"{self.name}: {dt:.4f}s", self.path)
