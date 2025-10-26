import cv2
from pathlib import Path
import numpy as np


def write_video_from_order(
    frames_dir, order_path, out_video="output/final_reconstructed.mp4", fps=30
):
    order = np.load(order_path).astype(int)
    frames = sorted(Path(frames_dir).glob("frame_*.jpg"))
    if len(frames) == 0:
        raise RuntimeError("No frames found in frames_dir")
    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, fps, (w, h))
    for idx in order:
        path = frames[idx]
        img = cv2.imread(str(path))
        writer.write(img)
    writer.release()
    print(f"[INFO] Wrote video {out_video} fps={fps} frames={len(order)}")
    return out_video


if __name__ == "__main__":
    write_video_from_order("output/frames", "output/order.npy")
