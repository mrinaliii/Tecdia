import cv2
import os


def extract_frames(video_path, output_dir, force=False):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Error opening video file: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Video opened: frames={total}, fps={fps}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
        if not os.path.exists(frame_path) or force:
            cv2.imwrite(frame_path, frame)
        frame_idx += 1

    cap.release()
    print(f"[INFO] Extracted {frame_idx} frames to {output_dir}")
    return frame_idx, fps


if __name__ == "__main__":
    extract_frames("input/jumbled_video.mp4", "output/frames")
