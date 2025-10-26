import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from pathlib import Path


def frames_from_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def compute_mse(a, b):
    return float(((a.astype("float32") - b.astype("float32")) ** 2).mean())


def evaluate_reconstruction(recon_video, gt_video=None):
    recon = frames_from_video(recon_video)
    res = {}
    if gt_video is not None:
        gt = frames_from_video(gt_video)
        n = min(len(recon), len(gt))
        mses = []
        ssims = []
        for i in range(n):
            mses.append(compute_mse(recon[i], gt[i]))
            gray1 = cv2.cvtColor(recon[i], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(gt[i], cv2.COLOR_BGR2GRAY)
            ssims.append(ssim(gray1, gray2))
        res["mse_mean"] = float(np.mean(mses))
        res["ssim_mean"] = float(np.mean(ssims))
    else:
        diffs = []
        for i in range(len(recon) - 1):
            diffs.append(compute_mse(recon[i], recon[i + 1]))
        res["continuity_mse_mean"] = float(np.mean(diffs)) if diffs else None
    return res


if __name__ == "__main__":
    r = evaluate_reconstruction("output/final_reconstructed.mp4", gt_video=None)
    print("Evaluation:", r)
