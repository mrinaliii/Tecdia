import cv2
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count


def orb_desc_for_path(path, nfeatures=500):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kps, desc = orb.detectAndCompute(img, None)
    if desc is None:
        return np.zeros(32, dtype=np.float32)
    return np.mean(desc.astype(np.float32), axis=0)


def compute_orb_embeddings(
    frames_dir, out_path="output/embeddings_orb.npy", nfeatures=500, workers=None
):
    frames = sorted(Path(frames_dir).glob("frame_*.jpg"))
    if workers is None:
        workers = max(1, cpu_count() - 1)
    args = [(str(p), nfeatures) for p in frames]
    with Pool(workers) as p:
        feats = p.map(_orb_worker, args)
    feats = np.stack(feats, axis=0)
    np.save(out_path, feats)
    print(f"[INFO] Saved ORB embeddings to {out_path} shape={feats.shape}")
    return out_path


def _orb_worker(args):
    path, nfeatures = args
    return orb_desc_for_path(path, nfeatures=nfeatures)


if __name__ == "__main__":
    compute_orb_embeddings("output/frames", out_path="output/embeddings_orb.npy")
