import argparse
import os
import numpy as np
from src.extract_frames import extract_frames
from src.orb_baseline import compute_orb_embeddings
from src.resnet_ai import compute_resnet_embeddings
from src.similarity import build_similarity
from src.ordering import compute_order
from src.reconstruct_video import write_video_from_order
from src.evaluate import evaluate_reconstruction
from src.utils import Timer, log_time


def run_pipeline(
    video_path,
    mode="ai",
    frames_out="output/frames",
    embeddings_out=None,
    sim_out="output/similarity.npy",
    order_out="output/order.npy",
    reconstructed_out="output/final_reconstructed.mp4",
    batch=16,
    workers=6,
    fps=30,
):
    # 1. Extract frames
    with Timer("extract_frames"):
        n, video_fps = extract_frames(video_path, frames_out)
    # 2. Feature extraction
    if mode == "baseline":
        embeddings_out = embeddings_out or "output/embeddings_orb.npy"
        with Timer("features_orb"):
            compute_orb_embeddings(
                frames_out, out_path=embeddings_out, nfeatures=500, workers=workers
            )
        sim_method = "euclidean"
    else:
        embeddings_out = embeddings_out or "output/embeddings_resnet.npy"
        with Timer("features_resnet"):
            compute_resnet_embeddings(
                frames_out,
                out_path=embeddings_out,
                batch_size=batch,
                num_threads=workers,
                device="cpu",
            )
        sim_method = "cosine"
    # 3. Similarity
    with Timer("similarity"):
        build_similarity(embeddings_out, out_path=sim_out, method=sim_method)
    sim = np.load(sim_out)
    # 4. Ordering
    with Timer("ordering"):
        order = compute_order(sim, method="spectral+2opt")
        np.save(order_out, order)
    # 5. Reconstruct
    with Timer("reconstruct"):
        write_video_from_order(
            frames_out, order_out, out_video=reconstructed_out, fps=fps
        )
    # 6. Evaluate (no ground truth by default)
    with Timer("evaluate"):
        metrics = evaluate_reconstruction(reconstructed_out, gt_video=None)
    # Save metrics
    import json

    Path = os.path
    metrics_path = os.path.join("output", "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log_time(f"Pipeline finished. mode={mode}. metrics={metrics}")
    print("[INFO] Pipeline complete. Metrics:", metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "ai"], default="ai")
    parser.add_argument("--video", required=True)
    parser.add_argument("--batch", type=int, default=16, help="Batch size for ResNet")
    parser.add_argument(
        "--workers", type=int, default=6, help="Multiprocessing workers"
    )
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    run_pipeline(
        args.video, mode=args.mode, batch=args.batch, workers=args.workers, fps=args.fps
    )
