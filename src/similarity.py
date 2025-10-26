import numpy as np
from pathlib import Path


def cosine_similarity_matrix(emb):
    return emb.dot(emb.T)


def euclidean_to_similarity(emb):
    xx = np.sum(emb * emb, axis=1, keepdims=True)
    D = xx + xx.T - 2.0 * (emb.dot(emb.T))
    sim = -D
    return sim


def build_similarity(emb_path, out_path="output/similarity.npy", method="cosine"):
    emb = np.load(emb_path)
    if method == "cosine":
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10
        emb = emb / norms
        S = cosine_similarity_matrix(emb)
    else:
        S = euclidean_to_similarity(emb)
    np.save(out_path, S.astype(np.float32))
    print(f"[INFO] Saved similarity matrix to {out_path} shape={S.shape}")
    return out_path


if __name__ == "__main__":
    build_similarity(
        "output/embeddings_resnet.npy",
        out_path="output/sim_resnet.npy",
        method="cosine",
    )
