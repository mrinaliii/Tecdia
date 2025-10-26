import numpy as np
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh


def spectral_seriation(sim):
    L = laplacian(sim, normed=True)
    vals, vecs = eigh(L)
    if vecs.shape[1] < 2:
        order = np.argsort(vecs[:, 0])
    else:
        fiedler = vecs[:, 1]
        order = np.argsort(fiedler)
    return order


def greedy_nearest(sim, start=None):
    N = sim.shape[0]
    if start is None:
        start = np.argmax(sim.sum(axis=1))
    order = [start]
    used = set(order)
    for _ in range(N - 1):
        last = order[-1]
        cand = -1
        best_sim = -1e9
        for j in range(N):
            if j in used:
                continue
            if sim[last, j] > best_sim:
                best_sim = sim[last, j]
                cand = j
        order.append(cand)
        used.add(cand)
    return np.array(order, dtype=int)


def path_cost(order, dissim):
    idx = order
    return np.sum(dissim[idx[:-1], idx[1:]])


def two_opt(order, dissim, max_iter=200):
    N = len(order)
    best = order.copy()
    best_cost = path_cost(best, dissim)
    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        for i in range(0, N - 2):
            for j in range(i + 2, N):
                new_order = best.copy()
                new_order[i + 1 : j + 1] = new_order[i + 1 : j + 1][::-1]
                new_cost = path_cost(new_order, dissim)
                if new_cost < best_cost - 1e-12:
                    best = new_order
                    best_cost = new_cost
                    improved = True
                    break
            if improved:
                break
    return best


def compute_order(sim, method="spectral+2opt"):
    smin, smax = sim.min(), sim.max()
    if smax - smin > 0:
        sim_norm = (sim - smin) / (smax - smin)
    else:
        sim_norm = sim - smin
    dissim = 1.0 - sim_norm
    if method.startswith("spectral"):
        order = spectral_seriation(sim)
    elif method.startswith("greedy"):
        order = greedy_nearest(sim)
    else:
        order = np.arange(sim.shape[0], dtype=int)
    if "+2opt" in method:
        order = two_opt(order.copy(), dissim)
    return order


if __name__ == "__main__":
    import numpy as np

    sim = np.load("output/similarity.npy")
    sim = np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)
    order = compute_order(sim, method="spectral+2opt")
    np.save("output/order.npy", order)
    print("[INFO] Saved order to output/order.npy")
