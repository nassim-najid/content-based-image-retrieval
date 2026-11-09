from pathlib import Path
import os
import numpy as np
from scipy.spatial.distance import euclidean, cityblock, cosine
from src.local_features.bovw import extract_bovw

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "features" / "bovw_features.npy"
PATHS_PATH = ROOT / "features" / "bovw_paths.npy"

def search_bovw(query_path, top_k=5, metric="cosine"):
    if not FEATURES_PATH.exists() or not PATHS_PATH.exists():
        raise FileNotFoundError("Index BoVW absent. Lancez : python -m src.local_features.index_bovw")
    features = np.load(FEATURES_PATH)
    paths = np.load(PATHS_PATH, allow_pickle=False)
    query_norm = os.path.normcase(os.path.abspath(query_path))
    query = extract_bovw(query_path)
    distance_fn = {"euclidean": euclidean, "manhattan": cityblock, "cosine": cosine}.get(metric, cosine)
    results = []
    for path, vector in zip(paths, features):
        candidate = os.path.abspath(str(path))
        if os.path.normcase(candidate) == query_norm:
            continue
        dist = float(distance_fn(query, vector))
        similarity = 1.0 / (1.0 + dist)
        results.append((candidate, similarity))
    return sorted(results, key=lambda item: item[1], reverse=True)[:top_k]
