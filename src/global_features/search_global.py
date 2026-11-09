from pathlib import Path
import os
import numpy as np
from scipy.spatial.distance import euclidean, cityblock, cosine
from src.extract import extract_global_features

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "features" / "features_global.npy"
PATHS_PATH = ROOT / "features" / "paths_global.npy"

def search_global(query_image_path, method="euclidean", top_k=5):
    if not FEATURES_PATH.exists() or not PATHS_PATH.exists():
        raise FileNotFoundError("Index global absent. Lancez d'abord : python -m src.global_features.index_global")
    features = np.load(FEATURES_PATH)
    paths = np.load(PATHS_PATH, allow_pickle=False)
    query_path = os.path.normcase(os.path.abspath(query_image_path))
    query_features = extract_global_features(query_image_path)
    distance_fn = {"euclidean": euclidean, "manhattan": cityblock, "cosine": cosine}.get(method, euclidean)
    results = []
    for path, feature in zip(paths, features):
        candidate = os.path.abspath(str(path))
        if os.path.normcase(candidate) == query_path:
            continue
        dist = float(distance_fn(query_features, feature))
        results.append((candidate, dist))
    return sorted(results, key=lambda item: item[1])[:top_k]
