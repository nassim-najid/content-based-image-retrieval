from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from src.extract import extract_orb_features

ROOT = Path(__file__).resolve().parents[2]
VOCAB_PATH = ROOT / "features" / "vocab_kmeans.npy"

def extract_bovw(image_path):
    if not VOCAB_PATH.exists():
        raise FileNotFoundError("Vocabulaire BoVW absent. Lancez : python -m src.local_features.train_bovw")
    centers = np.load(VOCAB_PATH).astype(np.float32)
    _, descriptors = extract_orb_features(image_path)
    hist = np.zeros(len(centers), dtype=np.float32)
    if len(descriptors):
        distances = cdist(descriptors.astype(np.float32), centers, metric="euclidean")
        words = np.argmin(distances, axis=1)
        hist = np.bincount(words, minlength=len(centers)).astype(np.float32)
    hist /= (np.linalg.norm(hist) + 1e-8)
    return hist
