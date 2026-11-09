from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from src.extract import extract_orb_features

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
FEATURES_DIR = ROOT / "features"
VOCAB_SIZE = 50

def train_vocabulary():
    image_paths = sorted(p for p in DATA_DIR.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    descriptor_sets = []
    for i, path in enumerate(image_paths, 1):
        _, descriptors = extract_orb_features(path)
        if len(descriptors):
            descriptor_sets.append(descriptors)
        print(f"[{i}/{len(image_paths)}] {path.name}")
    if not descriptor_sets:
        raise RuntimeError("Aucun descripteur ORB trouvé. Vérifiez le dataset.")
    all_descriptors = np.vstack(descriptor_sets).astype(np.float32)
    if len(all_descriptors) < VOCAB_SIZE:
        raise RuntimeError(f"Pas assez de descripteurs ({len(all_descriptors)}) pour créer {VOCAB_SIZE} clusters.")
    model = KMeans(n_clusters=VOCAB_SIZE, random_state=42, n_init=10)
    model.fit(all_descriptors)
    FEATURES_DIR.mkdir(exist_ok=True)
    np.save(FEATURES_DIR / "vocab_kmeans.npy", model.cluster_centers_.astype(np.float32))
    print(f"Vocabulaire sauvegardé : {VOCAB_SIZE} mots visuels.")
if __name__ == "__main__":
    train_vocabulary()
