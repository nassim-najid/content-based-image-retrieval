from pathlib import Path
import numpy as np
from src.local_features.bovw import extract_bovw

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
FEATURES_DIR = ROOT / "features"

def index_bovw():
    if not (FEATURES_DIR / "vocab_kmeans.npy").exists():
        raise FileNotFoundError("Vocabulaire absent. Lancez d'abord : python -m src.local_features.train_bovw")
    paths = sorted(p for p in DATA_DIR.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not paths:
        raise RuntimeError("Aucune image trouvée dans data/.")
    vectors, valid_paths = [], []
    for i, path in enumerate(paths, 1):
        vectors.append(extract_bovw(path))
        valid_paths.append(str(path.resolve()))
        print(f"[{i}/{len(paths)}] {path.name}")
    FEATURES_DIR.mkdir(exist_ok=True)
    np.save(FEATURES_DIR / "bovw_features.npy", np.asarray(vectors, dtype=np.float32))
    np.save(FEATURES_DIR / "bovw_paths.npy", np.asarray(valid_paths))
    print(f"Index local BoVW créé : {len(valid_paths)} images.")
if __name__ == "__main__":
    index_bovw()
