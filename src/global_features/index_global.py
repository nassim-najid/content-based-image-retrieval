from pathlib import Path
import numpy as np
from src.extract import extract_global_features

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
FEATURES_DIR = ROOT / "features"

def index_global():
    paths = sorted(p for p in DATA_DIR.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not paths:
        raise RuntimeError("Aucune image trouvée dans data/. Ajoutez les images dans les dossiers de classes.")
    vectors = []
    valid_paths = []
    for i, path in enumerate(paths, 1):
        try:
            vectors.append(extract_global_features(path))
            valid_paths.append(str(path.resolve()))
            print(f"[{i}/{len(paths)}] {path.name}")
        except Exception as exc:
            print(f"Image ignorée ({path}): {exc}")
    FEATURES_DIR.mkdir(exist_ok=True)
    np.save(FEATURES_DIR / "features_global.npy", np.asarray(vectors, dtype=np.float32))
    np.save(FEATURES_DIR / "paths_global.npy", np.asarray(valid_paths))
    print(f"Index global créé : {len(valid_paths)} images.")
if __name__ == "__main__":
    index_global()
