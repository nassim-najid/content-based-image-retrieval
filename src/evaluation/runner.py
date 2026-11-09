from pathlib import Path
import csv
import numpy as np
from src.evaluation.metrics import precision_recall_f1, average_precision
from src.global_features.search_global import search_global
from src.local_features.search_bovw import search_bovw

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

def run_full_evaluation(k_values=(5, 10, 25)):
    paths = sorted(str(p.resolve()) for p in DATA.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if len(paths) < 2:
        raise RuntimeError("Ajoutez au moins deux images dans data/ avant l'évaluation.")
    rows = []
    for mode in ("global", "local"):
        for metric in ("euclidean", "manhattan", "cosine"):
            aps, accum = [], {k: [[], [], []] for k in k_values}
            for i, query in enumerate(paths, 1):
                results = (search_global(query, metric, len(paths) - 1) if mode == "global"
                           else search_bovw(query, len(paths) - 1, metric))

                ranked = [path for path, score in results]
                aps.append(average_precision(ranked, query, paths))
                for k in k_values:
                    p, r, f = precision_recall_f1(ranked, query, k, paths)
                    accum[k][0].append(p); accum[k][1].append(r); accum[k][2].append(f)
                if i % 20 == 0:
                    print(f"{mode}/{metric}: {i}/{len(paths)}")
            map_score = float(np.mean(aps))
            for k in k_values:
                p, r, f = (float(np.mean(v)) for v in accum[k])
                rows.append([mode, metric, k, p, r, f, map_score])
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "evaluation_report.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "distance", "K", "precision", "recall", "f1", "MAP"])
        writer.writerows(rows)
    for row in rows:
        print(f"{row[0]:6} {row[1]:10} K={row[2]:2} P={row[3]:.4f} R={row[4]:.4f} F1={row[5]:.4f} MAP={row[6]:.4f}")
    print(f"Rapport : {out}")
    return rows

if __name__ == "__main__":
    run_full_evaluation()
