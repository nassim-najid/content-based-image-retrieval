import os

def class_from_path(path):
    return os.path.basename(os.path.dirname(os.path.abspath(path)))

def precision_recall_f1(ranked_paths, query_path, k, all_paths):
    label = class_from_path(query_path)
    q = os.path.normcase(os.path.abspath(query_path))
    relevant_total = sum(1 for p in all_paths if os.path.normcase(os.path.abspath(p)) != q and class_from_path(p) == label)
    hits = sum(1 for p in ranked_paths[:k] if class_from_path(p) == label and os.path.normcase(os.path.abspath(p)) != q)
    precision = hits / k if k else 0.0
    recall = hits / relevant_total if relevant_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1

def average_precision(ranked_paths, query_path, all_paths):
    label = class_from_path(query_path)
    q = os.path.normcase(os.path.abspath(query_path))
    total = sum(1 for p in all_paths if os.path.normcase(os.path.abspath(p)) != q and class_from_path(p) == label)
    if not total:
        return 0.0
    hits, score = 0, 0.0
    for rank, path in enumerate(ranked_paths, 1):
        if os.path.normcase(os.path.abspath(path)) == q:
            continue
        if class_from_path(path) == label:
            hits += 1
            score += hits / rank
    return score / total
