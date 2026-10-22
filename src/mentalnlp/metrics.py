"""Evaluation. The paper reports micro-averaged precision, recall and F1."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .data import LABELS


def evaluate(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict:
    ids = list(range(len(LABELS)))
    res = {}
    for avg in ("micro", "macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=ids, average=avg, zero_division=0)
        res[avg] = {"precision": 100 * p, "recall": 100 * r, "f1": 100 * f}
    p, r, f, n = precision_recall_fscore_support(y_true, y_pred, labels=ids, zero_division=0)
    res["per_class"] = {lab: {"precision": 100 * p[i], "recall": 100 * r[i], "f1": 100 * f[i], "support": int(n[i])}
                        for i, lab in enumerate(LABELS)}
    res["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=ids).tolist()
    return res


def summary_line(res: Dict) -> str:
    m = res["micro"]
    return f"micro P {m['precision']:.2f} · R {m['recall']:.2f} · F1 {m['f1']:.2f} | macro F1 {res['macro']['f1']:.2f}"


def save(res: Dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=2), encoding="utf-8")
