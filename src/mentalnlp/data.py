"""Data loading for the Reddit mental-illness benchmark (Murarka et al., 2021)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

LABELS: List[str] = ["adhd", "anxiety", "bipolar", "depression", "ptsd", "none"]
LABEL2ID: Dict[str, int] = {l: i for i, l in enumerate(LABELS)}
FIELDS = ("title", "post", "both")

_ALIASES = {"adha": "adhd", "no": "none", "control": "none", "post-traumatic stress disorder": "ptsd"}


def normalise_label(label) -> str:
    key = str(label).strip().lower()
    key = _ALIASES.get(key, key)
    if key not in LABEL2ID:
        raise ValueError(f"Unknown label {label!r}; expected one of {LABELS}")
    return key


def find_split(data_dir: str | Path, split: str) -> Path:
    """Locate ``<split>.csv`` (or the original ``<split>.csv`` variants) in ``data_dir``."""
    data_dir = Path(data_dir)
    for name in (f"{split}.csv", f"{split}_data.csv", f"{split}.tsv"):
        if (data_dir / name).exists():
            return data_dir / name
    raise FileNotFoundError(f"No {split} file found in {data_dir}")


def load_split(path: str | Path, title_col: str = "title", post_col: str = "post",
               label_col: str = "class_name") -> pd.DataFrame:
    """Return a frame with ``title``, ``post``, ``label`` (name) and ``label_id`` columns.

    Missing titles/posts become empty strings. If ``label_col`` is absent, a numeric
    ``class_id`` column is used instead when present.
    """
    path = str(path)
    df = pd.read_csv(path, sep="\t" if path.endswith(".tsv") else ",")
    out = pd.DataFrame({
        "title": df[title_col].fillna("").astype(str) if title_col in df else "",
        "post": df[post_col].fillna("").astype(str) if post_col in df else "",
    })
    if label_col in df:
        out["label"] = [normalise_label(l) for l in df[label_col]]
    elif "class_id" in df:
        out["label"] = [LABELS[int(i)] for i in df["class_id"]]
    else:
        raise KeyError(f"No label column {label_col!r} or 'class_id' in {path}")
    out["label_id"] = out["label"].map(LABEL2ID)
    return out


def get_texts(df: pd.DataFrame, field: str) -> List[str]:
    """Texts for single-input models. ``both`` joins title and post with a separator."""
    if field == "title":
        return df["title"].tolist()
    if field == "post":
        return df["post"].tolist()
    if field == "both":
        return (df["title"] + " . " + df["post"]).tolist()
    raise ValueError(f"field must be one of {FIELDS}")


def load_dataset(data_dir: str | Path, **kw) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return tuple(load_split(find_split(data_dir, s), **kw) for s in ("train", "dev", "test"))  # type: ignore
