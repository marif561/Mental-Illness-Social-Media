"""Classical ML baselines: TF-IDF word n-grams (1–3, top 1,000) + LinearSVC / LR / NB / RF."""
from __future__ import annotations

import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from .data import get_texts, load_dataset
from .metrics import evaluate, save, summary_line

MODELS = {
    "linearsvc": lambda s: LinearSVC(random_state=s),
    "lr": lambda s: LogisticRegression(max_iter=2000, random_state=s),
    "nb": lambda s: MultinomialNB(),
    "rf": lambda s: RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=s),
}


def build(name: str, max_features: int = 1000, seed: int = 42):
    vec = TfidfVectorizer(ngram_range=(1, 3), max_features=max_features, lowercase=True)
    return make_pipeline(vec, MODELS[name](seed))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--field", choices=["title", "post", "both"], default="post")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--max-features", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/ml")
    args = ap.parse_args(argv)

    train, _, test = load_dataset(args.data_dir)
    for name in args.models:
        clf = build(name, args.max_features, args.seed).fit(get_texts(train, args.field), train["label_id"])
        res = evaluate(test["label_id"], clf.predict(get_texts(test, args.field)))
        print(f"{name:<10} [{args.field}] {summary_line(res)}")
        save(res, f"{args.out}/{name}_{args.field}.json")


if __name__ == "__main__":
    main()
