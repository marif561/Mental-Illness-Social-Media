from pathlib import Path

import pytest

from mentalnlp.baselines_ml import main as ml_main
from mentalnlp.data import LABELS, get_texts, load_dataset, normalise_label
from mentalnlp.metrics import evaluate

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample"


def test_load_sample():
    train, dev, test = load_dataset(SAMPLE)
    assert len(train) == 12 and len(dev) == 6 and len(test) == 6
    assert set(train["label"]) == set(LABELS)
    assert train["label_id"].between(0, 5).all()


def test_fields():
    train, _, _ = load_dataset(SAMPLE)
    both = get_texts(train, "both")
    assert both[0].startswith(train["title"][0]) and both[0].endswith(train["post"][0])
    with pytest.raises(ValueError):
        get_texts(train, "body")


def test_label_aliases():
    assert normalise_label("ADHA") == "adhd"
    assert normalise_label(" PTSD ") == "ptsd"
    with pytest.raises(ValueError):
        normalise_label("ocd")


def test_metrics_micro_equals_accuracy():
    res = evaluate([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 5, 5])
    assert abs(res["micro"]["f1"] - 100 * 5 / 6) < 1e-9
    assert res["confusion_matrix"][4][5] == 1


def test_ml_baseline_runs(tmp_path, capsys):
    ml_main(["--data-dir", str(SAMPLE), "--models", "lr", "nb", "--out", str(tmp_path)])
    assert (tmp_path / "lr_post.json").exists()
    assert "micro" in capsys.readouterr().out
