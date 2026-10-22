"""Forward/backward checks for the neural models (skipped when PyTorch is not installed)."""
import pytest

torch = pytest.importorskip("torch")

from mentalnlp.models import FUSIONS, Embedding, FusionClassifier, build_encoder  # noqa: E402

CFG = {"d_model": 32, "layers": 2, "heads": 2, "ff_dim": 16, "dropout": 0.2,
       "hidden": 8, "kernels": [3, 4], "filters": 6}


def batch(b=3, lens=None):
    lens = lens or {"title": 5, "post": 12}
    out = {}
    for f, n in lens.items():
        mask = torch.ones(b, n, dtype=torch.long)
        mask[0, n // 2:] = 0
        out[f] = {"ids": torch.randint(2, 50, (b, n)), "mask": mask}
    return out


def encoders(kind):
    return {f: build_encoder(kind, Embedding(50, 24), CFG, max_len=16) for f in ("title", "post")}


@pytest.mark.parametrize("fusion", FUSIONS)
def test_transformer_fusion(fusion):
    model = FusionClassifier(encoders("transformer"), 6, fusion)
    logits = model(batch())
    assert logits.shape == (3, 6)
    logits.sum().backward()


@pytest.mark.parametrize("kind", ["cnn", "lstm", "bilstm"])
def test_dl_baselines(kind):
    model = FusionClassifier({"post": build_encoder(kind, Embedding(50, 24), CFG, 16)}, 6)
    assert model(batch(lens={"post": 12})).shape == (3, 6)


def test_concat_width():
    model = FusionClassifier(encoders("transformer"), 6, "concat")
    assert model.classifier.in_features == 64
