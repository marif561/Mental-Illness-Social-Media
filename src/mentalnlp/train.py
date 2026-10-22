"""Train the proposed late-fusion Transformer or a CNN / LSTM / BiLSTM baseline.

Paper settings (Section 5.1, Table 3): RoBERTa pretrained embeddings as the input layer,
Adam (lr 1e-5, eps 1e-8) with gradient clipping, batch size 16, early stopping with
patience 5 on the dev set, max length 35 for titles and 512 for posts.

    --field title | post        one encoder on a single text
    --field both                one encoder per text + late fusion (--fusion)
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from transformers import AutoModel, AutoTokenizer

from .data import LABELS, load_dataset
from .metrics import evaluate, save, summary_line
from .models import FUSIONS, Embedding, FusionClassifier, build_encoder


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize(tok, texts, max_len):
    enc = tok(list(texts), truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def make_inputs(df, fields, tok, max_lens):
    return {f: tokenize(tok, df[f], max_lens[f]) for f in fields}


def batches(inputs, labels, bs, shuffle, seed=0):
    n = len(labels)
    order = np.random.default_rng(seed).permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        idx = torch.as_tensor(order[i:i + bs])
        yield {f: {"ids": ids[idx], "mask": m[idx]} for f, (ids, m) in inputs.items()}, labels[idx]


def to_device(batch, device):
    return {f: {k: v.to(device) for k, v in d.items()} for f, d in batch.items()}


@torch.no_grad()
def predict(model, inputs, labels, bs, device):
    model.eval()
    preds = []
    for b, _ in batches(inputs, labels, bs, shuffle=False):
        preds.append(model(to_device(b, device)).argmax(-1).cpu())
    return torch.cat(preds).numpy()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model", choices=["transformer", "cnn", "lstm", "bilstm"], default="transformer")
    ap.add_argument("--field", choices=["title", "post", "both"], default="both")
    ap.add_argument("--fusion", choices=FUSIONS, default="concat")
    ap.add_argument("--config", default=None, help="defaults to configs/<model>.yaml")
    ap.add_argument("--embeddings", default="roberta-base",
                    help="pretrained model whose tokenizer and input embeddings are used")
    ap.add_argument("--max-epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args(argv)

    cfg_path = args.config or f"configs/{'transformer' if args.model == 'transformer' else 'dl'}.yaml"
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    mcfg = cfg[args.model] if args.model in cfg else cfg
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train, dev, test = load_dataset(args.data_dir)
    fields = ["title", "post"] if args.field == "both" else [args.field]
    max_lens = {"title": cfg["max_len_title"], "post": cfg["max_len_post"]}

    tok = AutoTokenizer.from_pretrained(args.embeddings)
    weights = AutoModel.from_pretrained(args.embeddings).get_input_embeddings().weight.detach().clone()
    encoders = {f: build_encoder(args.model,
                                 Embedding(weights.size(0), weights.size(1), weights, pad_id=tok.pad_token_id),
                                 mcfg, max_lens[f])
                for f in fields}
    model = FusionClassifier(encoders, len(LABELS), args.fusion, mcfg["dropout"]).to(device)

    tr_in, dv_in, te_in = (make_inputs(d, fields, tok, max_lens) for d in (train, dev, test))
    ytr, ydv, yte = (torch.as_tensor(d["label_id"].values) for d in (train, dev, test))

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=float(cfg["learning_rate"]), eps=float(cfg["adam_eps"]))
    loss_fn = nn.CrossEntropyLoss()
    best, best_state, bad = -1.0, None, 0
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        total = 0.0
        for b, y in batches(tr_in, ytr, cfg["batch_size"], shuffle=True, seed=args.seed + epoch):
            opt.zero_grad()
            loss = loss_fn(model(to_device(b, device)), y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            opt.step()
            total += loss.item() * len(y)
        dev_f1 = evaluate(ydv.numpy(), predict(model, dv_in, ydv, 64, device))["micro"]["f1"]
        print(f"epoch {epoch:>3}  loss {total / len(ytr):.4f}  dev micro-F1 {dev_f1:.2f}")
        if dev_f1 > best:
            best, best_state, bad = dev_f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print(f"early stopping (patience {cfg['patience']})")
                break

    model.load_state_dict(best_state)
    pred = predict(model, te_in, yte, 64, device)
    res = evaluate(yte.numpy(), pred)
    tag = f"{args.model}_{args.field}" + (f"_{args.fusion}" if args.field == "both" else "")
    print(f"\n{tag}: {summary_line(res)}")
    out = Path(args.out) / tag
    save(res, out / "metrics.json")
    test.assign(prediction=[LABELS[i] for i in pred]).to_csv(out / "test_predictions.csv", index=False)


if __name__ == "__main__":
    main()
