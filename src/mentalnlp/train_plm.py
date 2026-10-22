"""Fine-tune a pretrained language model (BERT / ALBERT / RoBERTa) as a transfer-learning baseline.

Paper settings: softmax output layer, lr 1e-5, batch size 16, early stopping (patience 5)
on the dev set, max length 35 for titles and 512 for posts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          EarlyStoppingCallback, Trainer, TrainingArguments)

from .data import LABELS, get_texts, load_dataset
from .metrics import evaluate, save, summary_line


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model", default="roberta-base", help="bert-base-uncased | albert-base-v2 | roberta-base")
    ap.add_argument("--field", choices=["title", "post", "both"], default="post")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/plm")
    args = ap.parse_args(argv)

    max_len = 35 if args.field == "title" else 512
    train, dev, test = load_dataset(args.data_dir)
    tok = AutoTokenizer.from_pretrained(args.model)

    def ds(df):
        d = Dataset.from_dict({"text": get_texts(df, args.field), "labels": df["label_id"].tolist()})
        return d.map(lambda b: tok(b["text"], truncation=True, max_length=max_len), batched=True,
                     remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
        label2id={l: i for i, l in enumerate(LABELS)})
    tag = f"{args.model.split('/')[-1]}_{args.field}"
    targs = TrainingArguments(
        output_dir=f"{args.out}/{tag}/checkpoints", learning_rate=1e-5,
        per_device_train_batch_size=16, per_device_eval_batch_size=32, num_train_epochs=args.epochs,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model="micro_f1", report_to=[], seed=args.seed, fp16=torch.cuda.is_available())
    trainer = Trainer(
        model=model, args=targs, train_dataset=ds(train), eval_dataset=ds(dev),
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=lambda p: {"micro_f1": evaluate(p.label_ids, np.argmax(p.predictions, -1))["micro"]["f1"]},
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)])
    trainer.train()

    pred = np.argmax(trainer.predict(ds(test)).predictions, -1)
    res = evaluate(test["label_id"], pred)
    print(f"{tag}: {summary_line(res)}")
    save(res, Path(args.out) / tag / "metrics.json")


if __name__ == "__main__":
    main()
