# Data

The experiments use the Reddit mental-illness benchmark released by **Murarka, Radhakrishnan and Ravichandran (2021)**, *Classification of mental illnesses on social media using RoBERTa* (LOUHI workshop). The dataset is not redistributed here. Get it from the original authors and follow their terms of use.

## Expected format

Place one CSV per split in a folder, for example `data/reddit/`:

```
data/reddit/
├── train.csv
├── dev.csv
└── test.csv
```

| Column | Content |
|---|---|
| `title` | Post title |
| `post` | Post body |
| `class_name` | `adhd`, `anxiety`, `bipolar`, `depression`, `ptsd` or `none` (case-insensitive) |

A numeric `class_id` column (0–5, in the order above) is accepted if `class_name` is missing. File names `train_data.csv` and `.tsv` also work.

## Sample

[`sample/`](sample/) contains a handful of **invented** posts in this format so the code and tests run without the real data. They are not taken from Reddit or from the benchmark.
