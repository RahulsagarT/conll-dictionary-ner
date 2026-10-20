# Automatic Dictionary Construction for NER (CoNLL-2003)

A pure-Python, dictionary-based Named Entity Recognition baseline.

1. Extract annotated **PERSON**, **LOCATION** and **ORGANIZATION** phrases from the CoNLL-2003 training set
2. Build three unique entity dictionaries
3. Tag 200 unseen test sentences using **longest-phrase matching** (greedy, left to right)
4. Evaluate with **entity-level** precision, recall and F1

Ambiguous phrases (present in more than one dictionary) are assigned the type they had most often in training.

## Project structure

```
conll-dictionary-ner/
├── dictionary_ner.py   # main program
├── download_data.py    # fetches CoNLL-2003 train/test into data/
├── requirements.txt    # no third-party dependencies
├── results/
│   ├── results.txt     # dictionary sizes + scores
│   └── predictions.txt # per-sentence predicted vs gold entities
├── .gitignore
├── LICENSE
└── README.md
```

## Usage

```bash
python download_data.py      # downloads data/train.txt and data/test.txt
python dictionary_ner.py     # builds dictionaries, evaluates, writes results/
```

Requires Python 3.8+ and nothing else.

## Results (200 random test sentences, seed 42)

**Dictionary sizes:** PER 3,614 · LOC 1,332 · ORG 2,403

Gold entities: 263 · Predicted: 190 · Correct: 133

| Type | Precision | Recall | F1 |
|------|-----------|--------|-----|
| PER  | 0.600 | 0.158 | 0.250 |
| LOC  | 0.677 | 0.798 | 0.733 |
| ORG  | 0.764 | 0.618 | 0.683 |
| **Overall** | **0.700** | **0.506** | **0.587** |

## Observations

- **PER recall is very low**: person names rarely repeat across documents, so a lookup table cannot cover unseen names.
- **LOC does best**: places like "France" or "Japan" recur constantly.
- **Precision is limited by ambiguity**: sports text labels city names as teams (ORG), and noisy entries such as "The" can end up in a dictionary.
- Ideas for improvement: frequency thresholds, stopword filtering, capitalization/context rules, or a statistical model (CRF / BiLSTM / transformer).

## Data note

CoNLL-2003 is built on licensed Reuters text, so the data files are **not** included in this repo (`data/` is git-ignored). `download_data.py` fetches a public mirror for local use.

## License

MIT
