"""
Automatic Dictionary Construction for NER  --  CoNLL-2003
=========================================================
1. Extract annotated PERSON / LOCATION / ORGANIZATION phrases from the training set
2. Build three unique entity dictionaries
3. Tag 200 unseen test sentences with longest-phrase (greedy, left-to-right) matching
4. Report entity-level precision, recall and F1

Usage:  python dictionary_ner.py [data/train.txt] [data/test.txt]
Files are in CoNLL format:  token POS chunk NER-tag   (blank line = sentence break)
"""
import os
import random
import sys
from collections import Counter, defaultdict

TYPES = ("PER", "LOC", "ORG")
N_TEST = 200
SEED = 42


# ---------------------------------------------------------------- data loading
def read_conll(path):
    """Return list of sentences; each is (tokens, tags)."""
    sents, toks, tags = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("-DOCSTART-"):
                if toks:
                    sents.append((toks, tags))
                    toks, tags = [], []
                continue
            parts = line.split()
            toks.append(parts[0])
            tags.append(parts[-1])
    if toks:
        sents.append((toks, tags))
    return sents


def get_entities(tags):
    """BIO/IOB1 tags -> list of (start, end_exclusive, type). Handles both schemes."""
    ents, start, cur = [], None, None
    for i, t in enumerate(tags + ["O"]):
        if t == "O":
            label, begin = None, False
        else:
            prefix, label = t.split("-", 1)
            begin = prefix == "B"
        if cur is not None and (label != cur or begin):
            ents.append((start, i, cur))
            cur = None
        if label is not None and cur is None:
            start, cur = i, label
    return ents


# ------------------------------------------------------- dictionary construction
def build_dictionaries(train_sents):
    """Return dicts {type: set(phrases)} and phrase->Counter(type) for tie-breaking."""
    dicts = {t: set() for t in TYPES}
    phrase_type_counts = defaultdict(Counter)
    for toks, tags in train_sents:
        for s, e, typ in get_entities(tags):
            if typ not in TYPES:        # MISC is not required
                continue
            phrase = tuple(toks[s:e])
            dicts[typ].add(phrase)
            phrase_type_counts[phrase][typ] += 1
    return dicts, phrase_type_counts


# ---------------------------------------------------------- longest-phrase match
def predict(tokens, phrase_type_counts, max_len):
    """Greedy left-to-right longest match. Ambiguous phrases -> most frequent type."""
    preds, i, n = [], 0, len(tokens)
    while i < n:
        matched = False
        for L in range(min(max_len, n - i), 0, -1):
            phrase = tuple(tokens[i:i + L])
            if phrase in phrase_type_counts:
                typ = phrase_type_counts[phrase].most_common(1)[0][0]
                preds.append((i, i + L, typ))
                i += L
                matched = True
                break
        if not matched:
            i += 1
    return preds


# -------------------------------------------------------------------- evaluation
def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main():
    train_path = sys.argv[1] if len(sys.argv) > 1 else "data/train.txt"
    test_path = sys.argv[2] if len(sys.argv) > 2 else "data/test.txt"

    train = read_conll(train_path)
    test_all = read_conll(test_path)
    random.seed(SEED)
    test = random.sample(test_all, N_TEST)           # 200 unseen sentences

    dicts, ptc = build_dictionaries(train)
    max_len = max(len(p) for p in ptc)

    print("=" * 70)
    print("DICTIONARY SIZES (unique phrases from training data)")
    print("=" * 70)
    for t in TYPES:
        print(f"  {t:<4}: {len(dicts[t]):>6}")
    print(f"  Longest phrase: {max_len} tokens")
    for t in TYPES:
        ex = sorted(dicts[t], key=lambda p: -len(p))[:3]
        print(f"  e.g. {t}: {[' '.join(p) for p in ex]}")

    tp, fp, fn = Counter(), Counter(), Counter()
    predicted_all, correct_all, per_sentence = [], [], []
    for idx, (toks, tags) in enumerate(test):
        gold = {e for e in get_entities(tags) if e[2] in TYPES}
        pred = set(predict(toks, ptc, max_len))
        for e in pred:
            (tp if e in gold else fp)[e[2]] += 1
        for e in gold - pred:
            fn[e[2]] += 1
        fmt = lambda e: (" ".join(toks[e[0]:e[1]]), e[2])
        predicted_all += [fmt(e) for e in sorted(pred)]
        correct_all += [fmt(e) for e in sorted(pred & gold)]
        per_sentence.append((idx, " ".join(toks), sorted(map(fmt, pred)), sorted(map(fmt, gold))))

    print("\n" + "=" * 70)
    print(f"RESULTS ON {N_TEST} UNSEEN TEST SENTENCES")
    print("=" * 70)
    print(f"  Gold entities (PER/LOC/ORG) : {sum(tp.values()) + sum(fn.values())}")
    print(f"  Predicted entities          : {len(predicted_all)}")
    print(f"  Correctly identified        : {len(correct_all)}")

    print(f"\n  {'Type':<8}{'P':>8}{'R':>8}{'F1':>8}{'TP':>6}{'FP':>6}{'FN':>6}")
    for t in TYPES:
        p, r, f = prf(tp[t], fp[t], fn[t])
        print(f"  {t:<8}{p:8.4f}{r:8.4f}{f:8.4f}{tp[t]:6}{fp[t]:6}{fn[t]:6}")
    p, r, f = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    print(f"  {'OVERALL':<8}{p:8.4f}{r:8.4f}{f:8.4f}{sum(tp.values()):6}{sum(fp.values()):6}{sum(fn.values()):6}")

    print("\nSample predicted entities (first 25):")
    for e in predicted_all[:25]:
        print("   ", e)
    print("\nSample correctly identified entities (first 25):")
    for e in correct_all[:25]:
        print("   ", e)

    os.makedirs("results", exist_ok=True)
    with open("results/predictions.txt", "w", encoding="utf-8") as out:
        for idx, sent, pred, gold in per_sentence:
            out.write(f"[{idx}] {sent}\n  PRED: {pred}\n  GOLD: {gold}\n\n")
    print("\nPer-sentence predictions saved to results/predictions.txt")


if __name__ == "__main__":
    main()
