# Skript na zistenie zakladnych statistik suroveho datasetu, ktorymi su:
# pocet viet, 
# pocet unikatnych slov (tokenov), 
# distribucia tried sentimentu,
# priemerny pocet tokenov vo vete,
# priemerny pocet znakov vo vete.

import re
from collections import Counter, defaultdict
from statistics import mean, median

FILE_PATH = "data/financialphrasebank/raw/Sentences_50Agree.txt"

TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)

def tokenize(text):
    return TOKEN_RE.findall(text.lower())

def parse_line(line):
    line = line.strip()
    if not line:
        return None

    if "@" not in line:
        return (line, None)

    text, label = line.rsplit("@", 1)
    return text.strip(), label.strip().lower()

texts = []
labels = []
token_counts = []
char_counts = []

per_label_tokens = defaultdict(list)
per_label_chars = defaultdict(list)

global_vocab = set()

with open(FILE_PATH, "r", encoding="utf-8", errors="ignore") as f:
    for raw in f:
        parsed = parse_line(raw)
        if not parsed:
            continue

        text, label = parsed
        if label not in {"positive", "negative", "neutral"}:
            continue

        tokens = tokenize(text)

        texts.append(text)
        labels.append(label)
        token_counts.append(len(tokens))
        char_counts.append(len(text))

        per_label_tokens[label].append(len(tokens))
        per_label_chars[label].append(len(text))
        global_vocab.update(tokens)

print("\nRAW DATASET STATS")
print("_________________")
print(f"Total sentences: {len(texts)}")
print(f"Global vocabulary size: {len(global_vocab)}")

print("\nLABEL DISTRIBUTION")
print("_________________")
label_counts = Counter(labels)
for lab in ["positive", "negative", "neutral"]:
    cnt = label_counts.get(lab, 0)
    pct = (cnt / len(labels)) * 100 if labels else 0
    print(f"{lab:>8}: {cnt:6} ({pct:5.1f}%)")

print("\nAVG NUMBER OF TOKENS(~WORDS) IN SENTENCE")
print("_________________")
for lab in ["positive", "negative", "neutral"]:
    lengths = per_label_tokens[lab]
    print(
        f"{lab:>8} | "
        f"avg={mean(lengths):.2f}"
    )

print("\nAVG NUMBER OF CHARS IN SENTENCE")
print("_________________")
for lab in ["positive", "negative", "neutral"]:
    lengths = per_label_chars[lab]
    print(
        f"{lab:>8} | "
        f"avg={mean(lengths):.2f}"
    )

