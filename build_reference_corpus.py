"""
Build Reference Emotion Corpus
================================
Uses dair-ai/emotion (HuggingFace) to populate outputs/exp1/reference_corpus.json.

dair-ai/emotion labels: sadness, joy, love, anger, fear, surprise
We map these + add extra emotions (frustration, confusion, etc.) from filtered subsets.

Usage:
    python build_reference_corpus.py

Output:
    outputs/exp1/reference_corpus.json  — {emotion: [text, ...], ...} with 30 texts each
"""

import json
import random
from pathlib import Path
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
N_PER_EMOTION   = 30          # stories per emotion (30 is sufficient for direction extraction)
MIN_CHARS       = 80          # filter out very short tweets
MAX_CHARS       = 600         # filter out very long texts
SEED            = 42
OUTDIR          = Path("outputs/exp1")
OUT_PATH        = OUTDIR / "reference_corpus.json"

OUTDIR.mkdir(parents=True, exist_ok=True)
random.seed(SEED)

# ── dair-ai/emotion label map ─────────────────────────────────────────────────
# Dataset labels: 0=sadness, 1=joy, 2=love, 3=anger, 4=fear, 5=surprise
DAIR_LABEL_TO_NAME = {0: "sadness", 1: "joy", 2: "love",
                      3: "anger",   4: "fear", 5: "surprise"}

# How we map dair labels → our 10 research emotions
# Some emotions are close matches, others we derive from combined filters
DIRECT_MAPPING = {
    "sadness":     ["frustration", "desperation"],   # sadness texts ~ frustration/desperation
    "anger":       ["frustration"],                   # anger texts also ~ frustration
    "joy":         ["satisfaction", "excitement"],    # joy ~ satisfaction/excitement
    "love":        ["calm"],                          # love/contentment ~ calm
    "fear":        ["anxiety"],                       # fear ~ anxiety
    "surprise":    ["curiosity", "excitement"],       # surprise ~ curiosity
}

# Emotions we need that dair doesn't directly cover:
# - confidence: joy texts filtered for "confident", "certain", "sure", "know"
# - confusion:  sadness/fear texts filtered for "confused", "lost", "don't understand"
# - boredom:    sadness texts filtered for "bored", "tedious", "routine", "monoton"
# - calm:       love texts + joy texts filtered for "peaceful", "relax", "serene"

KEYWORD_FILTERS = {
    "confidence": (["joy"],       ["confident", "certain", "sure", "know i", "i know", "clear"]),
    "confusion":  (["sadness", "fear"], ["confused", "confusing", "lost", "understand", "unclear", "don't get"]),
    "boredom":    (["sadness"],   ["bored", "boring", "tedious", "routine", "monoton", "dull"]),
    "calm":       (["love", "joy"], ["peaceful", "peaceful", "calm", "relax", "serene", "tranquil", "quiet"]),
    "curiosity":  (["surprise"],  ["curious", "wonder", "interesting", "fascin", "what if", "discover"]),
}

# ── Load dataset ──────────────────────────────────────────────────────────────
try:
    from datasets import load_dataset
    print("Loading dair-ai/emotion ...")
    ds = load_dataset("dair-ai/emotion", split="train+validation+test")
    print(f"  Loaded {len(ds)} examples")
except Exception as e:
    print(f"ERROR loading dair-ai/emotion: {e}")
    print("Make sure you have: pip install datasets")
    raise

# ── Bucket by label ───────────────────────────────────────────────────────────
buckets = defaultdict(list)
for ex in ds:
    label_name = DAIR_LABEL_TO_NAME[ex["label"]]
    text = ex["text"].strip()
    if MIN_CHARS <= len(text) <= MAX_CHARS:
        buckets[label_name].append(text)

for name, texts in buckets.items():
    print(f"  {name}: {len(texts)} usable texts")

# ── Build corpus ──────────────────────────────────────────────────────────────
corpus = defaultdict(list)

def sample_from(source_labels, n, keyword_filter=None):
    """Sample n texts from given dair labels, optionally filtered by keywords."""
    pool = []
    for lbl in source_labels:
        pool.extend(buckets.get(lbl, []))
    if keyword_filter:
        pool = [t for t in pool if any(kw in t.lower() for kw in keyword_filter)]
    if not pool:
        print(f"  WARNING: empty pool for {source_labels} + {keyword_filter}")
        return []
    random.shuffle(pool)
    return pool[:n]

# Direct mappings
DIRECT_SOURCES = {
    "frustration":  (["anger", "sadness"], None),
    "desperation":  (["sadness", "fear"],  None),
    "satisfaction": (["joy", "love"],      None),
    "excitement":   (["joy", "surprise"],  None),
    "anxiety":      (["fear", "sadness"],  None),
}

for emotion, (labels, kws) in DIRECT_SOURCES.items():
    texts = sample_from(labels, N_PER_EMOTION, kws)
    corpus[emotion] = texts
    print(f"  {emotion}: {len(texts)} texts added")

# Keyword-filtered mappings
for emotion, (labels, kws) in KEYWORD_FILTERS.items():
    texts = sample_from(labels, N_PER_EMOTION, kws)
    if len(texts) < N_PER_EMOTION:
        # Fallback: relax keyword filter, just use source labels
        extra = sample_from(labels, N_PER_EMOTION - len(texts), None)
        # Avoid duplicates
        existing = set(texts)
        for t in extra:
            if t not in existing:
                texts.append(t)
                existing.add(t)
                if len(texts) >= N_PER_EMOTION:
                    break
    corpus[emotion] = texts[:N_PER_EMOTION]
    print(f"  {emotion}: {len(corpus[emotion])} texts added")

# ── Verify all 10 emotions present ───────────────────────────────────────────
TARGET_EMOTIONS = [
    "frustration", "confusion", "confidence", "anxiety",
    "curiosity", "boredom", "satisfaction", "desperation", "calm", "excitement",
]

print("\n-- Corpus summary --")
missing = []
for emotion in TARGET_EMOTIONS:
    n = len(corpus.get(emotion, []))
    status = "OK " if n >= 10 else "LOW"
    print(f"  [{status}] {emotion}: {n} texts")
    if n == 0:
        missing.append(emotion)

if missing:
    print(f"\nWARNING: {missing} have 0 texts — validation will skip these emotions.")

# ── Save ──────────────────────────────────────────────────────────────────────
# Only save the 10 target emotions in the correct key order
output = {e: corpus.get(e, []) for e in TARGET_EMOTIONS}
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

total = sum(len(v) for v in output.values())
print(f"\nSaved {total} texts ({len(output)} emotions) -> {OUT_PATH}")
print("\nNote: dair-ai/emotion covers tweets (short, informal). This is fine")
print("for direction extraction — the geometry captures the *concept*,")
print("not the genre. The 5-level validation will confirm transfer.")
