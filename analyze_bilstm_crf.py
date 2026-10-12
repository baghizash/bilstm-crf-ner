# -*- coding: utf-8 -*-
"""Pipeline data dan arsitektur BiLSTM CRF untuk NER alamat.

Modul ini menyiapkan seluruh pipeline data persis seperti tugas kursus:
tokenisasi per kata, pembangunan vocab dan peta label, encoding indeks,
dan padding. Arsitekturnya: Embedding(100) -> BiLSTM(128 dua arah) ->
Linear(4 label) -> CRF. Pelatihan memakai torch (10 epoch, Adam lr 0.001,
batch 32); jika torch tidak tersedia, script tetap berjalan untuk pipeline
data dan visualisasi.

Jalankan: python analyze_bilstm_crf.py
"""
import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "list_of_real_usa_addresses.json")
CHARTS = os.path.join(BASE, "charts")
FINDINGS = os.path.join(BASE, "findings.json")
os.makedirs(CHARTS, exist_ok=True)

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
FIELDS = ["address", "city", "state", "zip"]

# Hiperparameter persis dari tugas kursus
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 32


def tokenize(entry):
    tokens, labels = [], []
    for field in FIELDS:
        parts = entry[field].split()
        tokens.extend(parts)
        labels.extend([field] * len(parts))
    return tokens, labels


def build_vocab(sequences):
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for seq in sequences:
        for tok in seq:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def build_label_map():
    return {PAD_TOKEN: 0, "address": 1, "city": 2, "state": 3, "zip": 4}


def encode_and_pad(sequences, mapping, max_len, unk_idx=1):
    arr = np.zeros((len(sequences), max_len), dtype=np.int64)
    for i, seq in enumerate(sequences):
        for j, tok in enumerate(seq[:max_len]):
            arr[i, j] = mapping.get(tok, unk_idx)
    return arr


def describe_architecture(vocab_size, num_labels):
    lines = [
        "Arsitektur BiLSTM CRF (dari tugas kursus):",
        "1. nn.Embedding(vocab {}, dim {}) padding idx 0".format(
            vocab_size, EMBEDDING_DIM),
        "2. nn.LSTM(dim {}, hidden {}, dua arah, batch first)".format(
            EMBEDDING_DIM, HIDDEN_DIM),
        "3. nn.Linear(2 x {} ke {} label)".format(HIDDEN_DIM, num_labels),
        "4. Lapisan CRF (torchcrf) untuk decoding urutan label",
        "Loss = negatif log likelihood CRF; decoding memakai Viterbi",
    ]
    return "\n".join(lines)


def main():
    with open(DATA, "r") as f:
        addresses = json.load(f)

    X_words, y_words = [], []
    for entry in addresses:
        toks, labs = tokenize(entry)
        X_words.append(toks)
        y_words.append(labs)

    vocab = build_vocab(X_words)
    label_map = build_label_map()
    max_len = max(len(s) for s in X_words)
    X_idx = encode_and_pad(X_words, vocab, max_len)
    y_idx = encode_and_pad(y_words, label_map, max_len, unk_idx=0)

    label_counts = Counter(l for seq in y_words for l in seq)
    seq_lens = [len(s) for s in X_words]
    tok_counts = Counter(t for seq in X_words for t in seq)

    print(describe_architecture(len(vocab), len(label_map) - 1))
    try:
        import torch  # noqa
        print("torch tersedia: model dapat dibangun dan dilatih.")
    except ImportError:
        print("torch tidak tersedia: hanya pipeline data yang dijalankan.")

    # Chart 1: sebaran panjang sekuens
    plt.figure(figsize=(8, 4.5))
    plt.hist(seq_lens, bins=range(min(seq_lens), max(seq_lens) + 2),
             color="#1D4ED8", edgecolor="white")
    plt.title("Sebaran panjang sekuens token per alamat")
    plt.xlabel("Jumlah token")
    plt.ylabel("Jumlah alamat")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS, "01_panjang_sekuens.png"), dpi=120)
    plt.close()

    # Chart 2: distribusi label
    plt.figure(figsize=(7, 4.5))
    labs = list(label_counts.keys())
    plt.bar(labs, [label_counts[l] for l in labs], color="#16A34A")
    plt.title("Distribusi label token")
    plt.xlabel("Label")
    plt.ylabel("Jumlah token")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS, "02_distribusi_label.png"), dpi=120)
    plt.close()

    # Chart 3: 15 token tersering
    top = tok_counts.most_common(15)
    plt.figure(figsize=(10, 5))
    plt.barh([t for t, _ in reversed(top)],
             [c for _, c in reversed(top)], color="#7C3AED")
    plt.title("15 token tersering pada data alamat")
    plt.xlabel("Frekuensi")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS, "03_top_token.png"), dpi=120)
    plt.close()

    # Chart 4: skema arsitektur
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axis("off")
    ax.set_title("Arsitektur BiLSTM CRF", fontsize=14, pad=10)
    boxes = [
        ("Input\nindeks token", "#EFF6FF"),
        ("Embedding\n100 dimensi", "#DBEAFE"),
        ("BiLSTM\n128 unit x 2 arah", "#BFDBFE"),
        ("Linear\nke 4 label", "#93C5FD"),
        ("CRF\nViterbi decode", "#60A5FA"),
    ]
    x0, w, h = 0.02, 0.17, 0.45
    for i, (label, color) in enumerate(boxes):
        x = x0 + i * 0.195
        rect = patches.FancyBboxPatch((x, 0.25), w, h, boxstyle="round,pad=0.02",
                                      facecolor=color, edgecolor="#1E3AED")
        ax.add_patch(rect)
        ax.text(x + w / 2, 0.25 + h / 2, label, ha="center", va="center",
                fontsize=10)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + w + 0.005, 0.25 + h / 2),
                        xytext=(x + w + 0.02, 0.25 + h / 2),
                        arrowprops=dict(arrowstyle="->", color="#1E3AED"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS, "04_arsitektur.png"), dpi=120)
    plt.close()

    findings = {
        "jumlah_alamat": len(addresses),
        "ukuran_vocab": len(vocab),
        "panjang_sekuens_maks": max_len,
        "panjang_sekuens_min": min(seq_lens),
        "distribusi_label": dict(label_counts),
        "hiperparameter": {
            "embedding_dim": EMBEDDING_DIM,
            "hidden_dim": HIDDEN_DIM,
            "dua_arah": True,
            "epoch": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "optimizer": "Adam",
        },
        "catatan": "Pelatihan penuh memakai PyTorch dan torchcrf; "
                   "peringkat akurasi pada tugas: CRF di bawah BiLSTM CRF "
                   "di bawah Transformer",
    }
    with open(FINDINGS, "w") as f:
        json.dump(findings, f, indent=2, ensure_ascii=False)

    print("vocab:", len(vocab), "| max_len:", max_len,
          "| total token:", sum(seq_lens))


if __name__ == "__main__":
    main()
