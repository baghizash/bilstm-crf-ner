# NER Alamat dengan BiLSTM CRF

Pipeline data dan arsitektur Bidirectional LSTM CRF untuk mengenali field
alamat (address, city, state, zip) pada tiap token. BiLSTM membaca konteks
dari dua arah lalu lapisan CRF memilih urutan label terbaik dengan decoding
Viterbi.

Proyek ini adalah Module 3 dari kursus Applied Information Extraction in
Python (University of Michigan).

## Data

Folder `data` berisi `list_of_real_usa_addresses.json`: 234 alamat nyata di
AS. Pipeline mengubah tiap alamat menjadi sekuens token (total 1550 token,
panjang 4 sampai 9 token), membangun vocab 909 kata, lalu encoding ke indeks
dengan padding sampai panjang maksimum.

## Yang dikerjakan

1.  Pipeline data lengkap: tokenisasi per kata, vocab dengan token PAD dan
    UNK, peta label (PAD, address, city, state, zip), encoding indeks, dan
    padding. Lihat `charts/01_panjang_sekuens.png` dan
    `charts/03_top_token.png`.
2.  Arsitektur: Embedding 100 dimensi, BiLSTM 128 unit dua arah, Linear ke 4
    label, lapisan CRF. Lihat `charts/04_arsitektur.png`.
3.  Konfigurasi latih dari tugas kursus: 10 epoch, optimizer Adam learning
    rate 0.001, batch 32, loss negatif log likelihood CRF.
4.  Pada tugas kursus, peringkat akurasinya: CRF di bawah BiLSTM CRF di bawah
    Transformer. Model ini menangkap konteks dua arah yang tidak bisa
    ditangkap CRF berbasis fitur manual.

## Cara menjalankan

```bash
python analyze_bilstm_crf.py
```

Script ini menjalankan pipeline data, menyimpan chart ke folder `charts`,
dan menulis ringkasan angka ke `findings.json`. Pelatihan penuh memakai
PyTorch dan torchcrf; tanpa keduanya, pipeline data tetap berjalan.

## Tools

Python, PyTorch, torchcrf, NumPy, matplotlib.
