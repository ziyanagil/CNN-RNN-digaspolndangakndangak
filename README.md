# IF3270 Pembelajaran Mesin — Tugas Besar 2

## Convolutional Neural Network dan Recurrent Neural Network

**Kelompok 62**

| NIM      | Nama                    |
| -------- | ----------------------- |
| 13523113 | Kefas Kurnia Jonathan   |
| 13523118 | Farrel Athalla Putra    |
| 13622076 | Ziyan Agil Nur Ramadhan |

---

## Deskripsi

Repositori ini berisi implementasi forward propagation **CNN**, **SimpleRNN**, dan **LSTM** _from scratch_ menggunakan NumPy, beserta pipeline _image captioning_ encoder-decoder berbasis arsitektur **pre-inject**

**Task CNN** : Image classification pada dataset Intel Image Classification (~25.000 gambar, 6 kelas). Eksperimen variasi jumlah layer, jumlah filter, ukuran filter, dan jenis pooling.

**Task RNN/LSTM** : Image captioning pada dataset Flickr8k (8.092 gambar, 5 caption/gambar). Decoder SimpleRNN dan LSTM dilatih dan dibandingkan.

**Bonus** : Grad-CAM, Beam Search, Init-inject, Batch Inference, Backward Propagation.

---

## Struktur Repositori

```
CNN-RNN-digaspolndangakndangak/
├── data/
│   ├── intel/                      # Dataset Intel Image Classification
│   └── flickr8k/                   # Dataset Flickr8k
├── features/                       # Cache fitur CNN dan vocab (di-generate saat run)
├── models/
│   ├── cnn/                        # Bobot model CNN tersimpan
│   ├── rnn/                        # Bobot model RNN tersimpan
│   └── lstm/                       # Bobot model LSTM tersimpan
├── doc/                            # Laporan PDF
├── requirements.txt
└── src/
    ├── shared/
    │   ├── image_utils.py          # Utility load/batch gambar dan ekstraksi fitur
    │   ├── caption_utils.py        # Tokenisasi, vocab, encoding caption
    │   ├── dense.py                # Dense layer from scratch (reuse Tubes 1)
    │   └── metrics.py              # Macro F1, BLEU-4, METEOR
    ├── cnn/
    │   ├── layers/
    │   │   ├── activations.py      # ReLU, Softmax
    │   │   ├── conv2d.py           # Conv2D from scratch (shared parameter)
    │   │   ├── locally_connected.py# LocallyConnected2D from scratch (non-shared)
    │   │   ├── pooling.py          # MaxPooling, AvgPooling, GlobalMax, GlobalAvg
    │   │   └── flatten.py          # Flatten
    │   ├── model.py                # CNNScratch wrapper
    │   ├── train_keras.py          # Builder dan training loop Keras
    │   └── evaluate.py             # Evaluasi Keras vs scratch
    ├── rnn/
    │   ├── layers/
    │   │   ├── activations.py      # tanh, sigmoid, softmax
    │   │   ├── embedding.py        # Embedding layer from scratch
    │   │   └── simple_rnn_cell.py  # SimpleRNNCell dan SimpleRNN from scratch
    │   ├── model.py                # RNNDecoder (generate_caption, generate_captions_batch)
    │   └── train_keras.py          # Builder dan training loop Keras RNN decoder
    ├── lstm/
    │   ├── layers/
    │   │   ├── activations.py
    │   │   ├── embedding.py
    │   │   └── lstm_cell.py        # LSTMCell dan LSTM from scratch (+ batch support)
    │   ├── model.py                # LSTMDecoder (generate_caption, generate_captions_batch)
    │   └── train_keras.py          # Builder dan training loop Keras LSTM decoder
    ├── bonus/
    │   ├── gradcam.py              # Grad-CAM dan intermediate feature map visualization
    │   ├── beam_search.py          # Beam Search decoder (k=3 dan k=5)
    │   ├── init_inject.py          # Arsitektur init-inject RNN dan LSTM
    │   └── backprop/
    │       ├── conv2d_backward.py  # Backward propagation Conv2D
    │       ├── rnn_backward.py     # BPTT SimpleRNN
    │       └── lstm_backward.py    # BPTT LSTM
    └── notebooks/
        ├── 01_CNN_training.ipynb           # Training 16 variasi CNN + LocallyConnected2D
        ├── 02_CNN_experiments.ipynb        # Analisis hyperparameter CNN
        ├── 03_CNN_scratch_eval.ipynb       # Evaluasi Keras vs CNN from scratch
        ├── 04_CNN_bonus_gradcam.ipynb      # [BONUS] Grad-CAM dan feature maps
        ├── 05_Feature_extraction.ipynb     # Ekstraksi fitur Flickr8k (InceptionV3)
        ├── 06_Caption_preprocessing.ipynb  # Preprocessing caption dan bangun vocab
        ├── 07_RNN_training.ipynb           # Training 6 variasi SimpleRNN decoder
        ├── 08_RNN_experiments.ipynb        # Evaluasi dan analisis RNN
        ├── 09_LSTM_training.ipynb          # Training 6 variasi LSTM decoder
        ├── 10_LSTM_experiments.ipynb       # Evaluasi dan analisis LSTM
        ├── 11_RNN_vs_LSTM_comparison.ipynb # Perbandingan RNN vs LSTM (analisis utama)
        ├── 12_Bonus_beam_search.ipynb      # [BONUS] Beam Search vs greedy decoding
        └── 13_Bonus_init_inject.ipynb      # [BONUS] Arsitektur init-inject
```

---

## Setup

### Prasyarat

- Python 3.10+
- (Opsional) GPU dengan CUDA untuk mempercepat training Keras

### Instalasi

```bash
git clone <repo-url>
cd CNN-RNN-digaspolndangakndangak
pip install -r requirements.txt
```

### Download Dataset

1. **Intel Image Classification** — download dari [Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification), ekstrak ke `data/intel/` sehingga strukturnya:

   ```
   data/intel/seg_train/
   data/intel/seg_test/
   data/intel/seg_pred/
   ```

2. **Flickr8k** — download dari [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k), ekstrak ke `data/flickr8k/` sehingga strukturnya:
   ```
   data/flickr8k/Images/
   data/flickr8k/captions.txt
   ```

---

## Cara Menjalankan

Jalankan notebook secara berurutan sesuai nomornya. Semua notebook dapat dijalankan lokal maupun di Kaggle.

### CNN

| Notebook                     | Keterangan                                                                        |
| ---------------------------- | --------------------------------------------------------------------------------- |
| `01_CNN_training.ipynb`      | Training 16 variasi Conv2D + 1 LocallyConnected2D. Simpan bobot ke `models/cnn/`. |
| `02_CNN_experiments.ipynb`   | Analisis pengaruh jumlah layer, filter, kernel size, pooling.                     |
| `03_CNN_scratch_eval.ipynb`  | Validasi forward pass from scratch vs Keras, hitung macro F1.                     |
| `04_CNN_bonus_gradcam.ipynb` | Visualisasi Grad-CAM dan intermediate feature maps.                               |

### RNN / LSTM

| Notebook                          | Keterangan                                                                                                                                      |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `05_Feature_extraction.ipynb`     | Ekstraksi fitur CNN (InceptionV3) untuk semua gambar Flickr8k. Cache ke `features/`. Jalankan sekali saja.                                      |
| `06_Caption_preprocessing.ipynb`  | Tokenisasi caption, bangun vocab, encode, simpan ke `features/`.                                                                                |
| `07_RNN_training.ipynb`           | Training 6 variasi SimpleRNN decoder. Simpan bobot ke `models/rnn/`.                                                                            |
| `09_LSTM_training.ipynb`          | Training 6 variasi LSTM decoder. Simpan bobot ke `models/lstm/`.                                                                                |
| `08_RNN_experiments.ipynb`        | Evaluasi RNN: BLEU-4, METEOR, Keras vs Scratch, qualitative analysis.                                                                           |
| `10_LSTM_experiments.ipynb`       | Evaluasi LSTM: BLEU-4, METEOR, Keras vs Scratch, batch inference demo.                                                                          |
| `11_RNN_vs_LSTM_comparison.ipynb` | **Perbandingan utama** RNN vs LSTM: semua 12 variasi, Keras vs Scratch, qualitative 10 gambar, max caption length, analisis vanishing gradient. |
| `12_Bonus_beam_search.ipynb`      | Beam Search (k=3, k=5) vs greedy decoding.                                                                                                      |
| `13_Bonus_init_inject.ipynb`      | Arsitektur init-inject sebagai alternatif pre-inject.                                                                                           |

---

## Pembagian Tugas

| NIM      | Nama                    | Domain Utama                                                           | Bonus                                  |
| -------- | ----------------------- | ---------------------------------------------------------------------- | -------------------------------------- |
| 13523113 | Kefas Kurnia Jonathan   | CNN : layers from scratch, Keras training, experiments, evaluasi       | Grad-CAM + feature map visualization   |
| 13523118 | Farrel Athalla Putra    | SimpleRNN : layers from scratch, Keras training, experiments, evaluasi | Beam Search + Init-inject              |
| 13622076 | Ziyan Agil Nur Ramadhan | LSTM : layers from scratch, Keras training, experiments, evaluasi      | Batch inference + Backward propagation |

---

## Referensi

- Vinyals et al. (2015) — Show and Tell: A Neural Image Caption Generator
- Tanti et al. (2017) — Where to put the Image in an Image Caption Generator
- Selvaraju et al. (2016) — Grad-CAM
- [d2l.ai](https://d2l.ai) — CNN, RNN, LSTM, Image Captioning chapters
- Intel Image Classification dataset — Kaggle
- Flickr8k dataset — Kaggle
