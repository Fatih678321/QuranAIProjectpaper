# QuranAIProjectPaper

> Fine-tuning OpenAI Whisper Tiny untuk mengenali huruf Hijaiyah terisolasi dengan Fathah.

QuranAIProjectPaper adalah proyek penelitian **speech recognition** yang
mengeksplorasi apakah model Whisper yang telah dilatih sebelumnya dapat
diadaptasi untuk mengenali pelafalan huruf Hijaiyah yang pendek dan terisolasi.

Eksperimen saat ini mencakup **28 huruf Hijaiyah**, masing-masing diucapkan
dengan **satu harakat: Fathah (َ)**.

Proyek ini dibuat sebagai **proof of concept** dan sebagai dasar untuk
pengembangan lebih lanjut menuju sistem pembelajaran serta evaluasi bacaan
Al-Qur'an.

---

## 📖 Gambaran Umum

Model speech recognition pada umumnya dirancang untuk mengenali
kata, frasa, dan kalimat. Proyek ini mengeksplorasi tugas yang lebih spesifik:

> **Apakah model speech recognition yang telah dilatih sebelumnya dapat
> di-fine-tune untuk membedakan pelafalan huruf Hijaiyah yang pendek dan
> terisolasi?**

Alih-alih melatih model AI dari awal, proyek ini menggunakan
**OpenAI Whisper Tiny** sebagai model speech recognition yang telah
dilatih sebelumnya, kemudian melakukan **fine-tuning** menggunakan dataset
suara Hijaiyah yang dibuat secara khusus.

### Cakupan Eksperimen Saat Ini

- **28 huruf Hijaiyah**
- **1 harakat: Fathah**
- **1.120 rekaman audio**
- **40 rekaman per huruf**
- **Audio WAV 16 kHz**
- **Whisper Tiny**
- **5-fold stratified cross-validation**
- **Character Error Rate (CER)** sebagai metrik evaluasi utama

> Jumlah 1.120 rekaman didasarkan pada 28 kelas dengan 40 rekaman per kelas.
> Detail jumlah file aktual sebaiknya tetap disesuaikan dengan manifest dataset
> yang digunakan dalam eksperimen.

---

# 🎯 Tujuan Penelitian

Tujuan utama proyek ini adalah:

- Mempelajari proses **fine-tuning Whisper** untuk tugas speech recognition yang spesifik.
- Membangun model yang mampu mengenali pelafalan huruf Hijaiyah secara terisolasi.
- Memahami keseluruhan pipeline machine learning untuk speech recognition.
- Mengevaluasi kemampuan Whisper dalam beradaptasi dengan dataset suara Arab yang kecil dan spesifik.
- Menjadikan proyek ini sebagai dasar untuk pengembangan sistem pembelajaran Al-Qur'an dan pengenalan bacaan di masa mendatang.

Proyek ini **belum ditujukan untuk mengenali ayat Al-Qur'an secara lengkap**.

Tugas saat ini sengaja dibatasi pada pengenalan huruf Hijaiyah secara terisolasi
dengan harakat Fathah.

---

# 🗂️ Dataset

Dataset terdiri dari:

**28 huruf × 40 rekaman = 1.120 rekaman audio**

Setiap rekaman berisi pelafalan satu huruf Hijaiyah secara terisolasi
dengan harakat Fathah.

### Karakteristik Audio

- **Format:** WAV
- **Sample rate:** 16 kHz
- **Channel:** Mono atau stereo
- **Preprocessing:** Audio multi-channel dikonversi menjadi mono
- **Durasi:** sekitar 1 detik per rekaman

### 🔤 Kelas Huruf Hijaiyah

Kelas yang digunakan oleh training pipeline adalah:

| No. | Huruf | Transliterasi |
|---:|---|---|
| 1 | اَ | Alif |
| 2 | بَ | Ba |
| 3 | تَ | Ta |
| 4 | ثَ | Tsa |
| 5 | جَ | Ja |
| 6 | حَ | Ha |
| 7 | خَ | Kha |
| 8 | دَ | Da |
| 9 | ذَ | Dza |
| 10 | رَ | Ra |
| 11 | زَ | Za |
| 12 | سَ | Sa |
| 13 | شَ | Sya |
| 14 | صَ | Sha |
| 15 | ضَ | Dha |
| 16 | طَ | Tho |
| 17 | ظَ | Zha |
| 18 | عَ | Ain |
| 19 | غَ | Gha |
| 20 | فَ | Fa |
| 21 | قَ | Qo |
| 22 | كَ | Ka |
| 23 | لَ | La |
| 24 | مَ | Ma |
| 25 | نَ | Na |
| 26 | هَ | Hha |
| 27 | وَ | Wa |
| 28 | يَ | Ya |

Daftar kelas tersebut didefinisikan secara terpusat di `config.py` dan digunakan
sebagai urutan canonical untuk pipeline proyek. :contentReference[oaicite:1]{index=1}

---

# 📊 Pembagian Dataset

Dengan 28 kelas dan 40 rekaman per kelas:

```text
1.120 total rekaman
│
├── 896 training pool (80%)
│     │
│     └── 5-Fold Stratified Cross-Validation
│
└── 224 holdout (20%)
      │
      └── Disimpan untuk evaluasi akhir
Training Pool

Sebanyak 896 rekaman digunakan sebagai training pool.

Data tersebut digunakan dalam 5-fold stratified cross-validation, sehingga
setiap fold mempertahankan proporsi kelas secara seimbang.

Holdout

Sebanyak 224 rekaman disimpan sebagai holdout set.

Data holdout tidak digunakan selama proses cross-validation dan disimpan untuk
evaluasi akhir.

Hal ini memungkinkan performa model diuji menggunakan data yang tidak digunakan
selama proses training maupun pemilihan model.

🤖 Model

Proyek ini menggunakan:

OpenAI Whisper Tiny

Whisper merupakan model speech recognition yang telah dilatih sebelumnya.
Pada proyek ini, model tersebut tidak dibuat dari awal.

Pipeline secara umum:

Pretrained Whisper Tiny
        │
        ▼
Hijaiyah Speech Dataset
        │
        ▼
Fine-Tuning
        │
        ▼
5-Fold Cross-Validation
        │
        ▼
Best Model
        │
        ▼
Hijaiyah Recognition

Konfigurasi model menggunakan:

Model      : openai/whisper-tiny
Language   : Arabic
Task       : Transcribe
Sample Rate: 16 kHz

Konfigurasi tersebut ditetapkan di config.py.

🔬 Training

Training menggunakan:

5-fold cross-validation
Stratified split
Seed: 42
Batch size: 8
Learning rate: 1e-5
Weight decay: 0.01
Maximum epoch: 20
FP16 ketika CUDA tersedia

Konfigurasi training ditentukan di config.py.

Apa itu Epoch?

Satu epoch berarti seluruh data training yang digunakan pada suatu proses
training telah diproses satu kali.

Contohnya:

1 Epoch
│
└── Seluruh data training → diproses 1×

Model dapat melakukan beberapa epoch agar parameter model dapat terus
disesuaikan berdasarkan data training.

📈 Evaluasi

Metrik utama yang digunakan dalam penelitian ini adalah:

Character Error Rate (CER)

CER digunakan untuk mengukur jumlah kesalahan pada tingkat karakter antara
hasil transkripsi model dan target yang sebenarnya.

Secara umum:

Semakin kecil nilai CER, semakin sedikit kesalahan karakter pada hasil
prediksi.

Hasil 5-Fold Cross-Validation
Fold	Best CER	Best Epoch
Fold 1	0.0667	5
Fold 2	0.0503	7
Fold 3	0.0447	7
Fold 4	0.0419	8
Fold 5	0.0726	6
Mean CER	0.0552	—

Nilai di atas merupakan hasil 5-fold cross-validation.
Nilai tersebut bukan hasil evaluasi akhir pada holdout set.

Hasil terbaik dari satu fold adalah CER 0.0419 pada Fold 4 di Epoch 8.
Nilai tersebut merupakan hasil terbaik pada salah satu fold dan tidak boleh
dianggap sebagai performa akhir model pada seluruh data holdout.

🎙️ Inference

Proyek menyediakan beberapa mode inference untuk menguji model secara langsung.

Mode yang tersedia
1. Record From Microphone
2. Live Microphone
3. Use Existing WAV File
Live Microphone

Pada mode live, audio dari microphone diproses secara berkelanjutan dan model
memberikan prediksi secara real-time.

Contoh:

LIVE MODE

Listening...

Prediction : بَ
Prediction : تَ
Prediction : جَ

Pipeline inference:

Microphone
    │
    ▼
Audio Processing
    │
    ▼
Whisper
    │
    ▼
Prediction
    │
    ▼
Hijaiyah Letter
⚠️ Keterbatasan

Eksperimen ini masih merupakan proof of concept, sehingga terdapat beberapa
keterbatasan:

Dataset masih relatif kecil.
Eksperimen hanya menggunakan satu harakat, yaitu Fathah.
Model berfokus pada pelafalan huruf secara terisolasi, bukan kata atau ayat.
Variasi speaker masih terbatas.
Performa pada data atau speaker yang berbeda belum dapat dianggap setara dengan
performa pada dataset eksperimen.
Whisper merupakan model speech recognition generatif, bukan classifier
khusus huruf Hijaiyah.
Mode live microphone masih dapat menghasilkan prediksi berulang atau
terpengaruh oleh segmentasi audio.

Karena itu, hasil eksperimen ini sebaiknya dipandang sebagai eksplorasi awal,
bukan sebagai sistem pengenalan bacaan Al-Qur'an yang sudah siap digunakan
secara umum.

🚀 Pengembangan Selanjutnya

Beberapa pengembangan yang dapat dilakukan:

1. Menambahkan Harakat

Memperluas dataset dari:

Fathah

menjadi:

Fathah
Dhammah
Kasrah
Tanwin
2. Memperbesar Dataset

Menambahkan:

Lebih banyak rekaman per kelas
Lebih banyak speaker
Variasi kualitas microphone
Variasi lingkungan perekaman
3. Memperluas Task

Pipeline dapat dikembangkan secara bertahap:

Huruf
  ↓
Huruf + Harakat
  ↓
Suku Kata
  ↓
Kata
  ↓
Ayat
  ↓
Evaluasi Bacaan
4. Evaluasi Tajwid dan Makhraj

Pengembangan lebih lanjut dapat mengeksplorasi kemampuan sistem dalam
menganalisis aspek bacaan seperti:

Makhraj
Tajwid
Panjang-pendek bacaan
Waqaf
Kesalahan pelafalan
📁 Struktur Repository
QuranAIProjectPaper/
│
├── config.py
├── train.py
├── inference.py
├── inference2.py
├── preprocess.py
├── audio_utils.py
├── build_manifests.py
├── metrics.py
│
├── models/
│   └── best_model/
│
├── tests/
│   └── test_hijaiyah_pipeline.py
│
├── requirements.txt
├── README.md
└── .gitignore

Dataset audio dan file manifest tidak disertakan dalam repository GitHub.

🛠️ Teknologi

Proyek ini menggunakan beberapa teknologi utama:

Python
PyTorch
Hugging Face Transformers
OpenAI Whisper
Hugging Face Datasets
JiWER
CUDA / NVIDIA GPU
📌 Status Proyek

Status: Proof of Concept

Eksperimen saat ini berfokus pada:

28 Huruf Hijaiyah
        +
1 Harakat (Fathah)
        +
1.120 Rekaman
        +
Whisper Tiny
        +
5-Fold Cross-Validation

Proyek ini merupakan langkah awal untuk mengeksplorasi penggunaan model
speech recognition modern dalam pengenalan pelafalan huruf Hijaiyah dan
pengembangan sistem pembelajaran Al-Qur'an di masa mendatang.