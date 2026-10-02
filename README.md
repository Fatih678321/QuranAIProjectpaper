# QuranAIProjectpaper

> Fine-tuning OpenAI Whisper untuk mengenali huruf Hijaiyah menggunakan dataset audio sederhana.

---

# Tentang Project

Project ini adalah penelitian awal (proof of concept) yang bertujuan untuk melihat apakah model **OpenAI Whisper** dapat diadaptasi untuk mengenali ucapan huruf Hijaiyah.

Whisper pada dasarnya dibuat untuk mengenali ucapan manusia dalam berbagai bahasa. Pada penelitian ini, model tersebut di-*fine-tune* memakai dataset kecil yang hanya berisi beberapa huruf Hijaiyah.

Project ini adalah langkah pertama menuju AI yang nanti akan mampu membantu mengkoreksi bacaan Al-Qur'an.

---

# Tujuan

Tujuan utama project ini adalah:

- mempelajari proses fine-tuning model Whisper,
- membuat model yang dapat mengenali ucapan huruf Hijaiyah,
- memahami pipeline machine learning untuk speech recognition,
- menjadi dasar pembuatan AI untuk pemebelajaran Al-Qur'an.

Project ini **belum ditujukan untuk mengenali ayat Al-Qur'an**, tetapi hanya sebagai fondasi sebelum masuk ke bagian yang lebih sulit.

---

# Dataset

Dataset yang digunakan terdiri dari:

- 600 file audio yang dipilih untuk eksperimen
- format WAV
- mono atau stereo (diubah menjadi mono saat preprocessing)
- 16 kHz
- durasi 1 detik

Huruf yang digunakan:

- اَ (Alif), بَ (Ba), تَ (Ta), ثَ (Tsa), جَ (Ja)
- حَ (Ha), خَ (Kha), دَ (Da), ذَ (Dza), رَ (Ra)
- زَ (Za), سَ (Sa), شَ (Sya), صَ (Sha), ضَ (Dha)

Masing-masing huruf memiliki 40 sampel audio. Empat file tambahan
`ta_041.wav` sampai `ta_044.wav` tidak termasuk dalam eksperimen ini.

Dataset dibagi menjadi:

- Training : 480 audio
- Testing : 120 audio

Manifest dibangun ulang dengan seed 42. Assignment lima kelas lama tetap
dipertahankan; sepuluh kelas tambahan masing-masing dibagi menjadi 32 audio
training dan 8 audio holdout. Training menggunakan 5-fold stratified
cross-validation pada 480 audio training. Sebanyak 120 audio testing tetap
menjadi holdout dan tidak digunakan saat training atau pemilihan fold. Fold
dengan validation CER terendah disimpan ke `models/best_model`.

Model di `models/best_model` yang sudah ada hanya mempelajari kelas lama.
Jalankan ulang `python train.py` sebelum menggunakan inference untuk 15 kelas.

---

# Struktur Project

```
QuranAIProjectpaper/
│
├── __pycache__/               
├── dataset_hijaiyah/           
├── logs/                       
├── models/
│   └── best_model/             
├── outputs/                    
├── recordings/                 # Hasil rekaman dari inference.py
├── venv/                       
│
├── baseline_results.csv       
├── config.py                   
├── data_collator.py            
├── evaluate_baseline.py        # Hasil evaluasi baseline Whisper    
├── evaluate.py                 
├── inference.py                
├── metadata.csv                
├── metrics.py                  
├── preprocess.py               
├── requirements-lock.txt      
├── requirements.txt            
├── research_log.md             
├── readme.md                   
├── test.csv                   
├── train.csv                    
└── train.py                    
```

---

# Penjelasan File

## config.py

Menyimpan seluruh konfigurasi project seperti:

- lokasi dataset
- model yang digunakan
- learning rate
- batch size
- epoch
- output folder

---

## preprocess.py

Melakukan preprocessing dataset.

Tugasnya adalah:

- membaca audio
- mengubah audio menjadi input Whisper
- mengubah label menjadi token
- menghasilkan dataset siap training

---

## data_collator.py

Mengatur proses padding sehingga semua batch memiliki ukuran yang sama sebelum diberikan ke model.

---

## metrics.py

Menghitung Character Error Rate (CER).

CER digunakan untuk mengukur seberapa banyak karakter yang salah diprediksi oleh model.

Semakin kecil nilai CER maka semakin baik performa model.

---

## train.py

Digunakan untuk melakukan fine-tuning model Whisper.

Tahapan:

1. load dataset
2. preprocessing
3. training
4. evaluasi setiap epoch
5. menyimpan model terbaik

Output:

```
models/best_model
```

---

## evaluate.py

Digunakan untuk menguji model yang sudah selesai dilatih.

Script ini akan:

- membaca seluruh data test
- melakukan prediksi
- membandingkan hasil dengan label sebenarnya
- menghitung akurasi
- menampilkan tabel hasil prediksi

Contoh output:

```
+------------+--------------+------------+--------+
| Huruf      | Audio File   | Prediksi   | Hasil  |
+------------+--------------+------------+--------+
| اَ (Alif)  | alif_017.wav | اَ         | Benar  |
| بَ (Ba)    | ba_017.wav   | بَ         | Benar  |
| تَ (Ta)    | ta_017.wav   | ثَ         | Salah  |
...
```

---

## inference.py

Digunakan untuk mencoba model pada audio baru.

Fitur:

- Record dari microphone
- Menggunakan file WAV
- Mode bebas (prediksi huruf apa saja)
- Mode target (mengecek apakah pelafalan sesuai huruf tertentu)

Contoh:

```
Target

بَ

Prediction

بَ

Status

BENAR
```

atau

```
Prediction

ثَ
```

pada mode bebas.

---

# Hasil Training

Dataset:

- Train : 80
- Test : 20

Training:

- 20 Epoch

Perkembangan CER:

| Epoch | CER |
|-------:|----:|
| 1 | 1.775 |
| 3 | 0.750 |
| 5 | 0.100 |
| 6 | 0.050 |
| 7-20 | 0.025 |

Model berhasil belajar mengenali dataset dengan baik.

---

# Cara Menjalankan

## Install dependency

```
pip install -r requirements.txt
```

---

## Training

```
python build_manifests.py
python train.py
```

---

## Evaluasi

```
python evaluate.py
```

---

## Inference

```
python inference.py
```

---

# Cara Kerja Sistem

```
Audio
        │
        ▼
Whisper Processor
        │
        ▼
Whisper Model
        │
        ▼
Prediksi Huruf
        │
        ▼
Hasil
```

Pada mode target:

```
Prediksi

↓

Bandingkan dengan target

↓

Benar / Salah
```

---

# Pengembangan Selanjutnya

Project ini dirancang agar dapat berkembang secara bertahap.

Roadmap yang direncanakan:

- Menambah seluruh 28 huruf Hijaiyah
- Menambahkan harakat lain
- Pengenalan suku kata
- Pengenalan kata
- Pengenalan ayat Al-Qur'an
- Penilaian tajwid
- Penilaian makhraj huruf
---

# Catatan

Project ini adalah penelitian pembelajaran mengenai fine-tuning model speech recognition .

Masih banyak yang akan dikembangkan, seperti ukuran dataset, jumlah huruf, dan kualitas audio.

Tapi project ini berhasil memberi tahu bahwa model Whisper bisa dijadikan fondasi awal untuk membuat sistem pengenalan pengucapan huruf Hijaiyah.
