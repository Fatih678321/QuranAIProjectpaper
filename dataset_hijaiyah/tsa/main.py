import sounddevice as sd
from scipy.io.wavfile import write
import os

# Setting audio
fs = 16000
seconds = 1

# Folder dataset
folder = "QuranAIProjectpaper/dataset_hijaiyah/tsa"

# Buat folder kalau belum ada
os.makedirs(folder, exist_ok=True)

# Cari jumlah file yang sudah ada
jumlah_file = len([
    f for f in os.listdir(folder)
    if f.endswith(".wav")
])

# Nama file berikutnya
filename = f"{folder}/tsa_{jumlah_file + 1:03d}.wav"

print("Mulai rekam...")
print("Ucapkan: tsa")

recording = sd.rec(
    int(seconds * fs),
    samplerate=fs,
    channels=1,
    dtype="int16",
    device=1
)

sd.wait()

write(filename, fs, recording)

print("Rekaman selesai")
print("Disimpan:", filename)