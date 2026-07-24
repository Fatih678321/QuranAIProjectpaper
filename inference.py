import os
import time
from datetime import datetime

import sounddevice as sd
import soundfile as sf
import torch

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from config import BEST_MODEL_DIR

# ======================================================
# CONFIG
# ======================================================

SAMPLE_RATE = 16000
RECORD_SECONDS = 2

TARGETS = {
    "1": ("اَ", "Alif"),
    "2": ("بَ", "Ba"),
    "3": ("تَ", "Ta"),
    "4": ("ثَ", "Tsa"),
    "5": ("جَ", "Ja")
}

# ======================================================
# LOAD MODEL
# ======================================================

print("=" * 50)
print("Loading Whisper Model...")
print("=" * 50)

processor = WhisperProcessor.from_pretrained(BEST_MODEL_DIR)
model = WhisperForConditionalGeneration.from_pretrained(BEST_MODEL_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)
model.eval()

print("Model loaded\n")

# ======================================================
# MENU
# ======================================================

print("=" * 50)
print("QuranAI Hijaiyah Recognition")
print("=" * 50)
print("1. Record From Microphone")
print("2. Use WAV File")

choice = input("\nPick [1/2] : ")

print()

print("\nPick Mode")
print("1. Random Letter")
print("2. Target Letter")

mode = input("\nPick [1/2] : ")

target_letter = None
target_name = None

if mode == "2":

    print("\nPick Target Letter\n")

    for key, value in TARGETS.items():
        print(f"{key}. {value[0]} ({value[1]})")

    target_choice = input("\nPick [1-5] : ")

    target_letter, target_name = TARGETS[target_choice]

    print(f"\nTarget : {target_letter} ({target_name})")

# ======================================================
# AUDIO INPUT
# ======================================================

if choice == "1":

    input("\nPress ENTER to start recording...")

    print()

    for i in [3, 2, 1]:
        print(i)
        time.sleep(1)

    print("\n🎤 Recording...")

    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    audio = audio.squeeze()

    os.makedirs("recordings", exist_ok=True)

    filename = datetime.now().strftime("record_%Y%m%d_%H%M%S.wav")

    filepath = os.path.join("recordings", filename)

    sf.write(filepath, audio, SAMPLE_RATE)

    print("Recording completed.")
    print("Saved :", filepath)

else:

    filepath = input("Enter WAV file path: ").strip()

    audio, SAMPLE_RATE = sf.read(filepath)

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

# ======================================================
# LOAD AUDIO
# ======================================================

if choice == "1":
    audio, SAMPLE_RATE = sf.read(filepath)

# ======================================================
# PREDICT
# ======================================================

print("\nProcessing...")

inputs = processor(
    audio,
    sampling_rate=SAMPLE_RATE,
    return_tensors="pt"
)

input_features = inputs.input_features.to(device)

start = time.time()

with torch.no_grad():

    predicted_ids = model.generate(input_features)

elapsed = time.time() - start

prediction = processor.batch_decode(
    predicted_ids,
    skip_special_tokens=True
)[0].strip()

# ======================================================
# RESULT
# ======================================================

print("\n" + "=" * 50)
print("HASIL")
print("=" * 50)

print(f"Prediksi : {prediction}")

if mode == "2":

    print(f"Target    : {target_letter}")

    if prediction == target_letter:
        print("\nRight")
    else:
        print("\nWrong")

print(f"\nWaktu Inferensi : {elapsed:.3f} detik")
print("=" * 50)