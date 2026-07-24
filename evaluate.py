import os

import torch
import pandas as pd
import soundfile as sf
from tabulate import tabulate
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from config import TEST_CSV, BEST_MODEL_DIR

# ===========================
# Load Model
# ===========================
print("Loading model...")

processor = WhisperProcessor.from_pretrained(BEST_MODEL_DIR)
model = WhisperForConditionalGeneration.from_pretrained(BEST_MODEL_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

print("Model loaded.\n")

# ===========================
# Load Test CSV
# ===========================
df = pd.read_csv(TEST_CSV)

results = []

correct = 0
wrong = 0

name_map = {
    "اَ": "اَ (Alif)",
    "بَ": "بَ (Ba)",
    "تَ": "تَ (Ta)",
    "ثَ": "ثَ (Tsa)",
    "جَ": "جَ (Ja)"
}

# ===========================
# Evaluate
# ===========================
for _, row in df.iterrows():

    audio_path = row["audio"]
    label = row["transcription"]

    audio, sr = sf.read(audio_path)

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    prediction = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0].strip()

    is_correct = prediction == label

    if is_correct:
        correct += 1
        status = "Benar"
    else:
        wrong += 1
        status = "Salah"

    results.append([
        name_map.get(label, label),
        os.path.basename(audio_path),
        prediction,
        status
    ])

# ===========================
# Print Table
# ===========================
print(tabulate(
    results,
    headers=["Huruf", "Audio File", "Prediksi", "Hasil"],
    tablefmt="grid"
))

# ===========================
# Summary
# ===========================
total = correct + wrong
accuracy = (correct / total) * 100 if total else 0

print()
print(f"Total Sampel : {total}")
print(f"Benar        : {correct}")
print(f"Salah        : {wrong}")
print(f"Akurasi      : {accuracy:.2f}%")