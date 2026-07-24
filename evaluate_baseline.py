from faster_whisper import WhisperModel
import csv

print("Loading Whisper...")

model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

total = 0
correct = 0

results = []

with open("test.csv", "r", encoding="utf-8") as f:

    reader = csv.DictReader(f)

    for row in reader:

        audio_path = row["audio"]
        true_label = row["label"]

        print(f"\nTesting: {audio_path}")

        segments, info = model.transcribe(
            audio_path,
            language="ar"
        )

        prediction = ""

        for segment in segments:
            prediction += segment.text.strip()

        is_correct = prediction == true_label

        if is_correct:
            correct += 1

        total += 1

        results.append([
            audio_path,
            true_label,
            prediction,
            is_correct
        ])

        print("Target    :", true_label)
        print("Prediction:", prediction)
        print("Correct   :", is_correct)

accuracy = (correct / total) * 100 if total > 0 else 0

with open(
    "baseline_results.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "audio",
        "target",
        "prediction",
        "correct"
    ])

    writer.writerows(results)

print("\n====================")
print("Total    :", total)
print("Correct  :", correct)
print("Accuracy :", round(accuracy, 2), "%")
print("====================")