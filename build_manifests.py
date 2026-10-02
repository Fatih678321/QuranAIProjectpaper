"""Build reproducible 80/20 train and holdout manifests for Hijaiyah audio."""

from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

from config import (
    DATASET_DIR,
    EXPECTED_TRANSCRIPTIONS,
    HIJAIYAH_CLASSES,
    PROJECT_ROOT,
    SEED,
    TEST_CSV,
    TRAIN_CSV,
)

SAMPLES_PER_CLASS = 40
TRAIN_SAMPLES_PER_CLASS = 32
TEST_SAMPLES_PER_CLASS = 8


def selected_wav_paths(folder: str) -> list[Path]:
    """Return the first 40 sorted WAV files selected for one class."""
    class_dir = DATASET_DIR / folder

    if folder == "ta":
        # The project intentionally keeps ta_041.wav through ta_044.wav out of
        # this fixed 400-recording experiment.
        paths = [class_dir / f"ta_{index:03d}.wav" for index in range(1, 41)]
    else:
        paths = sorted(class_dir.glob("*.wav"))

    if len(paths) < SAMPLES_PER_CLASS:
        raise ValueError(
            f"Expected at least {SAMPLES_PER_CLASS} WAV files for '{folder}', "
            f"found {len(paths)}."
        )
    paths = paths[:SAMPLES_PER_CLASS]

    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {folder} samples required by the manifest: {', '.join(missing)}"
        )

    if len(paths) != SAMPLES_PER_CLASS:
        raise ValueError(
            f"Expected exactly {SAMPLES_PER_CLASS} WAV files for '{folder}', "
            f"found {len(paths)}."
        )

    return paths


def collect_records() -> list[dict[str, str]]:
    """Collect the selected, labelled audio paths in a stable order."""
    records: list[dict[str, str]] = []

    for letter in HIJAIYAH_CLASSES:
        for audio_path in selected_wav_paths(letter.folder):
            records.append(
                {
                    "audio": audio_path.relative_to(PROJECT_ROOT).as_posix(),
                    "transcription": letter.transcription,
                }
            )

    return records


def load_existing_records() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Load assignments for every configured class already in the manifests."""
    with TRAIN_CSV.open(encoding="utf-8", newline="") as train_file:
        train_records = list(csv.DictReader(train_file))
    with TEST_CSV.open(encoding="utf-8", newline="") as test_file:
        test_records = list(csv.DictReader(test_file))

    configured_labels = set(EXPECTED_TRANSCRIPTIONS)
    return (
        [record for record in train_records if record["transcription"] in configured_labels],
        [record for record in test_records if record["transcription"] in configured_labels],
    )


def split_new_class(letter) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Create the deterministic 32/8 split for one newly added class."""
    paths = selected_wav_paths(letter.folder)
    records = [
        {
            "audio": audio_path.relative_to(PROJECT_ROOT).as_posix(),
            "transcription": letter.transcription,
        }
        for audio_path in paths
    ]
    test_indices = set(
        random.Random(SEED).sample(range(len(records)), TEST_SAMPLES_PER_CLASS)
    )
    train_records = [
        record for index, record in enumerate(records) if index not in test_indices
    ]
    test_records = [
        record for index, record in enumerate(records) if index in test_indices
    ]
    train_records.sort(key=lambda record: record["audio"])
    test_records.sort(key=lambda record: record["audio"])
    return train_records, test_records


def validate_split_records(
    train_records: Iterable[dict[str, str]],
    test_records: Iterable[dict[str, str]],
) -> None:
    """Ensure the split has complete, balanced, non-overlapping classes."""
    train_records = list(train_records)
    test_records = list(test_records)
    all_records = train_records + test_records
    expected_labels = set(EXPECTED_TRANSCRIPTIONS)

    expected_total = len(HIJAIYAH_CLASSES) * SAMPLES_PER_CLASS
    if len(all_records) != expected_total:
        raise ValueError(f"Expected {expected_total} records, found {len(all_records)}.")

    audio_paths = [record["audio"] for record in all_records]
    if len(audio_paths) != len(set(audio_paths)):
        raise ValueError("The manifests contain duplicate audio paths.")

    folder_by_label = {
        letter.transcription: letter.folder for letter in HIJAIYAH_CLASSES
    }
    for record in all_records:
        expected_prefix = f"dataset_hijaiyah/{folder_by_label[record['transcription']]}/"
        if not record["audio"].startswith(expected_prefix):
            raise ValueError(
                f"Audio path {record['audio']!r} does not match label "
                f"{record['transcription']!r}."
            )

    if set(record["transcription"] for record in all_records) != expected_labels:
        raise ValueError("The manifests do not contain exactly the configured labels.")

    train_counts = Counter(record["transcription"] for record in train_records)
    test_counts = Counter(record["transcription"] for record in test_records)
    expected_train = {label: TRAIN_SAMPLES_PER_CLASS for label in expected_labels}
    expected_test = {label: TEST_SAMPLES_PER_CLASS for label in expected_labels}

    if train_counts != expected_train:
        raise ValueError(f"Unexpected train counts: {dict(train_counts)}")
    if test_counts != expected_test:
        raise ValueError(f"Unexpected test counts: {dict(test_counts)}")


def build_splits() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Preserve current assignments and split classes missing from the manifests."""
    train_records, test_records = load_existing_records()
    existing_train_labels = set(record["transcription"] for record in train_records)
    existing_test_labels = set(record["transcription"] for record in test_records)
    if existing_train_labels != existing_test_labels:
        raise ValueError("Existing train and holdout classes do not match.")

    train_counts = Counter(record["transcription"] for record in train_records)
    test_counts = Counter(record["transcription"] for record in test_records)
    for label in existing_train_labels:
        if train_counts[label] != TRAIN_SAMPLES_PER_CLASS:
            raise ValueError(
                f"Existing training assignment for {label!r} must contain "
                f"{TRAIN_SAMPLES_PER_CLASS} samples, found {train_counts[label]}."
            )
        if test_counts[label] != TEST_SAMPLES_PER_CLASS:
            raise ValueError(
                f"Existing holdout assignment for {label!r} must contain "
                f"{TEST_SAMPLES_PER_CLASS} samples, found {test_counts[label]}."
            )

    for letter in HIJAIYAH_CLASSES:
        if letter.transcription not in existing_train_labels:
            new_train, new_test = split_new_class(letter)
            train_records.extend(new_train)
            test_records.extend(new_test)

    train_records.sort(key=lambda record: record["audio"])
    test_records.sort(key=lambda record: record["audio"])
    validate_split_records(train_records, test_records)
    return train_records, test_records


def write_manifest(path: Path, records: Iterable[dict[str, str]]) -> None:
    """Write a UTF-8 CSV manifest with the schema consumed by preprocess.py."""
    with path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=("audio", "transcription"),
        )
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    train_records, test_records = build_splits()
    write_manifest(TRAIN_CSV, train_records)
    write_manifest(TEST_CSV, test_records)
    print(f"Wrote {len(train_records)} training samples to {TRAIN_CSV}")
    print(f"Wrote {len(test_records)} holdout samples to {TEST_CSV}")


if __name__ == "__main__":
    main()
