"""Focused regression tests for manifests, audio normalization, and folds."""

from collections import Counter
import csv
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

import numpy as np
from sklearn.model_selection import StratifiedKFold
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from audio_utils import downmix_to_mono
from build_manifests import (
    TEST_SAMPLES_PER_CLASS,
    TRAIN_SAMPLES_PER_CLASS,
    build_splits,
    validate_split_records,
)
from config import (
    CROSS_VALIDATION_FOLDS,
    EXPECTED_TRANSCRIPTIONS,
    HIJAIYAH_CLASSES,
    TEST_CSV,
    TRAIN_CSV,
)
from inference2 import ALLOWED_LETTERS, TARGETS


class HijaiyahPipelineTests(unittest.TestCase):
    def test_manifest_split_is_complete_balanced_and_non_overlapping(self):
        train_records, test_records = build_splits()
        validate_split_records(train_records, test_records)

        self.assertEqual(len(train_records), 896)
        self.assertEqual(len(test_records), 224)
        self.assertEqual(
            Counter(record["transcription"] for record in train_records),
            {label: TRAIN_SAMPLES_PER_CLASS for label in EXPECTED_TRANSCRIPTIONS},
        )
        self.assertEqual(
            Counter(record["transcription"] for record in test_records),
            {label: TEST_SAMPLES_PER_CLASS for label in EXPECTED_TRANSCRIPTIONS},
        )
        self.assertFalse(
            {record["audio"] for record in train_records}
            & {record["audio"] for record in test_records}
        )
        selected_paths = {record["audio"] for record in train_records + test_records}
        folder_by_label = {
            letter.transcription: letter.folder for letter in HIJAIYAH_CLASSES
        }
        for record in train_records + test_records:
            self.assertTrue(
                record["audio"].startswith(
                    f"dataset_hijaiyah/{folder_by_label[record['transcription']]}/"
                )
            )
        for index in range(41, 45):
            self.assertNotIn(
                f"dataset_hijaiyah/ta/ta_{index:03d}.wav", selected_paths
            )

        with TRAIN_CSV.open(encoding="utf-8", newline="") as train_file:
            persisted_train_records = list(csv.DictReader(train_file))
        with TEST_CSV.open(encoding="utf-8", newline="") as test_file:
            persisted_test_records = list(csv.DictReader(test_file))

        self.assertEqual(persisted_train_records, train_records)
        self.assertEqual(persisted_test_records, test_records)

    def test_stereo_audio_is_downmixed_to_mono(self):
        stereo = np.array(
            [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
            dtype=np.float32,
        )
        mono = downmix_to_mono(stereo)

        np.testing.assert_allclose(mono, np.array([0.5, 2.5, 4.5]))
        self.assertEqual(mono.ndim, 1)

        np.testing.assert_allclose(
            downmix_to_mono(stereo.T), np.array([0.5, 2.5, 4.5])
        )

    def test_new_stereo_wav_is_downmixed_to_mono(self):
        audio, sample_rate = sf.read(
            PROJECT_ROOT / "dataset_hijaiyah" / "ha" / "ha_001.wav",
            always_2d=True,
        )
        mono = downmix_to_mono(audio)

        self.assertEqual(sample_rate, 16000)
        self.assertEqual(audio.shape[1], 2)
        self.assertEqual(mono.ndim, 1)
        self.assertEqual(len(mono), audio.shape[0])

    def test_prepare_dataset_passes_mono_audio_to_feature_extractor(self):
        feature_extractor = Mock(
            return_value=SimpleNamespace(input_features=[np.zeros((80, 10))])
        )
        fake_processor = SimpleNamespace(
            feature_extractor=feature_extractor,
            tokenizer=Mock(return_value=SimpleNamespace(input_ids=[1])),
        )

        with patch(
            "transformers.WhisperProcessor.from_pretrained",
            return_value=fake_processor,
        ):
            import preprocess

        stereo = np.array(
            [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
            dtype=np.float32,
        )
        with patch.object(preprocess, "processor", fake_processor):
            preprocess.prepare_dataset(
                {
                    "audio": {"array": stereo, "sampling_rate": 16000},
                    "transcription": "اَ",
                }
            )

        extracted_audio = feature_extractor.call_args.args[0]
        self.assertEqual(extracted_audio.ndim, 1)
        self.assertEqual(len(extracted_audio), stereo.shape[0])

    def test_every_fold_contains_all_configured_classes(self):
        train_records, _ = build_splits()
        labels = np.array([record["transcription"] for record in train_records])
        splitter = StratifiedKFold(
            n_splits=CROSS_VALIDATION_FOLDS,
            shuffle=True,
            random_state=42,
        )

        for train_indices, validation_indices in splitter.split(
            np.zeros(len(labels)), labels
        ):
            self.assertEqual(
                set(labels[train_indices]), set(EXPECTED_TRANSCRIPTIONS)
            )
            self.assertEqual(
                set(labels[validation_indices]), set(EXPECTED_TRANSCRIPTIONS)
            )

    def test_inference_menu_and_filter_contain_all_canonical_labels(self):
        self.assertEqual(
            EXPECTED_TRANSCRIPTIONS,
            (
                "اَ", "بَ", "تَ", "ثَ", "جَ", "حَ", "خَ", "دَ", "ذَ", "رَ",
                "زَ", "سَ", "شَ", "صَ", "ضَ", "طَ", "ظَ", "عَ", "غَ", "فَ",
                "قَ", "كَ", "لَ", "مَ", "نَ", "هَ", "وَ", "يَ",
            ),
        )
        self.assertEqual(len(HIJAIYAH_CLASSES), 28)
        self.assertEqual(set(TARGETS), {str(index) for index in range(1, 29)})
        self.assertEqual(ALLOWED_LETTERS, set(EXPECTED_TRANSCRIPTIONS))


if __name__ == "__main__":
    unittest.main()
