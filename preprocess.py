"""
==========================================
Preprocessing Pipeline
==========================================

Tasks:
1. Load train.csv & test.csv
2. Load Whisper Processor
3. Convert audio to Whisper input features
4. Convert transcription to token IDs
5. Return processed DatasetDict
"""

from datasets import load_dataset, Audio, Features, Value
from transformers import WhisperProcessor

from config import (
    TRAIN_CSV,
    TEST_CSV,
    MODEL_NAME,
    LANGUAGE,
    TASK,
    SAMPLE_RATE,
)


# =====================================================
# Load Whisper Processor
# =====================================================

processor = WhisperProcessor.from_pretrained(
    MODEL_NAME,
    language=LANGUAGE,
    task=TASK,
)


# =====================================================
# Preprocessing Function
# =====================================================

def prepare_dataset(batch):
    """
    Convert one audio sample into Whisper features
    and tokenize the transcription.
    """

    audio = batch["audio"]

    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
    ).input_features[0]

    batch["labels"] = processor.tokenizer(
        batch["transcription"]
    ).input_ids

    return batch


# =====================================================
# Load & Prepare Dataset
# =====================================================

def load_and_prepare_dataset():
    """
    Returns
    -------
    DatasetDict
        Processed train and test datasets.
    """

    features = Features({
        "audio": Audio(sampling_rate=SAMPLE_RATE),
        "transcription": Value("string")
    })

    dataset = load_dataset(
        "csv",
        data_files={
            "train": str(TRAIN_CSV),
            "test": str(TEST_CSV),
        },
        features=features,
    )

    dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset["train"].column_names,
        desc="Preprocessing Dataset",
    )

    return dataset


# =====================================================
# Preview
# =====================================================

def main():

    dataset = load_and_prepare_dataset()

    print("=" * 60)
    print(dataset)
    print("=" * 60)

    print("Train Samples :", len(dataset["train"]))
    print("Test Samples  :", len(dataset["test"]))
    print("=" * 60)

    sample = dataset["train"][0]

    print(sample.keys())
    print("=" * 60)

    print("Feature Length :", len(sample["input_features"]))
    print("Label IDs      :", sample["labels"])

    print("=" * 60)


if __name__ == "__main__":
    main()