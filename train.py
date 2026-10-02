"""
==========================================
Whisper Fine-Tuning Training Script
==========================================

Project:
    QuranAIProjectPaper

Purpose:
    Fine-tune Whisper for Hijaiyah Letter
    Recognition using a custom dataset,
    using Stratified K-Fold Cross Validation.

Author:
    M. Fatih
"""

from collections import Counter
from functools import partial

import numpy as np
from sklearn.model_selection import StratifiedKFold

from transformers import (
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from config import (
    MODEL_NAME,
    LANGUAGE,
    TASK,
    OUTPUT_DIR,
    BEST_MODEL_DIR,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    WARMUP_STEPS,
    LOGGING_STEPS,
    GRADIENT_ACCUMULATION_STEPS,
    MAX_GRAD_NORM,
    SAVE_TOTAL_LIMIT,
    LOAD_BEST_MODEL_AT_END,
    SAVE_STRATEGY,
    EVALUATION_STRATEGY,
    CROSS_VALIDATION_FOLDS,
    EXPECTED_TRANSCRIPTIONS,
    FP16,
    SEED,
)

from preprocess import (
    processor,
    load_dataset_splits,
    prepare_loaded_dataset,
)

from data_collator import (
    DataCollatorSpeechSeq2SeqWithPadding,
)

from metrics import (
    compute_metrics,
)


# =====================================================
# Helper: Create a fresh Whisper model for each fold
# =====================================================

def create_model():
    """Load & configure a fresh copy of Whisper for each fold."""
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    # Freeze convolutional feature encoder layers
    model.model.encoder.conv1.requires_grad_(False)
    model.model.encoder.conv2.requires_grad_(False)
    return model


# =====================================================
# Helper: Create training arguments for a fold
# =====================================================

def create_training_args(output_dir):
    """Return Seq2SeqTrainingArguments for a given fold output dir."""
    return Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=NUM_EPOCHS,
        warmup_steps=WARMUP_STEPS,
        max_grad_norm=MAX_GRAD_NORM,
        fp16=FP16,
        eval_strategy=EVALUATION_STRATEGY,
        predict_with_generate=True,
        generation_max_length=32,
        save_strategy=SAVE_STRATEGY,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
        metric_for_best_model="cer",
        greater_is_better=False,
        logging_steps=LOGGING_STEPS,
        logging_first_step=True,
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        seed=SEED,
    )


def validate_dataset_splits(split_class_labels):
    """Validate the manifest labels before expensive preprocessing/training."""
    expected_labels = set(EXPECTED_TRANSCRIPTIONS)
    train_labels = split_class_labels["train"]
    test_labels = split_class_labels["test"]
    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)

    if set(train_counts) != expected_labels:
        raise ValueError(
            "Training manifest labels do not match the configured Hijaiyah "
            f"classes. Found: {sorted(train_counts)}"
        )

    if set(test_counts) != expected_labels:
        raise ValueError(
            "Holdout manifest labels do not match the configured Hijaiyah "
            f"classes. Found: {sorted(test_counts)}"
        )

    insufficient_labels = {
        label: count
        for label, count in train_counts.items()
        if count < CROSS_VALIDATION_FOLDS
    }
    if insufficient_labels:
        raise ValueError(
            "Every training class needs at least "
            f"{CROSS_VALIDATION_FOLDS} samples for stratified cross-validation. "
            f"Insufficient classes: {insufficient_labels}"
        )

    print("\nDataset class counts:")
    for label in EXPECTED_TRANSCRIPTIONS:
        print(
            f"  {label}: train={train_counts[label]}, "
            f"holdout={test_counts[label]}"
        )

    return train_labels


def validate_fold_labels(labels, train_indices, validation_indices):
    """Ensure every cross-validation partition contains every configured label."""
    expected_labels = set(EXPECTED_TRANSCRIPTIONS)
    train_fold_labels = set(labels[train_indices])
    validation_fold_labels = set(labels[validation_indices])
    if train_fold_labels != expected_labels or validation_fold_labels != expected_labels:
        raise ValueError(
            "Every cross-validation fold must contain all configured labels. "
            f"Train labels: {sorted(train_fold_labels)}; "
            f"validation labels: {sorted(validation_fold_labels)}"
        )


# =====================================================
# Main
# =====================================================

def main():

    set_seed(SEED)

    # -------------------------------------------------
    # Load Dataset
    # -------------------------------------------------

    print("\nLoading dataset...")

    # Validate raw labels before the audio is decoded and transformed.
    raw_dataset, split_class_labels = load_dataset_splits()
    train_class_labels = validate_dataset_splits(split_class_labels)
    dataset = prepare_loaded_dataset(raw_dataset)

    print("Done.\n")

    # -------------------------------------------------
    # Prepare Cross Validation
    # -------------------------------------------------

    # train_class_labels is a list of strings like ["اَ", "بَ", ...]
    # StratifiedKFold needs one label per sample (single string, not token list)
    class_labels_array = np.array(train_class_labels)
    dummy_X = np.zeros(len(class_labels_array))

    splitter = StratifiedKFold(
        n_splits=CROSS_VALIDATION_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    metric_fn = partial(compute_metrics, processor=processor)
    fold_results = []
    best_cer = float("inf")

    print("=" * 60)
    print("Training Samples       :", len(dataset["train"]))
    print("Holdout Test Samples   :", len(dataset["test"]))
    print("Cross-validation folds :", CROSS_VALIDATION_FOLDS)
    print("Model                  :", MODEL_NAME)
    print("Device                 :", "CUDA" if FP16 else "CPU")
    print("=" * 60)

    # -------------------------------------------------
    # K-Fold Training Loop
    # -------------------------------------------------

    for fold_number, (train_indices, val_indices) in enumerate(
        splitter.split(dummy_X, class_labels_array),
        start=1,
    ):
        validate_fold_labels(class_labels_array, train_indices, val_indices)
        print(f"\n{'='*60}")
        print(f"Starting fold {fold_number}/{CROSS_VALIDATION_FOLDS}...")
        print(f"  Train size : {len(train_indices)}")
        print(f"  Val size   : {len(val_indices)}")
        print(f"{'='*60}")

        # Fresh model for every fold
        model = create_model()

        fold_output_dir = OUTPUT_DIR / f"fold_{fold_number}"

        trainer = Seq2SeqTrainer(
            args=create_training_args(fold_output_dir),
            model=model,
            train_dataset=dataset["train"].select(train_indices.tolist()),
            eval_dataset=dataset["train"].select(val_indices.tolist()),
            data_collator=DataCollatorSpeechSeq2SeqWithPadding(
                processor=processor,
                decoder_start_token_id=model.config.decoder_start_token_id,
            ),
            compute_metrics=metric_fn,
            tokenizer=processor.feature_extractor,
        )

        trainer.train()

        fold_eval = trainer.evaluate()
        fold_cer = fold_eval["eval_cer"]
        fold_results.append(fold_cer)
        print(f"Fold {fold_number} CER: {fold_cer:.4f}")

        # Save only the best model across all folds
        if fold_cer < best_cer:
            best_cer = fold_cer
            print(f"  -> New best model! Saving to {BEST_MODEL_DIR}")
            trainer.save_model(str(BEST_MODEL_DIR))
            processor.save_pretrained(str(BEST_MODEL_DIR))

    # -------------------------------------------------
    # Final Summary
    # -------------------------------------------------

    print("\n" + "=" * 60)
    print("Cross-validation complete.")
    print("=" * 60)
    for i, cer in enumerate(fold_results, start=1):
        print(f"  Fold {i} CER : {cer:.4f}")
    print(f"  Mean CER  : {np.mean(fold_results):.4f}")
    print(f"  Std CER   : {np.std(fold_results):.4f}")
    print(f"  Best CER  : {min(fold_results):.4f}")
    print(f"  Best model saved to: {BEST_MODEL_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
