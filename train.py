"""
==========================================
Whisper Fine-Tuning Training Script
==========================================

Project:
    QuranAIProjectPaper

Purpose:
    Fine-tune Whisper for Hijaiyah Letter
    Recognition using a custom dataset.

Author:
    M. Fatih
"""

from functools import partial

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
    FP16,
    SEED,
)

from preprocess import (
    processor,
    load_and_prepare_dataset,
)

from data_collator import (
    DataCollatorSpeechSeq2SeqWithPadding,
)

from metrics import (
    compute_metrics,
)


# =====================================================
# Main
# =====================================================

def main():

    # -------------------------------------------------
    # Set Random Seed
    # -------------------------------------------------

    set_seed(SEED)

    # -------------------------------------------------
    # Load Dataset
    # -------------------------------------------------

    print("\nLoading dataset...")

    dataset = load_and_prepare_dataset()

    print("Done.\n")

    # -------------------------------------------------
    # Load Whisper Model
    # -------------------------------------------------

    print("Loading Whisper model...")

    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME
    )

    print("Done.\n")

    # -------------------------------------------------
    # Configure Whisper
    # -------------------------------------------------

    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK

    # Disable legacy language forcing
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None

    # Freeze convolutional feature encoder
    # (Official Hugging Face recommendation)
    model.model.encoder.conv1.requires_grad_(False)
    model.model.encoder.conv2.requires_grad_(False)

        # -------------------------------------------------
    # Data Collator
    # -------------------------------------------------

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # -------------------------------------------------
    # Metric Wrapper
    # -------------------------------------------------

    metric_fn = partial(
        compute_metrics,
        processor=processor,
    )

    # -------------------------------------------------
    # Training Arguments
    # -------------------------------------------------

    training_args = Seq2SeqTrainingArguments(

        # Output
        output_dir=str(OUTPUT_DIR),

        # Training
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,

        num_train_epochs=NUM_EPOCHS,

        warmup_steps=WARMUP_STEPS,

        max_grad_norm=MAX_GRAD_NORM,

        fp16=FP16,

        # Evaluation
        evaluation_strategy=EVALUATION_STRATEGY,

        predict_with_generate=True,

        generation_max_length=32,

        # Saving
        save_strategy=SAVE_STRATEGY,

        save_total_limit=SAVE_TOTAL_LIMIT,

        load_best_model_at_end=LOAD_BEST_MODEL_AT_END,

        metric_for_best_model="cer",

        greater_is_better=False,

        # Logging
        logging_steps=LOGGING_STEPS,

        logging_first_step=True,

        report_to="none",

        # Dataloader
        dataloader_num_workers=0,

        dataloader_pin_memory=True,

        remove_unused_columns=False,

        # Reproducibility
        seed=SEED,
    )

    print("Training arguments initialized.\n")

    # -------------------------------------------------
    # Dataset Summary
    # -------------------------------------------------

    print("=" * 60)

    print("Training Samples :", len(dataset["train"]))

    print("Validation Samples :", len(dataset["test"]))

    print("Model :", MODEL_NAME)

    print("Language :", LANGUAGE)

    print("Task :", TASK)

    print("Device :", "CUDA" if FP16 else "CPU")

    print("=" * 60)

    print("Whisper configuration completed.\n")

    # -------------------------------------------------
    # Inisialisasi Trainer
    # -------------------------------------------------
    print("Initialize Trainer...")
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=metric_fn,
        tokenizer=processor.feature_extractor,
    )

    # -------------------------------------------------
    # Mulai Training
    # -------------------------------------------------
    print("Memulai proses training...")
    trainer.train()

    # -------------------------------------------------
    # Simpan Model Terbaik & Processor
    # -------------------------------------------------
    print("Training selesai! Menyimpan model terbaik...")
    trainer.save_model(str(BEST_MODEL_DIR))
    processor.save_pretrained(str(BEST_MODEL_DIR))
    
    print("Semua proses selesai!")

if __name__ == "__main__":
    main()