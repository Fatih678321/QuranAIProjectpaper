"""
metrics.py

Evaluation Metric untuk Whisper.

Menggunakan Character Error Rate (CER)
sesuai task pengenalan huruf hijaiyah.
"""

import QuranAIProjectpaper.evaluate as evaluate

# =====================================================
# Load Metric
# =====================================================

cer_metric = evaluate.load("cer")


# =====================================================
# Compute Metrics
# =====================================================

def compute_metrics(pred, processor):
    """
    Menghitung Character Error Rate (CER).

    Parameters
    ----------
    pred:
        Output dari Seq2SeqTrainer.

    processor:
        WhisperProcessor.
    """

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 dengan PAD token
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode Prediction
    pred_str = processor.tokenizer.batch_decode(
        pred_ids,
        skip_special_tokens=True
    )

    # Decode Ground Truth
    label_str = processor.tokenizer.batch_decode(
        label_ids,
        skip_special_tokens=True
    )

    cer = cer_metric.compute(
        predictions=pred_str,
        references=label_str
    )

    return {
        "cer": cer
    }