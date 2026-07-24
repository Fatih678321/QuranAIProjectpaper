"""
data_collator.py

Custom Data Collator for Whisper Sequence-to-Sequence.

Tasks:
1. Combine input_features into batch
2. Perform padding on labels
3. Replace padding tokens with -100
4. Remove BOS token if already present

Created following the official Hugging Face Whisper pipeline,
adapted for this project.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data Collator untuk Whisper.

    Parameters
    ----------
    processor : WhisperProcessor
        Processor yang berisi FeatureExtractor dan Tokenizer.

    decoder_start_token_id : int
        BOS token milik Whisper.
    """

    processor: Any
    decoder_start_token_id: int

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        # ===========================
        # INPUT FEATURES
        # ===========================

        input_features = [
            {"input_features": feature["input_features"]}
            for feature in features
        ]

        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt"
        )

        # ===========================
        # LABELS
        # ===========================

        label_features = [
            {"input_ids": feature["labels"]}
            for feature in features
        ]

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100
        )

        # ===========================
        # REMOVE BOS TOKEN
        # ===========================

        if (
            labels[:, 0] == self.decoder_start_token_id
        ).all().cpu().item():

            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch