"""Small audio-shape helpers shared by the training pipeline and tests."""

import numpy as np


def downmix_to_mono(audio: np.ndarray) -> np.ndarray:
    """Return a mono waveform, averaging a detected channel axis when needed.

    Hugging Face audio decoders can represent multichannel data as either
    ``(frames, channels)`` or ``(channels, frames)``. Audio clips have far
    more frames than channels, so the smallest dimension is the channel axis.
    """
    waveform = np.asarray(audio)

    if waveform.ndim == 1:
        return waveform

    if waveform.ndim != 2:
        raise ValueError(
            "Expected a mono or two-dimensional multichannel waveform, "
            f"received shape {waveform.shape}."
        )

    channel_axis = int(np.argmin(waveform.shape))
    return waveform.mean(axis=channel_axis)
