"""
QuranAI Hijaiyah Recognition
----------------------------
A CLI application for recognizing spoken Hijaiyah letters using a fine-tuned Whisper model.
"""

import os
import queue
import time
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from audio_utils import downmix_to_mono

# Local configuration imports
from config import BEST_MODEL_DIR, HIJAIYAH_CLASSES


# =============================================================================
# 1. CONSTANTS & CONFIGURATION
# =============================================================================

SAMPLE_RATE = 16000
RECORD_SECONDS = 2
LIVE_CHUNK_SECONDS = 1.0  # Live prediction every 1.0 seconds
RECORDINGS_DIR = "recordings"
PREDICTION_COOLDOWN = 1  # Minimum seconds between predictions in live mode

# Threshold for Voice Activity Detection (RMS Energy)
# Increase if it's picking up too much background noise, decrease if it skips your voice
SILENCE_THRESHOLD = 0.015 

# Dictionary mapping menu choices to (Arabic Letter, Latin Name).
# It is derived from the training class registry so inference accepts exactly
# the labels learned by the retrained model.
TARGETS = {
    str(index): (letter.transcription, letter.display_name)
    for index, letter in enumerate(HIJAIYAH_CLASSES, start=1)
}

# The Output Filter: Strictly allowed characters
ALLOWED_LETTERS = {value[0] for value in TARGETS.values()}


# =============================================================================
# 2. CORE MACHINE LEARNING & AUDIO PROCESSING
# =============================================================================

def load_model() -> Tuple[WhisperProcessor, WhisperForConditionalGeneration, str]:
    """Loads the Whisper processor and model into the optimal hardware device."""
    print("=" * 50)
    print("Loading Whisper Model... Please wait.")
    print("=" * 50)

    processor = WhisperProcessor.from_pretrained(BEST_MODEL_DIR)
    model = WhisperForConditionalGeneration.from_pretrained(BEST_MODEL_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print(f"Model Loaded Successfully! (Running on: {device.upper()})\n")
    return processor, model, device


def is_silence(audio: np.ndarray, threshold: float = SILENCE_THRESHOLD) -> bool:
    """Calculates RMS energy of the audio chunk. Returns True if it's considered silence."""
    if len(audio) == 0:
        return True
    
    # Root Mean Square (RMS) Calculation
    rms = np.sqrt(np.mean(audio**2))
    return rms < threshold


def predict_audio(
    processor: WhisperProcessor, 
    model: WhisperForConditionalGeneration, 
    device: str, 
    audio: np.ndarray, 
    sample_rate: int = SAMPLE_RATE
) -> Tuple[Optional[str], float, float]:
    """
    Predicts the audio and calculates confidence.
    Returns: (Prediction String or None if filtered, Confidence Score, Elapsed Time)
    """
    # 1. Voice Activity Detection (Skip if Silence)
    if is_silence(audio):
        return None, 0.0, 0.0

    # Resample to 16kHz if necessary
    if sample_rate != SAMPLE_RATE:
        num_samples = int(len(audio) * (SAMPLE_RATE / sample_rate))
        audio = np.interp(
            np.linspace(0, len(audio), num_samples, endpoint=False),
            np.arange(len(audio)),
            audio,
        )
        sample_rate = SAMPLE_RATE

    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    start_time = time.time()
    with torch.no_grad():
        # Using output_scores=True allows us to compute confidence
        outputs = model.generate(
            input_features,
            return_dict_in_generate=True,
            output_scores=True
        )
    elapsed_time = time.time() - start_time

    # Decode the text
    predicted_ids = outputs.sequences
    prediction = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    
    # 2. Confidence Score Calculation
    scores = outputs.scores
    if scores:
        # Get the max probability for each generated step using softmax
        step_probs = [torch.softmax(logits, dim=-1).max().item() for logits in scores]
        confidence = (sum(step_probs) / len(step_probs)) * 100
    else:
        confidence = 0.0

    # 3. Output Filter
    # If the model predicts something outside our targeted letters, we reject it.
    if prediction not in ALLOWED_LETTERS:
        return None, confidence, elapsed_time
    
    return prediction, confidence, elapsed_time


# =============================================================================
# 3. USER INTERFACE & DISPLAY HELPERS
# =============================================================================

def prompt_target_mode() -> Tuple[str, Optional[str]]:
    """Asks the user to choose between random exploration or practicing a specific letter."""
    print("\nRecognition Mode:")
    print("1. Random / Exploration (Free practice)")
    print("2. Target Letter (Test a specific letter)")

    mode = input("\nPick [1/2]: ").strip()
    
    if mode != "2":
        return "1", None

    print("\nChoose Target Letter:")
    for key, (arabic, latin) in TARGETS.items():
        print(f"  {key}. {arabic} ({latin})")

    target_choice = input(f"\nPick [1-{len(TARGETS)}]: ").strip()
    target_info = TARGETS.get(target_choice)

    if not target_info:
        print("Invalid choice! Defaulting to Alif.")
        return "2", TARGETS["1"][0]

    target_letter = target_info[0]
    print(f"\nTarget Selected: {target_letter}")
    return "2", target_letter


def display_result(prediction: Optional[str], confidence: float, elapsed: float, mode: str, target: Optional[str]) -> None:
    """Prints the inference results in a clean, formatted box."""
    print("\n" + "=" * 50)
    print("RESULT")
    print("=" * 50)
    
    if prediction is None:
        print("Prediction     : [Skipped: Silence or Unrecognized]")
    else:
        print(f"Prediction     : {prediction}")
        print(f"Confidence     : {confidence:.1f}%")

        if mode == "2" and target:
            print(f"Target         : {target}")
            if prediction == target:
                print("\n✔ Correct! Great job.")
            else:
                print("\n✘ Wrong. Try again!")

    print(f"\nInference Time : {elapsed:.3f} seconds")
    print("=" * 50)


# =============================================================================
# 4. APPLICATION MODES (MIC, LIVE, FILE)
# =============================================================================

def run_microphone_mode(
    processor: WhisperProcessor, 
    model: WhisperForConditionalGeneration, 
    device: str, 
    mode: str, 
    target: Optional[str]
) -> None:
    """Handles recording a fixed length audio clip from the microphone and processing it."""
    input("\nPress ENTER to start recording...")

    for i in range(3, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)

    print("\n[ 🔴 Recording... Speak now! ]")
    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    audio = audio.squeeze() 
    print("[ ⏹️ Recording Complete ]")

    # Save audio file
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    filename = datetime.now().strftime("record_%Y%m%d_%H%M%S.wav")
    filepath = os.path.join(RECORDINGS_DIR, filename)
    sf.write(filepath, audio, SAMPLE_RATE)
    print(f"Audio saved to: {filepath}")

    prediction, confidence, elapsed = predict_audio(processor, model, device, audio, SAMPLE_RATE)
    display_result(prediction, confidence, elapsed, mode, target)


def run_live_mode(
    processor: WhisperProcessor, 
    model: WhisperForConditionalGeneration, 
    device: str, 
    mode: str, 
    target: Optional[str]
) -> None:
    """Continuously listens to the microphone and predicts in real-time chunks."""
    print("\n" + "=" * 50)
    print("LIVE MODE")
    print("=" * 50)
    print("Press Ctrl+C to stop.\n")

    audio_queue = queue.Queue()
    last_prediction = None
    last_prediction_time = 0.0
    chunk_size = int(SAMPLE_RATE * LIVE_CHUNK_SECONDS)

    def callback(indata, frames, time_info, status):
        if status:
            pass # Removed print to keep console clean during stream
        audio_queue.put(indata.copy())

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=chunk_size,
            callback=callback
        ):
            print("[ 🔴 Live Stream Active... Speak now! ]")
            while True:
                audio = audio_queue.get()
                audio = audio.squeeze()

                prediction, confidence, elapsed = predict_audio(processor, model, device, audio, SAMPLE_RATE)

                # Skip if it was silence OR not an allowed letter
                if prediction is None:
                    continue
                
                # Skip printing if it's identical to the previous chunk
                current_time = time.time()

                # Do not display if it is the same as the previous prediction
                if prediction == last_prediction:
                    continue

                # Cooldown to avoid prediction spam
                if current_time - last_prediction_time < PREDICTION_COOLDOWN:
                    continue

                last_prediction = prediction
                last_prediction_time = current_time

                print(f"\nPrediction : {prediction}")
                print(f"Confidence : {confidence:.1f}%")

                if mode == "2" and target:
                    if prediction == target:
                        print("✔ Correct")
                    else:
                        print("✘ Wrong")

                print(f"Inference  : {elapsed:.3f} sec")

    except KeyboardInterrupt:
        print("\nStopping Live Mode... Returning to main menu.")


def run_wav_file_mode(
    processor: WhisperProcessor, 
    model: WhisperForConditionalGeneration, 
    device: str, 
    mode: str, 
    target: Optional[str]
) -> None:
    """Handles loading an existing .wav file from disk and processing it."""
    filepath = input("\nEnter the full path to the WAV file: ").strip()

    if not os.path.isfile(filepath):
        print(f"\n[Error] File '{filepath}' not found. Please try again.")
        return

    try:
        audio, sample_rate = sf.read(filepath)
        
        audio = downmix_to_mono(audio)

        prediction, confidence, elapsed = predict_audio(processor, model, device, audio, sample_rate)
        display_result(prediction, confidence, elapsed, mode, target)
        
    except Exception as e:
        print(f"\n[Error] Failed to process audio file: {e}")


# =============================================================================
# 5. MAIN ENTRY POINT
# =============================================================================

def main() -> None:
    """Main application loop."""
    processor, model, device = load_model()

    while True:
        print("\n" + "=" * 50)
        print(" QuranAI Hijaiyah Recognition ")
        print("=" * 50)
        print("1. Record From Microphone")
        print("2. Live Microphone")
        print("3. Use Existing WAV File")
        print("0. Exit")

        choice = input("\nSelect an option [0-3]: ").strip()

        if choice == "0":
            print("\nExiting QuranAI. Goodbye!\n")
            break

        if choice not in ["1", "2", "3"]:
            print("\nInvalid option. Please try again.")
            continue

        mode, target = prompt_target_mode()

        if choice == "1":
            run_microphone_mode(processor, model, device, mode, target)
        elif choice == "2":
            run_live_mode(processor, model, device, mode, target)
        elif choice == "3":
            run_wav_file_mode(processor, model, device, mode, target)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nApplication interrupted by user. Exiting...")
