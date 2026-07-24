"""
==========================================
QuranAIProjectPaper Configuration
==========================================

All Project Configurations Are In This File.

Author : M. Fatih
Project: Fine-Tuning Whisper For 
Hijaiyah Letters Recognition
"""

from pathlib import Path
import torch

# =====================================================
# PROJECT PATH
# =====================================================

PROJECT_ROOT = Path(__file__).parent.resolve()

DATASET_DIR = PROJECT_ROOT / "dataset_hijaiyah"

TRAIN_CSV = PROJECT_ROOT / "train.csv"
TEST_CSV = PROJECT_ROOT / "test.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"
BEST_MODEL_DIR = MODEL_DIR / "best_model"
LOG_DIR = PROJECT_ROOT / "logs"

# otomatis membuat folder jika belum ada
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# =====================================================
# MODEL
# =====================================================

# Options:
# openai/whisper-tiny
# openai/whisper-base
# openai/whisper-small

MODEL_NAME = "openai/whisper-tiny"

LANGUAGE = "arabic"
TASK = "transcribe"

# =====================================================
# AUDIO
# =====================================================

SAMPLE_RATE = 16000

# =====================================================
# TRAINING
# =====================================================

TRAIN_BATCH_SIZE = 8

EVAL_BATCH_SIZE = 8

NUM_EPOCHS = 20

LEARNING_RATE = 1e-5

WEIGHT_DECAY = 0.01

WARMUP_STEPS = 100

LOGGING_STEPS = 5

GRADIENT_ACCUMULATION_STEPS = 1

MAX_GRAD_NORM = 1.0

SAVE_TOTAL_LIMIT = 2

LOAD_BEST_MODEL_AT_END = True

EVALUATION_STRATEGY = "epoch"

SAVE_STRATEGY = "epoch"

# =====================================================
# HARDWARE
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FP16 = torch.cuda.is_available()

# =====================================================
# RANDOM
# =====================================================

SEED = 42