"""
==========================================
QuranAIProjectPaper Configuration
==========================================

All Project Configurations Are In This File.

Author : M. Fatih
Project: Fine-Tuning Whisper For 
Hijaiyah Letters Recognition
"""

from dataclasses import dataclass
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
# HIJAIYAH CLASSES
# =====================================================


@dataclass(frozen=True)
class HijaiyahClass:
    """The canonical dataset and display details for one recognised letter."""

    folder: str
    transcription: str
    display_name: str


# Keep this order consistent everywhere the project presents a letter menu.
HIJAIYAH_CLASSES = (
    HijaiyahClass("alif", "اَ", "Alif"),
    HijaiyahClass("ba", "بَ", "Ba"),
    HijaiyahClass("ta", "تَ", "Ta"),
    HijaiyahClass("tsa", "ثَ", "Tsa"),
    HijaiyahClass("ja", "جَ", "Ja"),
    HijaiyahClass("ha", "حَ", "Ha"),
    HijaiyahClass("kha", "خَ", "Kha"),
    HijaiyahClass("da", "دَ", "Da"),
    HijaiyahClass("dza", "ذَ", "Dza"),
    HijaiyahClass("ra", "رَ", "Ra"),
    HijaiyahClass("za", "زَ", "Za"),
    HijaiyahClass("sa", "سَ", "Sa"),
    HijaiyahClass("sya", "شَ", "Sya"),
    HijaiyahClass("sha", "صَ", "Sha"),
    HijaiyahClass("dha", "ضَ", "Dha"),
    HijaiyahClass("tho", "طَ", "Tho"),
    HijaiyahClass("zha", "ظَ", "Zha"),
    HijaiyahClass("ain", "عَ", "Ain"),
    HijaiyahClass("gha", "غَ", "Gha"),
    HijaiyahClass("fa", "فَ", "Fa"),
    HijaiyahClass("qo", "قَ", "Qo"),
    HijaiyahClass("ka", "كَ", "Ka"),
    HijaiyahClass("la", "لَ", "La"),
    HijaiyahClass("ma", "مَ", "Ma"),
    HijaiyahClass("na", "نَ", "Na"),
    HijaiyahClass("hha", "هَ", "Hha"),
    HijaiyahClass("wa", "وَ", "Wa"),
    HijaiyahClass("ya", "يَ", "Ya"),
)

EXPECTED_TRANSCRIPTIONS = tuple(
    letter.transcription for letter in HIJAIYAH_CLASSES
)

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

CROSS_VALIDATION_FOLDS = 5

# =====================================================
# HARDWARE
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FP16 = torch.cuda.is_available()

# =====================================================
# RANDOM
# =====================================================

SEED = 42
