from datasets import load_dataset

from QuranAIProjectpaper.config import TRAIN_CSV
from QuranAIProjectpaper.config import TEST_CSV

dataset = load_dataset(
    "csv",
    data_files={
        "train": str(TRAIN_CSV),
        "test": str(TEST_CSV)
    }
)

print(dataset)

print()

print(dataset["train"][0])