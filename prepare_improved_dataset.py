import pandas as pd
from pathlib import Path


# ==========================================================
# PATH CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

RAW_DIR = DATA_DIR / "raw"

PROCESSED_DIR = DATA_DIR / "processed"

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# LOAD ORIGINAL ISOT DATASET
# ==========================================================

print("=" * 60)
print("LOADING ORIGINAL DATASETS")
print("=" * 60)


fake_path = DATA_DIR / "Fake.csv"

true_path = DATA_DIR / "True.csv"


fake_df = pd.read_csv(
    fake_path
)

true_df = pd.read_csv(
    true_path
)


print(
    "Original Fake:",
    len(fake_df)
)

print(
    "Original Real:",
    len(true_df)
)


# ==========================================================
# STANDARDIZE FAKE DATA
# ==========================================================

fake_data = pd.DataFrame()


fake_data["text"] = (
    fake_df["title"].fillna("")
    + " "
    + fake_df["text"].fillna("")
)


fake_data["label"] = 0


# ==========================================================
# STANDARDIZE REAL DATA
# ==========================================================

real_data = pd.DataFrame()


real_data["text"] = (
    true_df["title"].fillna("")
    + " "
    + true_df["text"].fillna("")
)


real_data["label"] = 1


datasets = [
    fake_data,
    real_data
]


# ==========================================================
# LOAD RECENT DATASET
# ==========================================================

recent_path = (
    RAW_DIR
    / "recent_fake_news.csv"
)


if recent_path.exists():

    print("\nLoading fresh dataset...")


    recent_data = pd.read_csv(
        recent_path
    )


    recent_data = recent_data[
        ["text", "label"]
    ]


    datasets.append(
        recent_data
    )


    print(
        "Fresh samples:",
        len(recent_data)
    )


else:

    print(
        "\nWARNING: Fresh dataset not found."
    )


# ==========================================================
# COMBINE ALL DATASETS
# ==========================================================

print("\nCombining datasets...")


dataset = pd.concat(
    datasets,
    ignore_index=True
)


# ==========================================================
# DATA CLEANING
# ==========================================================

print("Cleaning dataset...")


dataset = dataset.dropna()


dataset["text"] = (
    dataset["text"]
    .astype(str)
    .str.strip()
)


# Remove short content

dataset = dataset[
    dataset["text"].str.len() > 30
]


# Remove duplicates

dataset = dataset.drop_duplicates(
    subset=["text"]
)


# Ensure valid labels

dataset = dataset[
    dataset["label"].isin([0, 1])
]


dataset["label"] = (
    dataset["label"]
    .astype(int)
)


# ==========================================================
# SHUFFLE
# ==========================================================

dataset = dataset.sample(
    frac=1,
    random_state=42
).reset_index(
    drop=True
)


# ==========================================================
# FINAL INFORMATION
# ==========================================================

print("\n" + "=" * 60)
print("FINAL IMPROVED DATASET")
print("=" * 60)


print(
    "Total Samples:",
    len(dataset)
)


print("\nClass Distribution:")


print(
    dataset["label"]
    .value_counts()
)


# ==========================================================
# SAVE DATASET
# ==========================================================

output_path = (
    PROCESSED_DIR
    / "improved_dataset.csv"
)


dataset.to_csv(
    output_path,
    index=False
)


print(
    "\nDataset saved successfully."
)


print(
    output_path
)