from datasets import load_dataset
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

RAW_DIR = BASE_DIR / "data" / "raw"

RAW_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 60)
print("DOWNLOADING RECENT FAKE NEWS DATASET")
print("=" * 60)


try:

    dataset = load_dataset(
        "Arko007/fake-news-dataset",
        split="train",
        streaming=True
    )


    fake_samples = []
    real_samples = []


    MAX_PER_CLASS = 5000


    print(
        "Streaming dataset and collecting balanced samples..."
    )


    for item in dataset:

        text = str(
            item.get("text", "")
        ).strip()


        label = item.get(
            "label_binary",
            None
        )


        if not text or label is None:
            continue


        label = int(label)


        if (
            label == 0
            and len(fake_samples) < MAX_PER_CLASS
        ):

            fake_samples.append(
                {
                    "text": text,
                    "label": 0
                }
            )


        elif (
            label == 1
            and len(real_samples) < MAX_PER_CLASS
        ):

            real_samples.append(
                {
                    "text": text,
                    "label": 1
                }
            )


        if (
            len(fake_samples) >= MAX_PER_CLASS
            and len(real_samples) >= MAX_PER_CLASS
        ):

            break


    samples = (
        fake_samples
        + real_samples
    )


    df = pd.DataFrame(
        samples
    )


    # Shuffle

    df = df.sample(
        frac=1,
        random_state=42
    ).reset_index(
        drop=True
    )


    output_path = (
        RAW_DIR
        / "recent_fake_news.csv"
    )


    df.to_csv(
        output_path,
        index=False
    )


    print("\nDOWNLOAD SUCCESSFUL")

    print(
        "Total samples:",
        len(df)
    )

    print(
        "Fake samples:",
        len(
            df[df["label"] == 0]
        )
    )

    print(
        "Real samples:",
        len(
            df[df["label"] == 1]
        )
    )

    print(
        "\nSaved to:"
    )

    print(
        output_path
    )


except Exception as error:

    print(
        "\nERROR:"
    )

    print(error)