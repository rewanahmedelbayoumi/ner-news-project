import pandas as pd
from pathlib import Path


def load_conll_file(path):
    """
    Load a CoNLL-2003 formatted file.

    Each row contains:
    Word POS Chunk NER_Tag

    Empty lines separate sentences.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    rows = []
    sentence_id = 1

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            # Empty line means the end of a sentence
            if not line:
                sentence_id += 1
                continue

            # Skip CoNLL document markers
            if line.startswith("-DOCSTART-"):
                continue

            parts = line.split()

            if len(parts) >= 4:
                word = parts[0]
                pos = parts[1]
                chunk = parts[2]
                tag = parts[3]

                rows.append(
                    {
                        "Sentence #": sentence_id,
                        "Word": word,
                        "POS": pos,
                        "Chunk": chunk,
                        "Tag": tag,
                    }
                )

    df = pd.DataFrame(rows)

    return df


def main():
    dataset_path = "data/ner_dataset.csv/eng.train"

    print("Loading CoNLL-2003 dataset...")

    df = load_conll_file(dataset_path)

    print("\nDataset loaded successfully.")

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 10 rows:")
    print(df.head(10))

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nNER tag distribution:")
    print(df["Tag"].value_counts())

    output_path = "outputs/train_dataset.csv"

    df.to_csv(output_path, index=False)

    print(f"\nConverted dataset saved to:")
    print(output_path)


if __name__ == "__main__":
    main()