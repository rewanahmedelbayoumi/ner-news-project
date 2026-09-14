import pandas as pd


def reconstruct_sentences(input_path="outputs/train_dataset.csv"):
    """
    Reconstruct full sentences from the CoNLL-2003 token-level dataset.
    """

    df = pd.read_csv(input_path)

    # Remove rows where the word is missing
    df = df.dropna(subset=["Word"])

    # Make sure Word and Tag columns are strings
    df["Word"] = df["Word"].astype(str)
    df["Tag"] = df["Tag"].fillna("O").astype(str)

    sentences = []

    grouped = df.groupby("Sentence #")

    for sentence_id, group in grouped:
        words = group["Word"].tolist()
        tags = group["Tag"].tolist()

        sentence_text = " ".join(words)

        sentences.append(
            {
                "sentence_id": sentence_id,
                "text": sentence_text,
                "tags": " ".join(tags),
            }
        )

    sentences_df = pd.DataFrame(sentences)

    return sentences_df


def main():
    print("Reconstructing sentences from CoNLL tokens...")

    sentences_df = reconstruct_sentences()

    print("\nSentence reconstruction completed successfully.")

    print("\nTotal sentences:")
    print(len(sentences_df))

    print("\nFirst 5 reconstructed sentences:")
    print(sentences_df.head())

    output_path = "outputs/processed_sentences.csv"

    sentences_df.to_csv(output_path, index=False)

    print(f"\nProcessed sentences saved to:")
    print(output_path)


if __name__ == "__main__":
    main()