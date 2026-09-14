import pandas as pd
import spacy


# Load the pre-trained spaCy English NER model
nlp = spacy.load("en_core_web_sm")


def extract_entities(text):
    """
    Extract named entities from text using spaCy.
    """

    doc = nlp(str(text))

    entities = []

    for ent in doc.ents:
        entities.append(
            {
                "entity": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            }
        )

    return entities


def process_sentences(
    input_path="outputs/processed_sentences.csv",
    output_path="outputs/model_entities.csv",
    limit=500
):
    """
    Run spaCy NER on multiple processed sentences
    and save detected entities to a CSV file.
    """

    print("Loading processed sentences...")

    df = pd.read_csv(input_path)

    print("Processed sentences loaded successfully.")

    results = []

    print(f"\nProcessing first {limit} sentences...")

    for _, row in df.head(limit).iterrows():
        sentence_id = row["sentence_id"]
        text = row["text"]

        entities = extract_entities(text)

        for entity in entities:
            results.append(
                {
                    "sentence_id": sentence_id,
                    "text": text,
                    "entity": entity["entity"],
                    "label": entity["label"],
                    "start": entity["start"],
                    "end": entity["end"],
                }
            )

    results_df = pd.DataFrame(results)

    results_df.to_csv(output_path, index=False)

    return results_df


def main():
    results_df = process_sentences()

    print("\nNER processing completed successfully.")

    print("\nTotal detected entities:")
    print(len(results_df))

    print("\nFirst 20 detected entities:")
    print(results_df.head(20))

    print("\nEntity label distribution:")
    if not results_df.empty:
        print(results_df["label"].value_counts())
    else:
        print("No entities were detected.")

    print("\nResults saved to:")
    print("outputs/model_entities.csv")


if __name__ == "__main__":
    main()