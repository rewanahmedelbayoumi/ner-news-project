import pandas as pd
import spacy


# Load the pre-trained spaCy English NER model
nlp = spacy.load("en_core_web_sm")


def extract_entities(text):
    """
    Extract named entities from a given text using spaCy.
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


def main():
    input_path = "outputs/processed_sentences.csv"

    print("Loading processed sentences...")

    df = pd.read_csv(input_path)

    print("Processed sentences loaded successfully.")

    print("\nUsing first sentence for testing:")

    sample_text = df.iloc[0]["text"]

    print(sample_text)

    print("\nDetected entities:")

    entities = extract_entities(sample_text)

    if len(entities) == 0:
        print("No entities detected in this sentence.")
    else:
        for entity in entities:
            print(
                f"Entity: {entity['entity']} | "
                f"Label: {entity['label']} | "
                f"Start: {entity['start']} | "
                f"End: {entity['end']}"
            )


if __name__ == "__main__":
    main()