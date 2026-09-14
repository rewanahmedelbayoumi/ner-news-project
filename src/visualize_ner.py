import pandas as pd
import spacy
from spacy import displacy


def main():
    print("Loading spaCy model...")

    nlp = spacy.load("en_core_web_sm")

    print("Loading processed sentences...")

    df = pd.read_csv("outputs/processed_sentences.csv")

    sample_text = df.iloc[0]["text"]

    print("\nSample sentence:")
    print(sample_text)

    doc = nlp(sample_text)

    print("\nDetected entities:")

    if len(doc.ents) == 0:
        print("No entities detected.")
    else:
        for ent in doc.ents:
            print(f"{ent.text} -> {ent.label_}")

    html = displacy.render(
        doc,
        style="ent",
        page=True
    )

    output_path = "outputs/ner_visualization.html"

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(html)

    print("\nVisualization saved successfully.")
    print(f"Open this file in your browser: {output_path}")


if __name__ == "__main__":
    main()