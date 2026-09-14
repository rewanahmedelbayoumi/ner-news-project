import pandas as pd


def analyze_entities(input_path="outputs/model_entities.csv"):
    """
    Analyze extracted named entities and display
    useful statistics about entity labels and frequency.
    """

    print("Loading extracted entities...")

    df = pd.read_csv(input_path)

    print("Entities loaded successfully.")

    print("\nTotal detected entities:")
    print(len(df))

    print("\nEntity label distribution:")
    print(df["label"].value_counts())

    print("\nTop 15 most frequent entities:")
    print(df["entity"].value_counts().head(15))

    print("\nTop PERSON entities:")
    person_entities = df[df["label"] == "PERSON"]["entity"].value_counts().head(10)

    if len(person_entities) > 0:
        print(person_entities)
    else:
        print("No PERSON entities found.")

    print("\nTop ORG entities:")
    organization_entities = (
        df[df["label"] == "ORG"]["entity"]
        .value_counts()
        .head(10)
    )

    if len(organization_entities) > 0:
        print(organization_entities)
    else:
        print("No ORG entities found.")

    print("\nTop GPE entities:")
    location_entities = (
        df[df["label"] == "GPE"]["entity"]
        .value_counts()
        .head(10)
    )

    if len(location_entities) > 0:
        print(location_entities)
    else:
        print("No GPE entities found.")


def main():
    analyze_entities()


if __name__ == "__main__":
    main()