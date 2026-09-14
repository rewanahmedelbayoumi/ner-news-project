import pandas as pd


def load_results():
    """
    Load model-based and rule-based NER results.
    """

    model_df = pd.read_csv("outputs/model_entities.csv")
    rule_df = pd.read_csv("outputs/rule_based_entities.csv")

    return model_df, rule_df


def compare_results(model_df, rule_df):
    """
    Compare model-based NER with rule-based NER.
    """

    print("NER Comparison")
    print("=" * 50)

    print("\nModel-Based NER")
    print("-" * 50)

    print("Total detected entities:")
    print(len(model_df))

    print("\nEntity label distribution:")
    print(model_df["label"].value_counts())

    print("\nRule-Based NER")
    print("-" * 50)

    print("Total detected entities:")
    print(len(rule_df))

    print("\nEntity label distribution:")
    print(rule_df["label"].value_counts())

    print("\nComparison Summary")
    print("-" * 50)

    model_total = len(model_df)
    rule_total = len(rule_df)

    print(f"Model-Based total entities: {model_total}")
    print(f"Rule-Based total entities: {rule_total}")

    difference = model_total - rule_total

    print(f"Difference in detected entities: {difference}")

    model_labels = set(model_df["label"].unique())
    rule_labels = set(rule_df["label"].unique())

    print("\nLabels detected by Model-Based NER:")
    print(sorted(model_labels))

    print("\nLabels detected by Rule-Based NER:")
    print(sorted(rule_labels))

    common_labels = model_labels.intersection(rule_labels)

    print("\nCommon labels:")
    print(sorted(common_labels))


def save_comparison(model_df, rule_df):
    """
    Save a simple comparison summary to CSV.
    """

    summary = pd.DataFrame(
        {
            "Method": [
                "Model-Based NER",
                "Rule-Based NER"
            ],
            "Total Entities": [
                len(model_df),
                len(rule_df)
            ],
            "Unique Labels": [
                model_df["label"].nunique(),
                rule_df["label"].nunique()
            ]
        }
    )

    output_path = "outputs/ner_comparison.csv"

    summary.to_csv(output_path, index=False)

    print("\nComparison summary saved to:")
    print(output_path)


def main():
    print("Loading NER results...")

    model_df, rule_df = load_results()

    print("NER results loaded successfully.")

    compare_results(model_df, rule_df)

    save_comparison(model_df, rule_df)


if __name__ == "__main__":
    main()