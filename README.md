# Named Entity Recognition for News Articles

A Natural Language Processing project that identifies and classifies named entities from news text using both model-based and rule-based approaches.

The project is built using Python, spaCy, Pandas, and Streamlit.

---

## Project Overview

Named Entity Recognition (NER) is an NLP task used to detect important entities inside text such as:

- People
- Organizations
- Countries
- Cities
- Dates
- Events
- Products
- Nationalities

This project processes news text and extracts named entities using two different approaches:

1. Model-Based NER using spaCy
2. Rule-Based NER using spaCy EntityRuler

The results of both approaches are analyzed and compared.

---

## Dataset

The project uses the CoNLL-2003 Named Entity Recognition dataset.

The original dataset contains token-level annotations including:

- Word
- POS Tag
- Chunk Tag
- NER Tag

The training data is converted into a structured CSV format and reconstructed into full sentences before applying NER.

---

## Project Features

- Load and process CoNLL-2003 NER data
- Convert token-level data into structured CSV format
- Reconstruct complete sentences
- Model-based Named Entity Recognition using spaCy
- Rule-based Named Entity Recognition using EntityRuler
- Extract entity text, label, start position, and end position
- Analyze entity frequency and distribution
- Compare model-based and rule-based NER
- Visualize entities using spaCy displaCy
- Interactive Streamlit dashboard
- Display entity statistics and charts

---

## Project Structure

```text
ner-news-project/
│
├── data/
│   └── ner_dataset.csv/
│       ├── eng.train
│       ├── eng.testa
│       └── eng.testb
│
├── outputs/
│   ├── train_dataset.csv
│   ├── processed_sentences.csv
│   ├── model_entities.csv
│   ├── rule_based_entities.csv
│   ├── ner_comparison.csv
│   └── ner_visualization.html
│
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── model_ner.py
│   ├── rule_based_ner.py
│   ├── entity_analysis.py
│   ├── compare_ner.py
│   └── visualize_ner.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md