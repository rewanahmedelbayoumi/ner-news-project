import streamlit as st
import spacy
from spacy import displacy
from collections import Counter


st.set_page_config(
    page_title="NER Intelligence",
    page_icon="🧠",
    layout="wide"
)


@st.cache_resource
def load_model():
    return spacy.load("en_core_web_sm")


nlp = load_model()


st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #020617 100%);
        color: #f8fafc;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero-card {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 24px;
        padding: 34px;
        margin-bottom: 28px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.30);
        backdrop-filter: blur(12px);
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 8px;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #cbd5e1;
        margin-bottom: 0;
        line-height: 1.7;
    }

    .section-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 12px 35px rgba(0,0,0,0.20);
        margin-bottom: 20px;
    }

    .metric-box {
        background: linear-gradient(145deg, rgba(30,41,59,0.95), rgba(15,23,42,0.95));
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 18px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
    }

    .metric-number {
        font-size: 2rem;
        font-weight: 800;
        color: #a5b4fc;
    }

    .metric-label {
        font-size: 0.95rem;
        color: #cbd5e1;
        margin-top: 5px;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 14px;
        height: 3.2rem;
        font-size: 1rem;
        font-weight: 700;
        border: none;
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        color: white;
        transition: 0.3s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(99,102,241,0.35);
    }

    textarea {
        border-radius: 16px !important;
    }

    .entity-pill {
        display: inline-block;
        padding: 7px 12px;
        margin: 5px;
        border-radius: 999px;
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(129,140,248,0.25);
        color: #e0e7ff;
        font-size: 0.9rem;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 35px;
    }

    .stDataFrame {
        border-radius: 16px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">NER Intelligence</div>
        <div class="hero-subtitle">
            Intelligent Named Entity Recognition for news and text analysis.
            Extract people, organizations, countries, dates, locations and more
            using spaCy's pre-trained NLP model.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


left_col, right_col = st.columns([1.6, 1])

with left_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    st.subheader("Analyze Text")

    text = st.text_area(
        "Enter news text or article",
        height=250,
        placeholder=(
            "Example: Apple CEO Tim Cook visited London on Monday "
            "to meet representatives from Microsoft and the United Nations."
        ),
        label_visibility="collapsed"
    )

    analyze_button = st.button("Analyze Named Entities")

    st.markdown('</div>', unsafe_allow_html=True)


with right_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    st.subheader("Supported Entity Types")

    entity_types = [
        "PERSON",
        "ORG",
        "GPE",
        "DATE",
        "NORP",
        "MONEY",
        "EVENT",
        "PRODUCT"
    ]

    pills_html = ""

    for entity_type in entity_types:
        pills_html += f'<span class="entity-pill">{entity_type}</span>'

    st.markdown(pills_html, unsafe_allow_html=True)

    st.markdown(
        """
        <br><br>
        <small style="color:#94a3b8;">
        The system identifies multiple entity categories using
        spaCy's English NER model.
        </small>
        """,
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)


if analyze_button:

    if not text.strip():
        st.warning("Please enter some text first.")

    else:
        doc = nlp(text)

        entities = [
            {
                "Entity": ent.text,
                "Label": ent.label_,
                "Start": ent.start_char,
                "End": ent.end_char
            }
            for ent in doc.ents
        ]

        label_counts = Counter(ent.label_ for ent in doc.ents)

        st.markdown("## Analysis Overview")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-number">{len(doc.ents)}</div>
                    <div class="metric-label">Total Entities</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-number">{len(label_counts)}</div>
                    <div class="metric-label">Entity Types</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c3:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-number">{len(text.split())}</div>
                    <div class="metric-label">Words Analyzed</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c4:
            top_label = label_counts.most_common(1)[0][0] if label_counts else "-"
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-number">{top_label}</div>
                    <div class="metric-label">Top Entity Type</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        if entities:
            col1, col2 = st.columns([1.1, 1])

            with col1:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.subheader("Detected Entities")

                st.dataframe(
                    entities,
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.subheader("Entity Distribution")

                count_data = {
                    "Entity Type": list(label_counts.keys()),
                    "Count": list(label_counts.values())
                }

                st.bar_chart(
                    count_data,
                    x="Entity Type",
                    y="Count"
                )

                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-card">', unsafe_allow_html=True)

            st.subheader("Highlighted Entity Visualization")

            html = displacy.render(
                doc,
                style="ent",
                page=False
            )

            st.markdown(
                f"""
                <div style="
                    background:white;
                    border-radius:16px;
                    padding:22px;
                    color:black;
                    overflow-x:auto;
                ">
                    {html}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-card">', unsafe_allow_html=True)

            st.subheader("Entity Summary")

            summary_cols = st.columns(
                min(len(label_counts), 4)
            )

            for i, (label, count) in enumerate(label_counts.items()):
                with summary_cols[i % len(summary_cols)]:
                    st.metric(
                        label=label,
                        value=count
                    )

            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.info(
                "No named entities were detected in the provided text."
            )


st.markdown(
    """
    <div class="footer">
        NER Intelligence • spaCy Named Entity Recognition • NLP Project
    </div>
    """,
    unsafe_allow_html=True
)