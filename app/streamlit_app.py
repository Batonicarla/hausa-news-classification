import numpy as np
import pandas as pd
import streamlit as st
from classifier import HausaNewsClassifier

MODEL_ID = "Batonicarla1/hausa-news-classifier"

st.set_page_config(page_title="Hausa News Topic Classifier", page_icon="📰")


@st.cache_resource(show_spinner="Loading the model (the first start takes a minute)...")
def get_classifier():
    return HausaNewsClassifier(MODEL_ID)


clf = get_classifier()

EXAMPLES = {
    "Health": "Lafiya Zinariya: Halin da ma'aikatan lafiya ke ciki a annobar cutar korona. Ku latsa lasifikar d ake sama don ssauraron cikakken shirin wanda Habiba Adamu ta gabatar. Wani bincike da aka wallafa a mujallar kiwon lafiya ta The Lancet ya gano cewa, ma'aikatan kiwon lafiya na fuskantar karin barazanar kamuwa da cutar Covid-19.",
    "Politics": "Kwankwaso ba shi da alkibla – Ganduje. Gwamnan jihar Kano da ke Najeriya, Abdullahi Umar Ganduje, ya bayyana tsohon gwamnan jihar Sanata Rabi'u Musa Kwankwaso a matsayin dan siyasa 'mara alkibla' wanda ba shi da 'tsari'.",
    "Technology": "Amurka na bincike kan wasu abubuwa da ke yawo a sararin samaniyarta. Wani rahoton gwamnatin Amurka kan ganin wasu abubuwa masu tashi sama da ba a san ko meye ba (UFOs), ya gano cewa babu wata shaidar cewa akwai wasu halittu na 'alien' amma ba su cire tsammanin akwai ba.",
    "Failure case (business, predicted religion)": "Asarar da Saudiyya ke yi saboda rashin mahajjata daga kasashen duniya. Saudiyya na daga cikin manyan kasashen duniya da annobar cutar korona ta haifar da koma baya gahanyoyin samun kudaden shigarta. Saudiyya na samum ɗumbin kudaden shiga a lokutan ayyukan Hajji da Umara da miliyoyin Musulmai kan halarta a duk shekara daga kasashen duniya.",
}

st.title("Hausa News Topic Classifier")
st.write("Classifies Hausa news text into 7 topics: business, entertainment, health, politics, "
         "religion, sports, technology. Fine-tuned on MasakhaNEWS (BBC Hausa). It only knows these "
         "7 topics and was trained on news articles, not social media.")

if "text" not in st.session_state:
    st.session_state["text"] = ""
choice = st.selectbox("Try an example", ["(none)"] + list(EXAMPLES))
if st.button("Load example", key="load") and choice != "(none)":
    st.session_state["text"] = EXAMPLES[choice]

text = st.text_area("Hausa news text (headline and/or article)", key="text", height=220,
                    placeholder="Paste a Hausa news headline or article here...")

if st.button("Classify", type="primary", key="classify"):
    if not text or len(text.split()) < 3:
        st.warning("Please enter a Hausa news headline or article (at least a few words).")
    else:
        probs, n_total, budget = clf.predict_probs(text)
        top = int(np.argmax(probs))
        st.success(f"Predicted topic: {clf.labels[top]} (confidence {probs[top]:.0%})")
        if probs[top] < 0.5:
            st.info("Low confidence: the text may mix topics or fall outside these categories.")
        if len(text.split()) < 10:
            st.info("Very short input: the model was trained on full news articles, so short text is less reliable.")
        if n_total > budget:
            st.info(f"Long text: only {budget} of {n_total} tokens were used ({clf.strategy} truncation).")
        st.bar_chart(pd.DataFrame({"probability": probs}, index=clf.labels))

with st.expander("About this model"):
    st.markdown("""
**Model:** AfroXLMR-base (`Davlan/afro-xlmr-base`) fine-tuned for 7-topic classification.

**Data:** Hausa part of MasakhaNEWS (BBC Hausa): 2,219 training, 317 validation, and 637 test articles.

**Test results:** accuracy 0.906, macro-F1 0.898 (baseline TF-IDF + logistic regression: 0.887 / 0.875).

**Input:** Only the first 256 tokens of the text are used. Very short text is less reliable.

**Limits:** It knows only these 7 topics, was trained on BBC News only, and can be confidently wrong on mixed-topic or ambiguous articles.

**Links:** [Model on Hugging Face](https://huggingface.co/Batonicarla1/hausa-news-classifier) · [Code on GitHub](https://github.com/Batonicarla/hausa-news-classification)
""")
