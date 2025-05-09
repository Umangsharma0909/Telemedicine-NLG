import streamlit as st
import pandas as pd

from modules.processor import AdvancedTranscriptProcessor, AdvancedReportGenerator, DataPreprocessor

st.set_page_config(page_title="Smart Medical Dashboard", page_icon="🩺", layout="wide")

st.title("🧠 Smart Medical Dashboard")
st.markdown("Upload a CSV with a 'transcription' column to see generated reports.")

uploaded = st.file_uploader("Upload transcript CSV", type=["csv"])
if uploaded:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    if 'transcription' not in df.columns:
        st.error("CSV must contain a 'transcription' column.")
        st.stop()
    if df['transcription'].dropna().empty:
        st.warning("The 'transcription' column is empty.")
        st.stop()

    st.success(f"Loaded {len(df)} records.")
    st.write(df.head())

    # Process first record as example
    idx = 0
    text = df['transcription'].iloc[idx]
    clean = DataPreprocessor.preprocess(text)
    proc = AdvancedTranscriptProcessor(clean)
    gen = AdvancedReportGenerator(proc)

    st.header(f"📄 Clinician Report (Case {idx+1})")
    st.code(gen.clinician_text(), language="markdown")

    st.header(f"🩺 Patient Summary (Case {idx+1})")
    st.code(gen.patient_text(), language="text")

    # Allow viewing subsequent cases
    if len(df) > 1:
        case = st.number_input("Select case index", min_value=1, max_value=len(df), value=1)
        if case-1 != idx:
            text2 = df['transcription'].iloc[case-1]
            clean2 = DataPreprocessor.preprocess(text2)
            proc2 = AdvancedTranscriptProcessor(clean2)
            gen2 = AdvancedReportGenerator(proc2)
            st.header(f"📄 Clinician Report (Case {case})")
            st.code(gen2.clinician_text(), language="markdown")
            st.header(f"🩺 Patient Summary (Case {case})")
            st.code(gen2.patient_text(), language="text")
