import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import zipfile
import textstat
from io import StringIO, BytesIO

from modules.processor import AdvancedTranscriptProcessor, AdvancedReportGenerator, DataPreprocessor

# Optional imports
try:
    from fpdf import FPDF
    pdf_available = True
except ImportError:
    pdf_available = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    SUMMARIZER_MODEL = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    summarizer_available = True
except Exception:
    summarizer_available = False

# Page configuration
st.set_page_config(page_title="Smart Medical Dashboard", page_icon="🩺", layout="wide")

# Theme color pickers
st.sidebar.header("🎨 Customize Theme")
primary_color = st.sidebar.color_picker("Primary Color", "#1e3a8a")
secondary_color = st.sidebar.color_picker("Secondary Color", "#3b82f6")
background_color = st.sidebar.color_picker("Background Color", "#f8fafc")
text_color = st.sidebar.color_picker("Text Color", "#1e293b")

# Inject custom CSS
st.markdown(f"""
<style>
.reportview-container .main .block-container {{
    background-color: {background_color};
    color: {text_color};
    padding: 1rem;
}}
.stSidebar {{
    background-color: {primary_color};
}}
.stButton>button {{
    background-color: {secondary_color} !important;
    color: white !important;
    border-radius: 8px;
    padding: 0.5rem 1rem;
}}
.stDownloadButton>button {{
    background-color: {primary_color} !important;
    color: white !important;
    border-radius: 8px;
    padding: 0.5rem 1rem;
}}
h1, h2, h3, h4, h5, h6 {{
    color: {primary_color};
}}
</style>
""", unsafe_allow_html=True)

# Title
st.title("🧠 Smart Medical Dashboard")
st.markdown("Upload a CSV with a 'transcription' column to generate AI-powered summaries and insights.")

# Navigation
menu = st.sidebar.radio("Navigate", ["Upload & Process", "Insights", "Export"])

if menu == "Upload & Process":
    st.header("🚀 Upload & Process")
    uploaded = st.file_uploader("Select transcript CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(df.head())
        if 'transcription' not in df.columns:
            st.error("CSV must contain 'transcription' column.")
            st.stop()
        if df['transcription'].dropna().empty:
            st.warning("'transcription' column is empty.")
            st.stop()
        st.success(f"Loaded {len(df)} records.")
        # Initialize storage
        sentiments, scores, risks, topics = [], [], [], {}
        reports, pdfs = [], []
        for idx, row in df.iterrows():
            text = str(row['transcription'])
            clean = DataPreprocessor.preprocess(text)
            proc = AdvancedTranscriptProcessor(clean)
            gen = AdvancedReportGenerator(proc)
            sentiments.append(proc.sentiment)
            score = textstat.flesch_reading_ease(clean)
            scores.append(score)
            risk = gen._risk()
            risks.append(risk)
            for t in proc.topics:
                topics[t] = topics.get(t, 0) + 1
            clin = gen.clinician_text()
            pat = gen.patient_text(detail='low')
            reports.append((f"case_{idx+1}_clinician.txt", clin))
            reports.append((f"case_{idx+1}_patient.txt", pat))
            if pdf_available:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 8, clin + "\n\n" + pat)
                pdf_bytes = pdf.output(dest='S').encode('latin1')
                pdfs.append((f"case_{idx+1}.pdf", pdf_bytes))
            st.subheader(f"Case {idx+1}")
            st.markdown("**Clinician Report:**")
            st.code(clin, language="text")
            st.markdown("**Patient Summary:**")
            st.code(pat, language="text")
        st.session_state['stats'] = {'sentiments': sentiments, 'scores': scores, 'risks': risks, 'topics': topics}
        st.session_state['reports'] = reports
        st.session_state['pdfs'] = pdfs

elif menu == "Insights":
    if 'stats' not in st.session_state:
        st.info("Upload data first.")
    else:
        st.header("📈 Insights")
        stats = st.session_state['stats']
        df_sent = pd.DataFrame(stats['sentiments'])
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sentiment Distribution")
            st.plotly_chart(px.box(df_sent, points='all'), use_container_width=True)
        with col2:
            st.subheader("Readability Scores")
            st.plotly_chart(px.histogram(stats['scores'], nbins=10), use_container_width=True)
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Risk Breakdown")
            counts = pd.Series(stats['risks']).value_counts()
            st.plotly_chart(px.pie(names=counts.index, values=counts.values), use_container_width=True)
        with col4:
            st.subheader("Top Topics")
            df_top = pd.DataFrame.from_dict(stats['topics'], orient='index', columns=['count']).reset_index()
            df_top.columns = ['topic','count']
            st.plotly_chart(px.bar(df_top, x='topic', y='count'), use_container_width=True)

elif menu == "Export":
    st.header("📤 Export")
    if 'reports' not in st.session_state:
        st.info("No reports to export.")
    else:
        buf = BytesIO()
        with zipfile.ZipFile(buf,'w') as zf:
            for name, content in st.session_state['reports']:
                zf.writestr(name, content)
        buf.seek(0)
        st.download_button("Download Text Reports", buf, file_name="reports.zip")
        if pdf_available and st.session_state['pdfs']:
            buf2 = BytesIO()
            with zipfile.ZipFile(buf2,'w') as zf2:
                for name, content in st.session_state['pdfs']:
                    zf2.writestr(name, content)
            buf2.seek(0)
            st.download_button("Download PDF Reports", buf2, file_name="reports_pdf.zip")
