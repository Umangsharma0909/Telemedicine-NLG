import streamlit as st
import pandas as pd
import plotly.express as px
import textstat
import zipfile
from io import StringIO, BytesIO

from modules.processor import AdvancedTranscriptProcessor, AdvancedReportGenerator, DataPreprocessor

# Optional imports
try:
    from fpdf import FPDF
    pdf_available = True
except ImportError:
    pdf_available = False

# Page configuration with static theme
st.set_page_config(
    page_title="Smart Medical Dashboard",
    page_icon="🩺",
    layout="wide"
)

# Static CSS for consistent look with animated medical theme
st.markdown("""
<style>
/* Full-page animated gradient background */
@keyframes gradientBG {
  0% {background-position: 0% 50%;}
  50% {background-position: 100% 50%;}
  100% {background-position: 0% 50%;}
}
body {
  margin: 0;
  padding: 0;
  background: linear-gradient(-45deg, #2a9df4, #39c1ed, #8de3f5, #d4f1f9);
  background-size: 400% 400%;
  animation: gradientBG 15s ease infinite;
}
.block-container {
    background-color: rgba(255, 255, 255, 0.85);
    color: #334e68;
    padding: 2rem;
    border-radius: 10px;
}
.stSidebar {
    background-color: rgba(42, 111, 151, 0.85);
}
.stButton>button, .stDownloadButton>button {
    background-color: #38a3a5 !important;
    color: white !important;
    border-radius: 8px;
    padding: 0.5rem 1rem;
}
h1, h2, h3 {
    color: #173f5f;
}
</style>
""", unsafe_allow_html=True)
<style>
.block-container {
    background-color: #f0f4f8;
    color: #334e68;
    padding: 2rem;
}
.stSidebar {
    background-color: #2a6f97;
}
.stButton>button, .stDownloadButton>button {
    background-color: #38a3a5 !important;
    color: white !important;
    border-radius: 8px;
    padding: 0.5rem 1rem;
}
h1, h2, h3 {
    color: #173f5f;
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("🧠 Smart Medical Dashboard")
st.markdown("Upload a CSV with a 'transcription' column to generate AI-powered summaries and insights.")

# Navigation
menu = st.sidebar.radio("Navigate", ["Upload & Process", "Insights", "Export"])

# Upload & Process
if menu == "Upload & Process":
    st.header("🚀 Upload & Process")
    uploaded = st.file_uploader("Select transcript CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        if 'transcription' not in df.columns:
            st.error("CSV must contain a 'transcription' column.")
            st.stop()
        if df['transcription'].dropna().empty:
            st.warning("'transcription' column is empty.")
            st.stop()
        st.success(f"Loaded {len(df)} records.")

        # Initialize storage
        stats = {'sentiments': [], 'scores': [], 'risks': [], 'topics': {}}
        reports, pdfs = [], []

        for idx, row in df.iterrows():
            text = str(row['transcription'])
            clean = DataPreprocessor.preprocess(text)
            proc = AdvancedTranscriptProcessor(clean)
            gen = AdvancedReportGenerator(proc)

            # Collect stats
            stats['sentiments'].append(proc.sentiment)
            score = textstat.flesch_reading_ease(clean)
            stats['scores'].append(score)
            risk = gen._risk()
            stats['risks'].append(risk)
            for topic in proc.topics:
                stats['topics'][topic] = stats['topics'].get(topic, 0) + 1

            # Generate outputs
            clin = gen.clinician_text()
            pat = gen.patient_text(detail='low')
            reports.extend([(f"case_{idx+1}_clinician.txt", clin), (f"case_{idx+1}_patient.txt", pat)])

            # PDF if available
            if pdf_available:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 8, clin + "\n\n" + pat)
                pdf_bytes = pdf.output(dest='S').encode('latin1')
                pdfs.append((f"case_{idx+1}.pdf", pdf_bytes))

            # Display
            st.subheader(f"Case {idx+1}")
            st.markdown("**Clinician Report:**")
            st.code(clin, language="text")
            st.markdown("**Patient Summary:**")
            st.code(pat, language="text")

        # Save to session
        st.session_state['stats'] = stats
        st.session_state['reports'] = reports
        st.session_state['pdfs'] = pdfs

# Insights
elif menu == "Insights":
    if 'stats' not in st.session_state:
        st.info("Please upload and process data first.")
    else:
        st.header("📈 Insights")
        stats = st.session_state['stats']
        df_sent = pd.DataFrame(stats['sentiments'])
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Sentiment Distribution")
            st.plotly_chart(px.box(df_sent, points='all'), use_container_width=True)
        with c2:
            st.subheader("Readability Scores")
            st.plotly_chart(px.histogram(stats['scores'], nbins=10), use_container_width=True)
        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Risk Breakdown")
            counts = pd.Series(stats['risks']).value_counts()
            st.plotly_chart(px.pie(names=counts.index, values=counts.values), use_container_width=True)
        with c4:
            st.subheader("Top Topics")
            df_top = pd.DataFrame.from_dict(stats['topics'], orient='index', columns=['count']).reset_index()
            df_top.columns = ['topic','count']
            st.plotly_chart(px.bar(df_top, x='topic', y='count'), use_container_width=True)

# Export
elif menu == "Export":
    st.header("📤 Export")
    if 'reports' not in st.session_state:
        st.info("No reports available to export.")
    else:
        buf = BytesIO()
        with zipfile.ZipFile(buf, 'w') as zf:
            for name, content in st.session_state['reports']:
                zf.writestr(name, content)
        buf.seek(0)
        st.download_button("Download Text Reports", buf, file_name="reports.zip")
        if pdf_available and st.session_state['pdfs']:
            buf2 = BytesIO()
            with zipfile.ZipFile(buf2, 'w') as zf2:
                for name, content in st.session_state['pdfs']:
                    zf2.writestr(name, content)
            buf2.seek(0)
            st.download_button("Download PDF Reports", buf2, file_name="reports_pdf.zip")
