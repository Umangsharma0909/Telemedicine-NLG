import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import zipfile
import textstat
from io import StringIO, BytesIO

# Optional imports
try:
    from fpdf import FPDF
    pdf_available = True
except ImportError:
    pdf_available = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    # initialize summarizer
    SUMMARIZER_MODEL = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    summarizer_available = True
except Exception:
    summarizer_available = False

# Local processor fallback
from modules.processor import (
    AdvancedTranscriptProcessor,
    AdvancedReportGenerator,
    DataPreprocessor
)

# Wrapper for summarization
def safe_summarize(text: str, max_length: int = 80, min_length: int = 20) -> str:
    if not summarizer_available:
        # Fallback: return first min_length words
        return ' '.join(text.split()[:min_length]) + '...'
    # adjust max_length to input
    words = text.split()
    max_len = min(max_length, max(5, len(words)-1))
    try:
        result = summarizer(text, max_length=max_len, min_length=min(min_length, max_len), do_sample=False)
        return result[0].get('summary_text', text)
    except Exception:
        return text

st.set_page_config(
    page_title="Smart Medical Dashboard",
    page_icon="🩺",
    layout="wide"
)

st.title("🧠 Smart Medical Dashboard")

menu = st.sidebar.radio("Navigation", ["Upload & Process", "Insights", "Export"])

if menu == "Upload & Process":
    file = st.file_uploader("Upload CSV with 'transcription' column", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.write(df.head())
        if 'transcription' not in df.columns:
            st.error("CSV must contain 'transcription' column.")
            st.stop()
        if df['transcription'].dropna().empty:
            st.error("Column is empty.")
            st.stop()
        st.success(f"Loaded {len(df)} records.")

        # data stores
        sentiments, scores, risks, topics = [], [], [], {}
        reports, pdfs = [], []

        for i, row in df.iterrows():
            text = str(row['transcription'])
            clean = DataPreprocessor.preprocess(text)
            proc = AdvancedTranscriptProcessor(clean)
            gen = AdvancedReportGenerator(proc)

            # collect stats
            sent = proc.sentiment
            sentiments.append(sent)
            fl_score = textstat.flesch_reading_ease(clean)
            scores.append(fl_score)
            risk = gen._risk()
            risks.append(risk)
            for t in proc.topics:
                topics[t] = topics.get(t, 0) + 1

            # generate reports
            clin = gen.clinician_text()
            pat = gen.patient_text(detail='low')
            reports.append((f"case_{i+1}_clinician.txt", clin))
            reports.append((f"case_{i+1}_patient.txt", pat))

            # optional PDF
            if pdf_available:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 8, clin + "\n\n" + pat)
                pdf_bytes = pdf.output(dest='S').encode('latin1')
                pdfs.append((f"case_{i+1}.pdf", pdf_bytes))

            # display
            st.subheader(f"Case {i+1}")
            st.markdown(f"""**Clinician Report:**
```text
{clin}
```""")
            st.markdown(f"""**Patient Summary:**
```text
{pat}
```""")

        # store in session
        st.session_state.update({
            'stats': {'sentiments': sentiments, 'scores': scores, 'risks': risks, 'topics': topics},
            'reports': reports, 'pdfs': pdfs
        })

elif menu == "Insights" and 'stats' in st.session_state:
    st.header("Insights")
    stats = st.session_state['stats']
    # sentiment box
    df_sent = pd.DataFrame(stats['sentiments'])
    st.plotly_chart(px.box(df_sent))
    # readability
    st.plotly_chart(px.histogram(stats['scores'], nbins=10))
    # risk pie
    st.plotly_chart(px.pie(names=['Low','Medium','High'], values=[stats['risks'].count('Low'), stats['risks'].count('Medium'), stats['risks'].count('High')]))
    # topics
    df_top = pd.DataFrame.from_dict(stats['topics'], orient='index', columns=['count']).reset_index()
    df_top.columns = ['topic','count']
    st.plotly_chart(px.bar(df_top, x='topic', y='count'))

elif menu == "Export" and 'reports' in st.session_state:
    st.header("Export")
    # text reports zip
    buf = BytesIO()
    with zipfile.ZipFile(buf,'w') as z:
        for name,content in st.session_state['reports']:
            z.writestr(name, content)
    buf.seek(0)
    st.download_button("Download Text Reports", buf, file_name="reports.zip")
    # pdfs
    if pdf_available and st.session_state['pdfs']:
        buf2=BytesIO()
        with zipfile.ZipFile(buf2,'w') as z2:
            for name,content in st.session_state['pdfs']:
                z2.writestr(name, content)
        buf2.seek(0)
        st.download_button("Download PDFs", buf2, file_name="reports_pdf.zip")
else:
    st.info("Please upload data in 'Upload & Process' first.")


