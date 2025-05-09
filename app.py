import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import base64
import zipfile
import textstat
from io import StringIO, BytesIO
from fpdf import FPDF
from modules.processor import (
    AdvancedTranscriptProcessor,
    AdvancedReportGenerator,
    DataPreprocessor
)

st.set_page_config(
    page_title="Smart Medical Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    body {
        background-color: #0f172a;
        color: white;
    }
    .reportview-container .main .block-container {
        padding: 2rem;
        background-color: #1e293b;
        border-radius: 10px;
    }
    .stDownloadButton>button, .stButton>button {
        background-color: #10b981;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .stMetric {
        background-color: #334155;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: white;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #38bdf8;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Smart Medical Dashboard")
st.markdown("Effortless AI-generated summaries and risk insights from medical transcripts.")

menu = st.sidebar.radio("Navigation", ["Upload & Process", "Insights", "Export"])

if menu == "Upload & Process":
    uploaded_file = st.file_uploader("📤 Upload transcript CSV with 'transcription' column", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:", df.head())

        if 'transcription' not in df.columns:
            st.error("❌ The uploaded CSV must contain a 'transcription' column.")
            st.stop()

        if df['transcription'].dropna().empty:
            st.warning("⚠️ The 'transcription' column is empty or contains only blank rows.")
            st.stop()
        st.success(f"Loaded {len(df)} records successfully.")

        search_id = st.text_input("🔍 Filter by Patient ID or Index (e.g., 1, 2, ...):")

        sentiments = {"pos": [], "neu": [], "neg": []}
        scores = []
        risks = {"Low": 0, "Medium": 0, "High": 0}
        all_topics = {}
        all_reports = []
        all_pdfs = []

        st.subheader("🧾 Patient Reports")

        for idx, row in df.iterrows():
            if search_id and str(idx+1) != search_id.strip():
                continue

            raw_text = row.get("transcription", "")
            if not raw_text.strip():
                continue

            clean_text = DataPreprocessor.preprocess(raw_text)
            proc = AdvancedTranscriptProcessor(clean_text)
            gen = AdvancedReportGenerator(proc)

            sent = proc.sentiment
            sentiments["pos"].append(sent.get("pos", 0))
            sentiments["neu"].append(sent.get("neu", 0))
            sentiments["neg"].append(sent.get("neg", 0))

            score = textstat.flesch_reading_ease(clean_text)
            scores.append(score)

            risk = gen._risk()
            risks[risk] += 1

            for topic in proc.topics:
                all_topics[topic] = all_topics.get(topic, 0) + 1

            clinician_txt = gen.clinician_text()
            patient_txt = gen.patient_text(detail="low")

            all_reports.append({"filename": f"case_{idx+1}/clinician_report.txt", "content": clinician_txt})
            all_reports.append({"filename": f"case_{idx+1}/patient_summary.txt", "content": patient_txt})

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, f"Clinician Report\n\n{clinician_txt}\n\nPatient Summary\n\n{patient_txt}")
            pdf_bytes = pdf.output(dest='S').encode('latin1')
            all_pdfs.append((f"case_{idx+1}.pdf", pdf_bytes))

            with st.expander(f"🩺 Case {idx+1} Summary"):
                st.metric("Flesch Score", f"{score:.1f}")
                st.metric("Sentiment", f"{sent.get('compound', 0):.2f}")
                st.metric("Risk Level", risk)
                st.code(clinician_txt, language="markdown")
                st.markdown(f"""```text\n{patient_txt}\n```""")

                st.download_button("⬇️ Download Clinician Report", clinician_txt, file_name=f"clinician_case_{idx+1}.txt")
                st.download_button("⬇️ Download Patient Summary", patient_txt, file_name=f"patient_case_{idx+1}.txt")

        st.session_state.df = df
        st.session_state.sentiments = sentiments
        st.session_state.scores = scores
        st.session_state.risks = risks
        st.session_state.all_topics = all_topics
        st.session_state.all_reports = all_reports
        st.session_state.all_pdfs = all_pdfs

elif menu == "Insights" and 'df' in st.session_state:
    st.header("📊 Aggregate Medical Insights")
    sentiments = st.session_state.sentiments
    scores = st.session_state.scores
    risks = st.session_state.risks
    all_topics = st.session_state.all_topics

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Sentiment Distribution")
        fig_sent = px.box(sentiments, points="all", title="Sentiment Scores by Category")
        st.plotly_chart(fig_sent, use_container_width=True)

    with col2:
        st.subheader("📚 Readability (Flesch Score)")
        fig_read = px.histogram(scores, nbins=20, labels={'value': 'Flesch Score'}, title="Flesch Reading Ease Distribution")
        st.plotly_chart(fig_read, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("🚨 Risk Levels")
        fig_risk = px.pie(names=list(risks.keys()), values=list(risks.values()), title="Overall Risk Distribution")
        st.plotly_chart(fig_risk, use_container_width=True)

    with col4:
        st.subheader("🧵 Frequent Topics")
        topic_df = pd.DataFrame.from_dict(all_topics, orient="index", columns=["Count"]).sort_values("Count", ascending=False)
        fig_topic = px.bar(topic_df, x=topic_df.index, y="Count", title="Most Frequent Topics")
        st.plotly_chart(fig_topic, use_container_width=True)

elif menu == "Export" and 'df' in st.session_state:
    st.header("📥 Export Reports & Summaries")

    # CSV Export
    sentiments = st.session_state.sentiments
    scores = st.session_state.scores
    risks = st.session_state.risks
    report_df = pd.DataFrame({
        "Flesch Score": scores,
        "Sentiment Positive": sentiments["pos"],
        "Sentiment Neutral": sentiments["neu"],
        "Sentiment Negative": sentiments["neg"],
        "Risk Level": [risk for risk, count in risks.items() for _ in range(count)]
    })

    csv_buffer = StringIO()
    report_df.to_csv(csv_buffer, index=False)
    st.download_button("⬇️ Download Metadata Summary (CSV)", data=csv_buffer.getvalue(), file_name="smart_medical_summary.csv", mime="text/csv")

    # ZIP Text Reports
    all_reports = st.session_state.all_reports
    if all_reports:
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zipf:
            for report in all_reports:
                zipf.writestr(report["filename"], report["content"])
        zip_buffer.seek(0)
        st.download_button("⬇️ Download All Reports (ZIP)", data=zip_buffer, file_name="all_medical_reports.zip", mime="application/zip")

    # PDF Bundle
    all_pdfs = st.session_state.all_pdfs
    if all_pdfs:
        pdf_zip_buffer = BytesIO()
        with zipfile.ZipFile(pdf_zip_buffer, "w") as pdf_zip:
            for name, content in all_pdfs:
                pdf_zip.writestr(name, content)
        pdf_zip_buffer.seek(0)
        st.download_button("⬇️ Download All Reports (PDF ZIP)", data=pdf_zip_buffer, file_name="all_case_reports_pdf.zip", mime="application/zip")
