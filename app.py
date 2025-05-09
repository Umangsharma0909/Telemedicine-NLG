import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import base64
import zipfile
import textstat

from io import StringIO, BytesIO
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
    .reportview-container {
        background: #f8f9fa;
    }
    .sidebar .sidebar-content {
        background-color: #dbeafe;
    }
    .block-container {
        padding: 2rem 2rem 2rem 2rem;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 10px;
        padding: 0.4rem 1rem;
    }
    .stDownloadButton>button {
        background-color: #10b981;
        color: white;
        border-radius: 10px;
        padding: 0.4rem 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Smart Medical Dashboard with Auto-Generated Insights")
st.markdown("""
This dashboard processes telemedicine transcripts and presents real-time clinical summaries,
patient-friendly explanations, and analytical insights — all powered by AI.
""")

uploaded_file = st.file_uploader("📤 Upload a transcript CSV file with a 'transcription' column", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded {len(df)} records successfully.")

    sentiments = {"pos": [], "neu": [], "neg": []}
    scores = []
    risks = {"Low": 0, "Medium": 0, "High": 0}
    all_topics = {}
    all_reports = []

    st.markdown("---")
    st.subheader("🧾 Case-wise Report Summaries")

    for idx, row in df.iterrows():
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

        with st.expander(f"📌 Case {idx+1}", expanded=False):
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("**👨‍⚕️ Clinician Report**")
                st.code(clinician_txt, language="markdown")
                st.download_button("⬇️ Download Clinician Report", clinician_txt, file_name=f"clinician_case_{idx+1}.txt")

            with col2:
                st.markdown("**🩺 Patient Summary**")
                st.markdown(f"```
{patient_txt}
```")
                st.download_button("⬇️ Download Patient Summary", patient_txt, file_name=f"patient_case_{idx+1}.txt")

    # Visualizations
    st.markdown("---")
    st.header("📊 Aggregate Insights")

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

    # Export metadata summary CSV
    st.subheader("📤 Export Report Metadata")
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

    # Export all reports as ZIP
    st.subheader("📦 Export All Reports as ZIP")
    if all_reports:
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zipf:
            for report in all_reports:
                zipf.writestr(report["filename"], report["content"])
        zip_buffer.seek(0)

        st.download_button(
            label="⬇️ Download All Reports (ZIP)",
            data=zip_buffer,
            file_name="all_medical_reports.zip",
            mime="application/zip"
        )



