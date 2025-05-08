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

st.set_page_config(layout="wide")
st.title("🧠 Smart Medical Dashboard with Auto-Generated Insights")

uploaded_file = st.file_uploader("📤 Upload a transcript CSV file with a 'transcription' column", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded {len(df)} records")

    # Initialize storage
    sentiments = {"pos": [], "neu": [], "neg": []}
    scores = []
    risks = {"Low": 0, "Medium": 0, "High": 0}
    all_topics = {}
    all_reports = []

    for idx, row in df.iterrows():
        raw_text = row.get("transcription", "")
        if not raw_text.strip():
            continue

        # Preprocessing + NLG
        clean_text = DataPreprocessor.preprocess(raw_text)
        proc = AdvancedTranscriptProcessor(clean_text)
        gen = AdvancedReportGenerator(proc)

        # Charts data
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

        # Generate reports
        clinician_txt = gen.clinician_text()
        patient_txt = gen.patient_text(detail="low")

        all_reports.append({"filename": f"case_{idx+1}/clinician_report.txt", "content": clinician_txt})
        all_reports.append({"filename": f"case_{idx+1}/patient_summary.txt", "content": patient_txt})

        # UI for each case
        with st.expander(f"📝 Case {idx+1}", expanded=False):
            st.subheader("📄 Clinician Report")
            st.code(clinician_txt, language="markdown")
            st.download_button(
                label="⬇️ Download Clinician Report",
                data=clinician_txt,
                file_name=f"clinician_case_{idx+1}.txt",
                mime="text/plain"
            )

            st.subheader("🩺 Patient Summary")
            st.markdown(f"```\n{patient_txt}\n```")
            st.download_button(
                label="⬇️ Download Patient Summary",
                data=patient_txt,
                file_name=f"patient_case_{idx+1}.txt",
                mime="text/plain"
            )

    # --- Summary Charts ---
    st.markdown("---")
    st.header("📊 Summary Insights")

    st.subheader("📈 Sentiment Distribution")
    fig_sent = px.box(sentiments, points="all", title="Sentiment Scores by Category")
    st.plotly_chart(fig_sent, use_container_width=True)

    st.subheader("📚 Readability (Flesch Score)")
    fig_read = px.histogram(scores, nbins=20, labels={'value': 'Flesch Score'}, title="Flesch Reading Ease Distribution")
    st.plotly_chart(fig_read, use_container_width=True)

    st.subheader("🚨 Risk Levels")
    fig_risk = px.pie(names=list(risks.keys()), values=list(risks.values()), title="Overall Risk Distribution")
    st.plotly_chart(fig_risk, use_container_width=True)

    st.subheader("🧵 Common Medical Topics")
    topic_df = pd.DataFrame.from_dict(all_topics, orient="index", columns=["Count"]).sort_values("Count", ascending=False)
    fig_topic = px.bar(topic_df, x=topic_df.index, y="Count", title="Most Frequent Topics")
    st.plotly_chart(fig_topic, use_container_width=True)

    # --- Metadata CSV Export ---
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
    csv_data = csv_buffer.getvalue()

    st.download_button(
        label="⬇️ Download Metadata Summary (CSV)",
        data=csv_data,
        file_name="smart_medical_summary.csv",
        mime="text/csv"
    )

    # --- All Reports ZIP Export ---
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


