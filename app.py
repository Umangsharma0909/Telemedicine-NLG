import streamlit as st
import pandas as pd
import io
import zipfile
from AdvancedTranscriptProcessor import AdvancedTranscriptProcessor
from AdvancedReportGenerator import AdvancedReportGenerator

# Configure page
st.set_page_config(
    page_title="Smart Medical Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar controls
st.sidebar.title("Dashboard Controls")
uploaded_file = st.sidebar.file_uploader("Upload transcription CSV", type=["csv"])
detail_level = st.sidebar.radio("Patient summary detail level", ("low", "high"), index=0)
generate = st.sidebar.button("Generate Reports")

@st.cache_data
def load_data(csv_file):
    # Read CSV from uploaded file-like object
    df = pd.read_csv(csv_file)
    return df

@st.cache_data
def process_case(text: str, detail: str):
    proc = AdvancedTranscriptProcessor(text)
    gen = AdvancedReportGenerator(proc)
    clinician_text = gen.clinician_text()
    patient_text = gen.patient_text(detail)
    return clinician_text, patient_text

if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        st.stop()

    total = len(df)
    st.sidebar.markdown(f"**Total cases:** {total}")

    if generate:
        clinician_reports = {}
        patient_reports = {}

        # Generate reports for each case
        for idx, row in df.iterrows():
            text = row.get('transcription', '') or ''
            if not text.strip():
                continue
            clin_txt, pat_txt = process_case(text, detail_level)
            clinician_reports[idx] = clin_txt
            patient_reports[idx] = pat_txt

        # Package reports into ZIP
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zipf:
            for idx, txt in clinician_reports.items():
                path = f"clinician_reports/case_{idx}_clinician_report.txt"
                zipf.writestr(path, txt)
            for idx, txt in patient_reports.items():
                path = f"patient_summaries/case_{idx}_patient_summary.txt"
                zipf.writestr(path, txt)
        buffer.seek(0)

        st.success("Reports generated successfully!")
        st.download_button(
            label="Download All Reports (ZIP)",
            data=buffer,
            file_name="all_medical_reports.zip",
            mime="application/zip"
        )

        # Display in tabs
        tabs = st.tabs(["Clinician Reports", "Patient Summaries"])
        with tabs[0]:
            st.header("Clinician Reports")
            for idx, txt in clinician_reports.items():
                with st.expander(f"Case {idx}"):
                    st.text(txt)
                    st.download_button(
                        label=f"Download Case {idx} Clinician Report",
                        data=txt,
                        file_name=f"case_{idx}_clinician_report.txt",
                        mime="text/plain"
                    )
        with tabs[1]:
            st.header("Patient Summaries")
            for idx, txt in patient_reports.items():
                with st.expander(f"Case {idx}"):
                    st.text(txt)
                    st.download_button(
                        label=f"Download Case {idx} Patient Summary",
                        data=txt,
                        file_name=f"case_{idx}_patient_summary.txt",
                        mime="text/plain"
                    )
    else:
        st.info("Configure options in the sidebar and click 'Generate Reports'.")
else:
    st.info("Please upload a CSV file to get started.")

