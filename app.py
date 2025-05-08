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
    return pd.read_csv(csv_file)

@st.cache_data
 def process_case(text, detail):
    proc = AdvancedTranscriptProcessor(text)
    gen = AdvancedReportGenerator(proc)
    return gen.clinician_text(), gen.patient_text(detail)

if uploaded_file is not None:
    df = load_data(uploaded_file)
    total = len(df)
    st.sidebar.markdown(f"**Total cases:** {total}")

    if generate:
        clinician_reports = {}
        patient_reports = {}
        
        # Process each case
        for idx, row in df.iterrows():
            text = row.get('transcription', '') or ''
            if not text.strip():
                continue
            clin, pat = process_case(text, detail_level)
            clinician_reports[idx] = clin
            patient_reports[idx] = pat
        
        # Create ZIP in-memory
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for idx, txt in clinician_reports.items():
                zf.writestr(f"clinician_reports/case_{idx}_clinician_report.txt", txt)
            for idx, txt in patient_reports.items():
                zf.writestr(f"patient_summaries/case_{idx}_patient_summary.txt", txt)
        buf.seek(0)
        
        st.success("All reports generated!")
        st.download_button(
            label="Download All Reports as ZIP",
            data=buf,
            file_name="all_medical_reports.zip",
            mime="application/zip"
        )
        
        # Display by section
        tab_clin, tab_pat = st.tabs(["Clinician Reports", "Patient Summaries"])
        
        with tab_clin:
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
        
        with tab_pat:
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
        st.info("Configure your settings in the sidebar and click **Generate Reports**.")
else:
    st.info("Please upload a CSV file to get started.")
