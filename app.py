import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import zipfile
import textstat
from io import StringIO, BytesIO
from modules.processor import AdvancedTranscriptProcessor, AdvancedReportGenerator, DataPreprocessor

# For animations
from streamlit_lottie import st_lottie
import requests

# Set page config
st.set_page_config(
    page_title="Smart Medical Dashboard",
    page_icon="🩺",
    layout="wide"
)

# Sidebar: logo and Lottie animation
st.sidebar.title("🏥 Dashboard Settings")
logo = st.sidebar.file_uploader("Upload Dashboard Logo", type=["png", "jpg", "jpeg"])
if logo:
    st.sidebar.image(logo, use_column_width=True)

lottie_url = "https://assets2.lottiefiles.com/packages/lf20_mjlh3hcy.json"  # sample medical animation
def load_lottie(url):
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
    return None

lottie_animation = load_lottie(lottie_url)
if lottie_animation:
    st.sidebar.markdown("**Welcome!**")
    st_lottie(lottie_animation, height=150, key="medical_anim")

# Top-level header
st.markdown(
    "<h1 style='text-align:center; color:#38bdf8;'>🧠 Smart Medical Dashboard</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; color:white;'>Effortless AI summaries & dynamic insights</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# Navigation
menu = st.sidebar.radio("Navigate to", ["Overview", "Upload & Process", "Insights", "Export"])

# Overview page
if menu == "Overview":
    st.header("📊 Dashboard Overview")
    st.markdown("Use the side menu to upload data, view insights, and export reports.")
    # Example KPI cards
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Cases", "120", "+5%")
    k2.metric("Avg. Readability", "75.4", "-2.3")
    k3.metric("High Risk Cases", "18", "+10%")
    k4.metric("Avg. Sentiment", "0.12", "+0.01")

# Upload & Process page
elif menu == "Upload & Process":
    st.header("🚀 Upload & Process Transcripts")
    uploaded = st.file_uploader("Upload CSV with 'transcription' column", type=["csv"] )
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(df.head())
        if 'transcription' not in df.columns:
            st.error("Your CSV must have a 'transcription' column.")
            st.stop()
        if df['transcription'].dropna().empty:
            st.error("Transcription column is empty.")
            st.stop()
        st.success(f"Loaded {len(df)} records.")
        # Process data
        reports = []
        pdfs = []
        stats = {'sentiments':[], 'scores':[], 'risks':[], 'topics':{}}
        for idx, row in df.iterrows():
            text = str(row['transcription'])
            clean = DataPreprocessor.preprocess(text)
            proc = AdvancedTranscriptProcessor(clean)
            gen = AdvancedReportGenerator(proc)
            # collect stats
            stats['sentiments'].append(proc.sentiment)
            score = textstat.flesch_reading_ease(clean)
            stats['scores'].append(score)
            risk = gen._risk()
            stats['risks'].append(risk)
            for t in proc.topics:
                stats['topics'][t] = stats['topics'].get(t,0) + 1
            # generate outputs
            clin = gen.clinician_text()
            pat = gen.patient_text(detail='low')
            reports.append((f"case_{idx+1}_clinician.txt", clin))
            reports.append((f"case_{idx+1}_patient.txt", pat))
            # show individual
            with st.expander(f"Case {idx+1}"):
                st.metric("Readability", f"{score:.1f}")
                st.metric("Risk", risk)
                st.markdown(f"**Clinician Report**\n```text\n{clin}\n```")
                st.markdown(f"**Patient Summary**\n```text\n{pat}\n```")
        # store session
        st.session_state['stats'] = stats
        st.session_state['reports'] = reports

# Insights page
elif menu == "Insights":
    if 'stats' not in st.session_state:
        st.info("Upload data first.")
    else:
        st.header("📈 Analytics")
        stats = st.session_state['stats']
        df_sent = pd.DataFrame(stats['sentiments'])
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Sentiment Distribution")
            st.plotly_chart(px.box(df_sent, points='all'))
        with c2:
            st.subheader("Readability Scores")
            st.plotly_chart(px.histogram(stats['scores'], nbins=10))
        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Risk Breakdown")
            risk_counts = pd.Series(stats['risks']).value_counts()
            st.plotly_chart(px.pie(risk_counts, names=risk_counts.index, values=risk_counts.values))
        with c4:
            st.subheader("Top Topics")
            df_topics = pd.DataFrame.from_dict(stats['topics'], orient='index', columns=['count']).reset_index()
            df_topics.columns = ['topic','count']
            st.plotly_chart(px.bar(df_topics, x='topic', y='count'))

# Export page
elif menu == "Export":
    st.header("📤 Export Reports")
    if 'reports' not in st.session_state:
        st.info("No reports to export.")
    else:
        buf = BytesIO()
        with zipfile.ZipFile(buf,'w') as zf:
            for name,content in st.session_state['reports']:
                zf.writestr(name, content)
        buf.seek(0)
        st.download_button("Download Text Reports", buf, file_name="reports.zip")



