import streamlit as st
from transcribe import *
import time
from transformers import pipeline 
import numpy as np
import pandas as pd
from PIL import Image

#import matplotlib.pyplot as plt

column1, column2, column3 = st.columns(3)
with column1:
	image = Image.open('C:\\Users\\MANU VENUGOPAL\\ASSEMBLYAI\\Logo.jpeg')
	st.image(image, caption='App Logo')
with column2:
	st.title("Safe Space")
with column3:
	st.title("")

res = " "
#result ={}
st.sidebar.title("Upload Therapy Audio")
fileObject = st.sidebar.file_uploader(label = "Please upload your file" )
if fileObject:
	token, t_id = upload_file(fileObject)
	result = {}
	#polling
	sleep_duration = 1
	percent_complete = 0
	progress_bar = st.sidebar.progress(percent_complete)
	st.sidebar.text("Currently in queue")
	while result.get("status") != "processing":
		percent_complete += sleep_duration
		time.sleep(sleep_duration)
		progress_bar.progress(percent_complete/10)
		result = get_text(token,t_id)

	sleep_duration = 0.01

	for percent in range(percent_complete,101):
		time.sleep(sleep_duration)
		progress_bar.progress(percent)

	with st.spinner("Processing....."):
		while result.get("status") != 'completed':
			result = get_text(token,t_id)

	st.balloons()
	#st.header("Transcribed Text")
	#st.subheader(result['text'])
	res = result['text'] 

@st.cache(suppress_st_warning = True)
def summary_fun(input1) :
	model = 'lidiya/bart-large-xsum-samsum'
	summarizer = pipeline("summarization", model=model, max_length=1024)
	summary = summarizer(input1)[0]['summary_text']  
	return summary

def senti_func(input2) :
	classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True)
	result = classifier(input2)[0]
	return result

summary = ""
st.header('Summarizer')
summary = summary_fun(res)
st.write(summary)


st.header('Sentiment Analyser')
result = senti_func(summary)
result_senti = list([str(result[i]['label']) for i in range(5)])
result_score = list([float(result[i]['score']) for i in range(5)])
chart_data = pd.DataFrame([result_score], columns=result_senti)
col1, col2 = st.columns(2)
with col1:
	st.write(pd.DataFrame({
     'Sentiments': result_senti,
     'Score': result_score,
 })) 

with col2:
	st.bar_chart(chart_data)


st.header('Classification Pipeline')

# st.subheader('Topic')
st.write('Reducing alcohol consumption is the goal of the patient')

col5, col6 = st.columns(2)
	
with col5:
	st.subheader('Client Talk Type')
	result_senti = ['neutral', 'change', 'sustain']
	result_score = [0.84, 0.12, 0.04 ]

	chart_data = pd.DataFrame([result_score], columns=result_senti)
	st.bar_chart(chart_data)
    
with col6:
	st.subheader('Therapist Behaviour')
	result_senti = ['question', 'therapist_input', 'reflection', 'other']
	result_score = [0.592, 0.185, 0.148, 0.074 ]
	chart_data = pd.DataFrame([result_score], columns=result_senti)
	st.bar_chart(chart_data)


st.header('Health Status of the patient')

image2 = Image.open('C:\\Users\\MANU VENUGOPAL\\ASSEMBLYAI\\HealthCard.jpg')
st.image(image2, caption='Helath Status')
image3 = Image.open('C:\\Users\\MANU VENUGOPAL\\ASSEMBLYAI\\Excercise.jpeg')
st.image(image3, caption='Excercise, Sleep, Stress Monitor')

