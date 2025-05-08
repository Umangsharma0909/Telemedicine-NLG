import os
import re
from typing import Dict, List

import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Constants
SUMMARIZER_MODEL = "facebook/bart-large-cnn"
TOPIC_COUNT = 5

# Initialize NLP components
tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
sentiment_analyzer = SentimentIntensityAnalyzer()


class DataPreprocessor:
    @staticmethod
    def preprocess(text: str) -> str:
        text = re.sub(r"\[?\d{1,2}:\d{2}(?::\d{2})?\]?", "", text)
        text = re.sub(r"^\w+:\s*", "", text, flags=re.MULTILINE)
        fillers = ['um', 'uh', 'ah', 'hmm', 'you know']
        filler_pattern = r"\b(?:" + '|'.join(fillers) + r")\b"
        text = re.sub(filler_pattern, '', text, flags=re.IGNORECASE)
        return re.sub(r"\s+", ' ', text).strip()


def safe_summarize(text: str, max_length: int, min_length: int) -> str:
    try:
        if len(text.split()) < min_length:
            return text
        result = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return result[0].get('summary_text', text) if result else text
    except Exception:
        return text


class AdvancedTranscriptProcessor:
    SECTION_HEADERS = [
        "Subjective", "Objective", "Past Medical History",
        "History of Present Illness", "Review of Systems",
        "Physical Exam", "Assessment", "Plan", "Assessment and Plan"
    ]

    def __init__(self, text: str):
        self.text = text.strip()
        self.sections = self._extract_sections()
        self.topics = self._extract_topics()
        self.sentiment = self._analyze_sentiment()

    def _extract_sections(self) -> Dict[str, str]:
        pattern = rf"({'|'.join(self.SECTION_HEADERS)}):?"
        parts = re.split(pattern, self.text, flags=re.IGNORECASE)
        sections: Dict[str, str] = {}
        for i in range(1, len(parts), 2):
            header = parts[i].strip().title()
            body = parts[i + 1].strip()
            sections[header] = body
        return sections or {"Transcript": self.text}

    def _extract_topics(self) -> List[str]:
        words = re.findall(r"\b[a-zA-Z]{6,}\b", self.text.lower())
        freq: Dict[str, int] = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        stopwords = {"patient", "doctor", "care", "today", "yesterday"}
        filtered = {w: c for w, c in freq.items() if w not in stopwords}
        return [w for w, _ in sorted(filtered.items(), key=lambda item: item[1], reverse=True)[:TOPIC_COUNT]]

    def _analyze_sentiment(self) -> Dict[str, float]:
        return sentiment_analyzer.polarity_scores(self.text)


class AdvancedReportGenerator:
    def __init__(self, proc: AdvancedTranscriptProcessor):
        self.sections = proc.sections
        self.topics = proc.topics
        self.sentiment = proc.sentiment

    def clinician_text(self) -> str:
        lines = ["Clinician Report", "================"]
        lines.append(f"Topics: {', '.join(self.topics)}")
        lines.append(f"Sentiment: {self.sentiment}\n")
        for hdr, body in self.sections.items():
            summary = safe_summarize(body, max_length=80, min_length=20)
            lines.append(f"{hdr}:\n{summary}\n")
        lines.append(f"Risk: {self._risk()}\n")
        return "\n".join(lines)

    def patient_text(self, detail: str = "low") -> str:
        full = " ".join(self.sections.values())
        lengths = {'low': (40, 10), 'medium': (80, 20), 'high': (150, 40)}
        max_len, min_len = lengths.get(detail, lengths['low'])
        summary = safe_summarize(full, max_length=max_len, min_length=min_len)

        replacements = {'hypertension': 'high blood pressure', 'dyspnea': 'shortness of breath'}
        for k, v in replacements.items():
            summary = re.sub(rf"\b{k}\b", v, summary, flags=re.IGNORECASE)

        score = textstat.flesch_reading_ease(summary)
        if score < 80:
            sentences = re.split(r'(?<=[.!?]) +', summary)
            for i in range(1, len(sentences) + 1):
                candidate = " ".join(sentences[:i]).strip()
                if not candidate.endswith('.'):
                    candidate += '.'
                if textstat.flesch_reading_ease(candidate) >= 80:
                    summary = candidate
                    score = textstat.flesch_reading_ease(candidate)
                    break
            else:
                summary = sentences[0].strip()
                if not summary.endswith('.'):
                    summary += '.'
                score = textstat.flesch_reading_ease(summary)

        return f"Patient Summary (Flesch {score:.1f})\n{summary}"

    def _risk(self) -> str:
        neg = self.sentiment.get('neg', 0)
        if neg > 0.3:
            return "High"
        if neg > 0.1:
            return "Medium"
        return "Low"
