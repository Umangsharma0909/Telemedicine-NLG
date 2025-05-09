import os
import re
from typing import Dict, List

import pandas as pd
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Try to import HuggingFace transformers for summarization
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    SUMMARIZER_MODEL = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    summarizer_available = True
except Exception:
    summarizer_available = False

# Constants
TOPIC_COUNT = 5

class DataPreprocessor:
    @staticmethod
    def preprocess(text: str) -> str:
        # Remove timestamps like [00:00:00]
        cleaned = re.sub(r"\[?\d{1,2}:\d{2}(?::\d{2})?\]?", "", text)
        # Strip speaker labels
        cleaned = re.sub(r"^\w+:\s*", "", cleaned, flags=re.MULTILINE)
        # Remove filler words
        fillers = ['um', 'uh', 'ah', 'hmm', 'you know']
        pattern = r"\b(?:" + '|'.join(fillers) + r")\b"
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        # Normalize whitespace
        cleaned = re.sub(r"\s+", ' ', cleaned)
        return cleaned.strip()


def safe_summarize(text: str, max_length: int = 80, min_length: int = 20) -> str:
    if not summarizer_available:
        # Fallback: return first min_length words
        return ' '.join(text.split()[:min_length]) + '...'
    words = text.split()
    max_len = min(max_length, max(5, len(words) - 1))
    try:
        result = summarizer(text, max_length=max_len, min_length=min(min_length, max_len), do_sample=False)
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
            body = parts[i+1].strip()
            sections[header] = body
        return sections or {"Transcript": self.text}

    def _extract_topics(self) -> List[str]:
        words = re.findall(r"\b[a-zA-Z]{6,}\b", self.text.lower())
        freq: Dict[str, int] = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        stopwords = {"patient","doctor","care","today","yesterday"}
        filtered = {w: c for w, c in freq.items() if w not in stopwords}
        top = sorted(filtered.items(), key=lambda item: item[1], reverse=True)[:TOPIC_COUNT]
        return [w for w, _ in top]

    def _analyze_sentiment(self) -> Dict[str, float]:
        analyzer = SentimentIntensityAnalyzer()
        return analyzer.polarity_scores(self.text)

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

    def patient_text(self, detail: str = "high") -> str:
        # Provide a longer, more detailed patient summary by default
        full = " ".join(self.sections.values())
        lengths = {'low': (40, 10), 'medium': (80, 20), 'high': (150, 40)}
        max_len, min_len = lengths.get(detail, lengths['high'])
        summary = safe_summarize(full, max_length=max_len, min_length=min_len)
        # Replace complex medical terms
        replacements = {'hypertension': 'high blood pressure', 'dyspnea': 'shortness of breath'}
        for k, v in replacements.items():
            summary = re.sub(rf"\b{k}\b", v, summary, flags=re.IGNORECASE)
        # Ensure readability score >= 80 if possible
        score = textstat.flesch_reading_ease(summary)
        if score < 80:
            sentences = re.split(r'(?<=[.!?]) +', summary)
            for i in range(1, len(sentences)+1):
                candidate = " ".join(sentences[:i]).strip()
                if not candidate.endswith('.'):
                    candidate += '.'
                if textstat.flesch_reading_ease(candidate) >= 80:
                    summary = candidate
                    break
            else:
                summary = sentences[0].strip() + '.'
        return f"Patient Summary (Flesch {textstat.flesch_reading_ease(summary):.1f})\n{summary}"

    def _risk(self) -> str:
        neg = self.sentiment.get('neg', 0)
        if neg > 0.3:
            return "High"
        if neg > 0.1:
            return "Medium"
        return "Low"
