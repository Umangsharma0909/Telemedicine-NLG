def safe_summarize(text: str, max_length: int, min_length: int) -> str:
    try:
        if len(text.split()) < min_length:
            return text
        result = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        if isinstance(result, list) and result:
            return result[0].get('summary_text', text)
        return text
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
        if not sections:
            sections = {"Transcript": self.text}
        return sections

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
        return sentiment_analyzer.polarity_scores(self.text)
