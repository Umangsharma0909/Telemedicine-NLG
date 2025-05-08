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
        # Replace medical jargon
        replacements = {'hypertension': 'high blood pressure', 'dyspnea': 'shortness of breath'}
        for k, v in replacements.items():
            summary = re.sub(rf"\b{k}\b", v, summary, flags=re.IGNORECASE)
        # Ensure readability by iterative simplification
        score = textstat.flesch_reading_ease(summary)
        if score < 80:
            sentences = re.split(r'(?<=[.!?]) +', summary)
            for i in range(1, len(sentences)+1):
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
