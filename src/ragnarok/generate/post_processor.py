import cohere
import time
import stanza
from typing import List, Dict, Any

class SentenceTokenizer:
    def __init__(self, lang='en', processors='tokenize'):
        self.pipeline = stanza.Pipeline(lang=lang, processors=processors)

    def tokenize(self, text: str, sep: str = "\n", strip: List[str] = ["-"]) -> List[str]:
        new_sentences = []
        for strip_str in strip:
            text = text.replace(strip_str, "")
        for sentence in self.pipeline(text).sentences:
            if sep in sentence.text:
                new_sentences += [text.strip() for text in sentence.text.split(sep)]
            else:
                new_sentences.append(sentence.text.strip())
        return new_sentences


class CoherePostProcessor:
    def __init__(self) -> None:
        self.tokenizer = SentenceTokenizer(lang='en', processors='tokenize')

    def _find_sentence_citations(self, text_output: str, sentence: str, cohere_citations: List[Dict[str, Any]]) -> List[int]:
        start_pos = text_output.find(sentence)
        end_pos = start_pos + len(sentence)
        citations = []
        for citation in cohere_citations:
            if citation["start"] >= start_pos and citation["end"] <= end_pos:
                citations += citation["document_ids"]
            elif citation["start"] < start_pos and citation["end"] <= end_pos and citation["end"] > start_pos:
                citations += citation["document_ids"]
            elif citation["start"] >= start_pos and citation["start"] < end_pos and citation["end"] > end_pos:
                citations += citation["document_ids"]
            elif citation["start"] < start_pos and citation["end"] > end_pos:
                citations += citation["document_ids"]

        if citations:
            citations = [int(doc_id.replace('doc_', '')) for doc_id in list(set(citations))]
            citations = sorted(citations)
        return citations

    def __call__(self, response) -> List[Dict[str, Any]]:
        text_output = response.text
        citations = [{
            "start": citation.start,
            "end": citation.end,
            "text": citation.text,
            "document_ids": list(citation.document_ids)
        } for citation in response.citations]

        sentences = self.tokenizer.tokenize(text_output)
        answers = []
        for sentence in sentences:
            answer_citations = self._find_sentence_citations(text_output, sentence, citations)
            answers.append({
                "text": sentence,
                "citations": answer_citations
            })
        return answers
