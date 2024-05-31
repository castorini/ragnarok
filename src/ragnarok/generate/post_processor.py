import re
import time

import spacy
import stanza
from typing import List, Dict, Any
from ragnarok.data import CitedSentence

class StanzaTokenizer:
    def __init__(self, lang='en', processors='tokenize'):
        self.pipeline = stanza.Pipeline(lang=lang, processors=processors)

    def tokenize(self, text: str, sep: str = "\n", strip: List[str] = ["-"]) -> List[str]:
        """
        Tokenize the input text into sentences.
        Args:
            text: str: input text
            sep: str: separator to split sentences
            strip: List[str]: list of strings to strip from the text
        
        Returns:
            List[str]: list of sentences
        """
        new_sentences = []
        for strip_str in strip:
            text = text.replace(strip_str, "")
        for sentence in self.pipeline(text).sentences:
            if sep in sentence.text:
                new_sentences += [text.strip() for text in sentence.text.split(sep)]
            else:
                new_sentences.append(sentence.text.strip())
        return new_sentences

class SpacyTokenizer:
    def __init__(self, model='en_core_web_trf'):
        self.nlp = spacy.load(model)
        
    def tokenize(self, text: str, replace_newline: str = " ") -> List[str]:
        """
        Tokenize the input text into sentences.
        Args:
            text: str: input text
            replace_newline: str: replace newline character with this character
            
        Returns:
            List[str]: list of sentences
        """
        sentences = []
        text = replace_newline.join(text.replace("\n", replace_newline).split(replace_newline))
        for sent in self.nlp(text).sents:
            if re.search('[a-zA-Z]', sent.text):
                sentences.append(sent.text)
        return sentences

class CoherePostProcessor:
    def __init__(self, tokenizer="spacy") -> None:
        self.tokenizer = StanzaTokenizer(lang='en', processors='tokenize') if tokenizer == "stanza" else SpacyTokenizer(model='en_core_web_trf')
        

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
        rag_exec_response = {"text": response.text, "citations": citations}
        sentences = self.tokenizer.tokenize(text_output)
        answers = []
        for sentence in sentences:
            answer_citations = self._find_sentence_citations(text_output, sentence, citations)
            answers.append(CitedSentence(text=sentence, citations=answer_citations))
        
        return answers, rag_exec_response
