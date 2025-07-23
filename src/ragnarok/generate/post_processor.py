import os
import re
from typing import Any, Dict, List, Tuple

import spacy
import stanza

from ragnarok.data import CitedSentence


class StanzaTokenizer:
    def __init__(self, lang="en", processors="tokenize"):
        self.pipeline = stanza.Pipeline(lang=lang, processors=processors)

    def tokenize(
        self, text: str, sep: str = "\n", strip: List[str] = ["-"]
    ) -> List[str]:
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
    def __init__(self, model="en_core_web_trf"):
        try:
            self.nlp = spacy.load(model)
        except:
            os.system(f"python -m spacy download {model}")
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
        text = replace_newline.join(
            text.replace("\n", replace_newline).split(replace_newline)
        )
        for sent in self.nlp(text).sents:
            if re.search("[a-zA-Z]", sent.text):
                sentences.append(sent.text)
        return sentences


class CoherePostProcessor:
    def __init__(self, tokenizer="spacy") -> None:
        self.tokenizer = (
            StanzaTokenizer(lang="en", processors="tokenize")
            if tokenizer == "stanza"
            else SpacyTokenizer(model="en_core_web_trf")
        )

    def _find_sentence_citations(
        self, text_output: str, sentence: str, cohere_citations: List[Dict[str, Any]]
    ) -> List[int]:
        start_pos = text_output.find(sentence)
        end_pos = start_pos + len(sentence)
        citations = []
        for citation in cohere_citations:
            if citation["start"] >= start_pos and citation["end"] <= end_pos:
                citations += citation["document_ids"]
            elif (
                citation["start"] < start_pos
                and citation["end"] <= end_pos
                and citation["end"] > start_pos
            ):
                citations += citation["document_ids"]
            elif (
                citation["start"] >= start_pos
                and citation["start"] < end_pos
                and citation["end"] > end_pos
            ):
                citations += citation["document_ids"]
            elif citation["start"] < start_pos and citation["end"] > end_pos:
                citations += citation["document_ids"]

        if citations:
            citations = [
                int(doc_id.replace("doc_", "")) for doc_id in list(set(citations))
            ]
            citations = sorted(citations)
        return citations

    def __call__(self, response) -> Tuple[List[CitedSentence], Dict[str, Any]]:
        text_output = response.text
        if not response.citations:
            citations = []
        else:
            citations = [
                {
                    "start": citation.start,
                    "end": citation.end,
                    "text": citation.text,
                    "document_ids": list(citation.document_ids),
                }
                for citation in response.citations
            ]
        rag_exec_response = {"text": response.text, "citations": citations}
        sentences = self.tokenizer.tokenize(text_output)
        answers = []
        for sentence in sentences:
            answer_citations = self._find_sentence_citations(
                text_output, sentence, citations
            )
            answers.append(CitedSentence(text=sentence, citations=answer_citations))

        return answers, rag_exec_response


class GPTPostProcessor:
    def __init__(self, tokenizer="spacy") -> None:
        self.tokenizer = SpacyTokenizer(model="en_core_web_trf")

    def _find_sentence_citations(
        self, sentence: str, citation_range: List[int] = list(range(50))
    ) -> tuple[str, List[int]]:
        # Regex pattern to find citations
        pattern = re.compile(r"\[\d+\](?:,? ?)")

        # Find all citations
        citations = pattern.findall(sentence)
        if citations:
            # Remove citations from text
            sentence = pattern.sub("", sentence).strip()

            # Extract indices from citations
            indices = [
                int(re.search(r"\d+", citation).group()) - 1 for citation in citations
            ]
            citations = [index for index in indices if index in citation_range]
        else:
            matches = re.findall(r"\[[^\]]*\]", sentence)
            if not matches:
                return sentence, []
            citations = []
            for match in matches:
                citation = match[1:-1]
                try:
                    if "," in citation:
                        flag = False
                        for cit in citation.split(","):
                            cit = int(cit) - 1
                            if cit in citation_range:
                                flag = True
                                citations.append(int(cit))
                        if flag:
                            sentence = sentence.replace(match, "")
                    else:
                        citation = int(citation) - 1
                        if citation in citation_range:
                            citations.append(citation)
                            sentence = sentence.replace(match, "")
                except:
                    print(f"Not a valid citation: {match}")

        sentence = re.sub(" +", " ", sentence)
        if len(sentence) > 3:
            if sentence[-2] == " ":
                sentence = sentence[:-2] + sentence[-1]
        return sentence, citations

    def __call__(self, response) -> List[Dict[str, Any]]:
        text_output = response
        # Remove all \nNote: and \nReferences: from the text
        text_output = re.sub(r"\nNote:.*", "", text_output)
        text_output = re.sub(r"\nReferences:.*", "", text_output)
        sentences = self.tokenizer.tokenize(text_output)
        answers = []
        citation_range = list(range(50))
        for sentence in sentences:
            sentence_parsed, citations = self._find_sentence_citations(
                sentence, citation_range
            )
            answers.append(CitedSentence(text=sentence_parsed, citations=citations))

        rag_exec_response = {"text": response, "citations": citation_range}

        return answers, rag_exec_response
