import stanza

class SentenceTokenizer:
    def __init__(self, lang='en', processors='tokenize'):
        self.pipeline = stanza.Pipeline(lang=lang, processors=processors)

    def tokenize(self, text: str, sep: str = "\n", strip: List[str] = ["-"]) -> List[str]:
        """
        Tokenize the input text into sentences.
        Args:
            text: str: input text

        Returns:
            List[str]: list of sentences
        """
        new_sentences = []

        # strip the text of any unwanted characters
        for strip_str in strip:
            text = text.replace(strip_str, "")

        # split the text into sentences
        for sentence in self.pipeline(text).sentences:
            if sep in sentence.text:
                new_sentences += [text.strip() for text in sentence.text.split(sep)]
            else:
                new_sentences.append(sentence.text.strip())

        return new_sentences

def find_sentence_citations(text_output: str, sentence: str, cohere_citations: list) -> list:
    """
    Convert a sentence to citations.
    Args:
        text_output: str: text output from Cohere API
        sentence: str: sentence to convert to citations
        cohere_citations: list: list of all citations from Cohere API
    Returns:
        list: list of citations
    """
    start_pos = text_output.find(sentence)
    end_pos = start_pos + len(sentence)

    citations = []

    for citation in cohere_citations:
        # Check if citation is within the sentence
        if citation["start"] >= start_pos and citation["end"] <= end_pos:
            citations += citation["document_ids"]

        # Edge case 1: if citation starts before the sentence and ends within the sentence
        elif citation["start"] < start_pos and citation["end"] <= end_pos and citation["end"] > start_pos:
            citations += citation["document_ids"]

        # Edge case 2: if citation starts within the sentence and ends after the sentence
        elif citation["start"] >= start_pos and citation["start"] < end_pos and citation["end"] > end_pos:
            citations += citation["document_ids"]

        # Edge case 3: if citation starts before the sentence and ends after the sentence
        elif citation["start"] < start_pos and citation["end"] > end_pos:
            citations += citation["document_ids"]

    if citations:
        # strip each 'doc_8' to be only 8
        citations = [int(doc_id.replace('doc_', '')) for doc_id in list(set(citations))]
        citations = sorted(citations)

    return citations

class CoherePostProcessor:
    def __init__(self):
        self.sentence_tokenizer = SentenceTokenizer(lang='en', processors='tokenize')
    def __call__(self, response):
        output_response, output_citations = response.text, response.citations
        cohere_citations = [{
                    "start": citation.start,
                    "end": citation.end,
                    "text": citation.text,
                    "document_ids": list(citation.document_ids)} for citation in output_citations]
        # define the answer list
        answers = []
        # get the sentences from the output
        sentences = self.sentence_tokenizer.tokenize(output_response)

        # get the citations for each sentence
        for sentence in sentences:
            citations = find_sentence_citations(output_response,
                                                sentence, cohere_citations)
            answers.append({
                "text": sentence,
                "citations": citations
            })
        return answers

