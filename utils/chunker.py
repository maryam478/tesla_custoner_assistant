from langchain.text_splitter import RecursiveCharacterTextSplitter
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize


class Chunker():
    def fixed_size(self, text, chunk_size=500, overlap=50):
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]

    def recursive(self, text, chunk_size=500, chunk_overlap=50):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return splitter.split_text(text)

    def nltk_sentence_chunks(self, text, sentences_per_chunk=5):
        sentences = sent_tokenize(text)
        return [' '.join(sentences[i:i+sentences_per_chunk]) for i in range(0, len(sentences), sentences_per_chunk)]

    def custom_delimiter(self, text, delimiter="\n\n"):
        return text.split(delimiter)