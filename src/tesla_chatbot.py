# rag_chatbot.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()



API_KEY = os.getenv("OPENAI_API_KEY")

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# --------- Chunker Class ---------

class Chunker:
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


# --------- Main Pipeline ---------


def load_pdf_text(pdf_path):
    loader = PyMuPDFLoader("Owners_Manual.pdf")
    pages = loader.load()
    return " ".join([page.page_content for page in pages])

def create_vector_store(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docs = [Document(page_content=chunk) for chunk in chunks]
    return FAISS.from_documents(docs, embedding_model)

def build_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    return RetrievalQA.from_chain_type(
        llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    openai_api_key=API_KEY
    ),
        retriever=retriever
    )

if __name__ == "__main__":
    # === Step 1: Load Tesla manual PDF ===
    pdf_path = "Owners_Manual.pdf"  # <- Update this path if needed
    print("Loading PDF...")
    all_text = load_pdf_text(pdf_path)

    # === Step 2: Choose Chunking Strategy ===
    chunker = Chunker()
    chunks = chunker.recursive(all_text)  # or: .fixed_size(), .nltk_sentence_chunks(), .custom_delimiter()
    print(f"Chunked into {len(chunks)} pieces")

    # === Step 3: Create Knowledge Base ===
    print("Creating vector store...")
    vectorstore = create_vector_store(chunks)

    # === Step 4: Chat Loop ===
    print("Setting up chatbot...")
    qa_chain = build_qa_chain(vectorstore)

    print("\nTesla Manual Chatbot is ready. Type your question (type 'exit' to quit):")
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = qa_chain.invoke(query)
        print(f"Tesla Assistant: {answer['result']}")


        # Code to add UI

def get_qa_chain(pdf_path="Owners_Manual.pdf"):
    all_text = load_pdf_text(pdf_path)
    chunker = Chunker()
    chunks = chunker.recursive(all_text)
    vectorstore = create_vector_store(chunks)
    return build_qa_chain(vectorstore)


