from langchain_community.document_loaders import PyPDFLoader
from utils.chunker import Chunker


# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.docstore.document import Document

loader = PyPDFLoader("Owners_Manual.pdf")
pages = loader.load()
all_text = " ".join([page.page_content for page in pages])


chunker = Chunker()
chunks = chunker.recursive(all_text)  # Choose one: .fixed_size, .nltk_sentence_chunks, etc.

# embedding_model = HuggingFaceEmbeddings()

# docs = [Document(page_content=chunk) for chunk in chunks]
# vectorstore = FAISS.from_documents(docs, embedding_model)
