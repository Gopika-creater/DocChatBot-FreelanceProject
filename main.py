import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain  # type: ignore
from langchain.chains import create_retrieval_chain  # type: ignore
from langchain_core.prompts import ChatPromptTemplate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

retrieval_chain = None
llm = None
embeddings = None


def init_llm():
    global llm, embeddings

    logger.info("Loading LLM...")
    llm = llm = OllamaLLM(model="qwen2:0.5b", temperature=0.1)

    logger.info("Loading embeddings...")
    embeddings = HuggingFaceEmbeddings(
     model_name="./model_cache/minilm",  # ✅ local path, no download needed
    model_kwargs={"device": DEVICE},
    )
    logger.info("LLM + Embeddings ready")


def process_document(pdf_path):
    global retrieval_chain

    logger.info(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    splits = splitter.split_documents(docs)
    logger.info(f"Created {len(splits)} chunks")

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"  # ✅ saves locally
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}  # retrieve top 4 chunks
    )

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question based only on the context provided.
If the answer is not in the context, say "I don't know based on the document."

Context:
{context}

Question:
{input}

Answer:""")

    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    logger.info("Retrieval chain ready!")


def ask_question(query):
    global retrieval_chain

    if retrieval_chain is None:
        raise ValueError("Load a document first!")

    response = retrieval_chain.invoke({"input": query})
    return response["answer"]


if __name__ == "__main__":
    init_llm()

    # ✅ Change this to your PDF path
    process_document("C:/Users/Admin/Downloads/realistic_insurance_leads.pdf")

    # Ask questions in a loop
    print("\n📄 Document loaded! Ask your questions (type 'quit' to exit)\n")
    while True:
        question = input("You: ")
        if question.lower() == "quit":
            break
        answer = ask_question(question)
        print(f"\nBot: {answer}\n")