import os
import logging
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile

# Configure environment
os.environ["HF_HOME"] = "./cache"
os.environ["XDG_CACHE_HOME"] = "./cache"
os.environ["TMPDIR"] = "./tmp"

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Function to chunk documents
def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Chunk documents into fixed-size pieces using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = []
    for doc in documents:
        split_docs = text_splitter.split_documents([doc])
        chunks.extend(split_docs)
    return chunks

# Function to initialize RAG system
def initialize_rag_system(pdf_path):
    logger.info("Loading Gemini model...")
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3,
            top_p=0.9,
            max_tokens=1024
        )
    except Exception as e:
        logger.error(f"Gemini loading failed: {str(e)}")
        st.error("Failed to load the language model. Please check your API key.")
        return None, None, None

    logger.info("Loading embeddings...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
    except Exception as e:
        logger.error(f"Embeddings loading failed: {str(e)}")
        st.error("Failed to load embeddings. Please try again.")
        return None, None, None

    logger.info("Loading PDF...")
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
    except Exception as e:
        logger.error(f"PDF loading failed: {str(e)}")
        st.error("Failed to load the PDF. Please upload a valid PDF file.")
        return None, None, None

    logger.info("Chunking PDF content...")
    try:
        pdf_docs = chunk_documents(pages)
        if not pdf_docs:
            st.error("Failed to process the PDF content. Please try a different file.")
            return None, None, None
    except Exception as e:
        logger.error(f"Chunking failed: {str(e)}")
        st.error("Failed to process the PDF content. Please try again.")
        return None, None, None

    all_docs = pdf_docs
    unique_docs = {doc.page_content: doc for doc in all_docs}.values()
    for i, doc in enumerate(unique_docs):
        doc.metadata["doc_id"] = i

    logger.info("Building vector store...")
    try:
        vectorstore = FAISS.from_documents(list(unique_docs), embedding=embeddings)
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        bm25_retriever = BM25Retriever.from_documents(list(unique_docs))
        bm25_retriever.k = 4
        retriever = EnsembleRetriever(retrievers=[faiss_retriever, bm25_retriever], weights=[0.5, 0.5])
    except Exception as e:
        logger.error(f"Vector store or retriever initialization failed: {str(e)}")
        st.error("Failed to initialize the search system. Please try again.")
        return None, None, None

    template = """
    You are an AI assistant providing accurate and context-aware responses based on the uploaded document. 
    Use the information from the provided context to answer the question concisely and clearly. 
    
    If the context does not contain enough relevant information, respond with: "I'm not sure about that, but I'd be happy to help if you provide more details!"
    
    ### Context:
    {context}
    
    ### Question:
    {question}
    
    ### Answer:
    """
    
    prompt = PromptTemplate.from_template(template)
    parser = StrOutputParser()
    chain = LLMChain(llm=llm, prompt=prompt, output_parser=parser)

    return retriever, vectorstore, chain

# Streamlit interface
st.title("PDF Chatbot")
st.write("Upload a PDF and ask questions about its content.")

# Initialize session state
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# PDF upload
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # Initialize RAG system
    with st.spinner("Processing PDF..."):
        st.session_state.retriever, st.session_state.vectorstore, st.session_state.chain = initialize_rag_system(pdf_path)
    
    # Clean up temporary file
    os.unlink(pdf_path)
    
    if st.session_state.retriever is not None:
        st.success("PDF processed successfully! You can now ask questions.")
    else:
        st.error("Failed to process the PDF. Please try again.")

# Chat interface
if st.session_state.retriever is not None:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Ask a question about the PDF:"):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process question
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    context_docs = st.session_state.retriever.invoke(prompt)
                    max_similarity = max([st.session_state.vectorstore.similarity_search_with_score(prompt, k=1)[0][1] for _ in context_docs], default=0)
                    if max_similarity < 0.25:
                        answer = "I'm not sure about that, but I'd be happy to help if you provide more details!"
                    else:
                        context_text = "\n".join([doc.page_content for doc in context_docs])
                        response = st.session_state.chain.invoke({"context": context_text, "question": prompt})
                        answer = response['text'].split("Answer:")[-1].strip()
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    logger.error(f"Error processing question: {str(e)}")
                    st.error("An error occurred while processing your question. Please try again.")
else:
    st.info("Please upload a PDF to start chatting.")