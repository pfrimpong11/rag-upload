import os
import nltk
import logging
import numpy as np
from sklearn.cluster import KMeans
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
import tempfile

# Configure NLTK and environment
nltk.data.path.append("./nltk_data")
os.environ["HF_HOME"] = "./cache"
os.environ["XDG_CACHE_HOME"] = "./cache"
os.environ["TMPDIR"] = "./tmp"

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Semantic chunking function
def semantic_chunk_with_embeddings(documents, embeddings, max_chunk_size=1000, min_sentences=2, overlap_sentences=1):
    """Chunk documents into semantically related groups using embeddings and clustering."""
    all_chunks = []
    for doc in documents:
        sentences = nltk.sent_tokenize(doc.page_content)
        if len(sentences) < min_sentences:
            all_chunks.append(Document(page_content=" ".join(sentences), metadata=doc.metadata))
            continue

        sentence_embeddings = embeddings.embed_documents(sentences)
        sentence_embeddings = np.array(sentence_embeddings)

        num_clusters = max(1, min(len(sentences) // min_sentences, 10))
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(sentence_embeddings)
        labels = kmeans.labels_

        clusters = {}
        for sentence, label in zip(sentences, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(sentence)

        for cluster_id, cluster_sentences in clusters.items():
            current_chunk = ""
            chunk_sentences = []
            for i, sentence in enumerate(cluster_sentences):
                if len(current_chunk) + len(sentence) < max_chunk_size:
                    current_chunk += sentence + " "
                    chunk_sentences.append(sentence)
                else:
                    all_chunks.append(Document(page_content=current_chunk.strip(), metadata=doc.metadata))
                    overlap = " ".join(chunk_sentences[-overlap_sentences:]) + " "
                    current_chunk = overlap + sentence + " "
                    chunk_sentences = chunk_sentences[-overlap_sentences:] + [sentence]
            if current_chunk:
                all_chunks.append(Document(page_content=current_chunk.strip(), metadata=doc.metadata))
    return all_chunks

# Function to initialize RAG system
def initialize_rag_system(pdf_path):
    logger.info("Loading Gemini model...")
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
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
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    logger.info("Loading PDF...")
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
    except Exception as e:
        logger.error(f"PDF loading failed: {str(e)}")
        st.error("Failed to load the PDF. Please upload a valid PDF file.")
        return None, None, None

    pdf_docs = semantic_chunk_with_embeddings(pages, embeddings)

    all_docs = pdf_docs
    unique_docs = {doc.page_content: doc for doc in all_docs}.values()
    for i, doc in enumerate(unique_docs):
        doc.metadata["doc_id"] = i

    logger.info("Building vector store...")
    vectorstore = FAISS.from_documents(list(unique_docs), embedding=embeddings)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    bm25_retriever = BM25Retriever.from_documents(list(unique_docs))
    bm25_retriever.k = 4
    retriever = EnsembleRetriever(retrievers=[faiss_retriever, bm25_retriever], weights=[0.5, 0.5])

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