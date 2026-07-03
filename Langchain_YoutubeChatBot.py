import os
import re
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

st.set_page_config(page_title="YouTube RAG Assistant", layout="centered")
st.title("YouTube AI Assistant")
st.caption("Paste a YouTube link, then ask questions about its content.")

# models are cached so they are not reloaded on every rerun
@st.cache_resource
def load_llm():
    return ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = load_llm()
embeddings = load_embeddings()


def get_video_id(url: str):
    # pulls the 11 char video id out of common youtube url formats
    pattern = r'(?:v=|\/|youtu\.be\/)([0-9A-Za-z_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None


def fetch_transcript(video_id: str):
    # newer versions (1.x) return FetchedTranscriptSnippet objects with .text attribute
    # older versions (0.x) return plain dicts with ['text'] key
    # this handles both so it works regardless of installed version
    try:
        ytt_api = YouTubeTranscriptApi()
        fetched = ytt_api.fetch(video_id)
        return " ".join(snippet.text for snippet in fetched)
    except AttributeError:
        # fallback for older api versions
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join(item["text"] for item in transcript_list)


def process_youtube(url: str):
    video_id = get_video_id(url)
    if not video_id:
        st.error("Could not extract a valid video ID from that URL.")
        return None

    try:
        full_text = fetch_transcript(video_id)
    except Exception as e:
        st.error(f"Could not fetch transcript. Video may not have captions. Details: {e}")
        return None

    if not full_text.strip():
        st.error("Transcript is empty for this video.")
        return None

    # wrap raw text into a langchain Document so the splitter can work on it
    docs = [Document(page_content=full_text, metadata={"source": url, "video_id": video_id})]

    # chunk_size and overlap tuned for spoken transcript text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    splits = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(splits, embeddings)

    # k=4 controls how many chunks get pulled per query, raise for longer videos
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return retriever, len(splits)


def format_docs(docs):
    # joins retrieved chunks into one context block, numbered for traceability
    return "\n\n".join(f"[Chunk {i+1}]\n{d.page_content}" for i, d in enumerate(docs))


PROMPT_TEMPLATE = """You are an assistant that answers questions strictly using the
video transcript context provided below. Follow these rules:

1. Base your answer only on the given context, do not use outside knowledge.
2. If the context does not contain enough information, say so clearly instead of guessing.
3. Keep the answer concise and well structured, use bullet points if it helps clarity.
4. Do not mention "chunks" or the retrieval process in your answer, just answer naturally.

Context:
{context}

Question:
{question}

Answer:"""

prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])


# session state holds the retriever so the video is not reprocessed on every query
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "processed_url" not in st.session_state:
    st.session_state.processed_url = None

url = st.text_input("Paste YouTube URL:")

if url and url != st.session_state.processed_url:
    with st.spinner("Fetching transcript and building vector store..."):
        result = process_youtube(url)
        if result:
            retriever, num_chunks = result
            st.session_state.retriever = retriever
            st.session_state.processed_url = url
            st.success(f"Video processed into {num_chunks} chunks. Ready for questions.")

if st.session_state.retriever:
    query = st.text_input("Ask a question about the video:")

    if query:
        with st.spinner("Searching transcript and generating answer..."):
            retriever = st.session_state.retriever

            # retrieve top matching chunks first, so we can also display them
            retrieved_docs = retriever.invoke(query)
            context_text = format_docs(retrieved_docs)

            chain = (
                {"context": RunnableLambda(lambda x: context_text), "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            response = chain.invoke(query)

        st.markdown("### Answer")
        st.write(response)

        with st.expander("View retrieved transcript chunks"):
            for i, d in enumerate(retrieved_docs):
                st.markdown(f"**Chunk {i+1}**")
                st.write(d.page_content)
                st.divider()