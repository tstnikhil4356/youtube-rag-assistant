# YouTube RAG Assistant

An AI-powered chatbot that lets you have a conversation with any YouTube video. Paste a video link, and ask questions about its content — the assistant retrieves the most relevant parts of the transcript and generates accurate, grounded answers using a Retrieval-Augmented Generation (RAG) pipeline.

**Live Demo:** [https://youtube-rag-assistant-2026.streamlit.app](https://youtube-rag-assistant-2026.streamlit.app)

---

## How It Works

![Flow Diagram](assets/1.png)

The application follows a standard RAG (Retrieval-Augmented Generation) architecture:

1. **Input** — User pastes a YouTube video URL into the app.
2. **Transcript Extraction** — The video ID is parsed from the URL, and the full transcript is fetched using `youtube-transcript-api`.
3. **Text Splitting** — The transcript is broken into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`, preserving context across chunk boundaries.
4. **Embedding Generation** — Each chunk is converted into a vector embedding using a HuggingFace sentence-transformer model (`all-MiniLM-L6-v2`).
5. **Vector Storage** — Embeddings are stored in a FAISS vector store (in-memory vector database) for fast similarity search.
6. **Query Input** — User types a question about the video.
7. **Semantic Retrieval** — The query is embedded and matched against stored chunks using FAISS's similarity search, returning the top-k most relevant chunks.
8. **Context Assembly** — Retrieved chunks are formatted and combined with the user's original question.
9. **Prompt Engineering** — Context and question are inserted into a carefully engineered prompt template that instructs the LLM to answer strictly from the given context.
10. **LLM Generation** — The prompt is sent to Groq's LLM (`llama-3.1-8b-instant`), which generates a natural language answer.
11. **Response Display** — The formatted answer is displayed on the Streamlit interface, along with an expandable view of the source transcript chunks used.

---

## Tech Stack

| Component            | Technology                                   |
|-----------------------|-----------------------------------------------|
| Frontend / UI          | Streamlit                                     |
| LLM Provider           | Groq (`llama-3.1-8b-instant`)                 |
| Orchestration          | LangChain                                     |
| Embeddings             | HuggingFace (`all-MiniLM-L6-v2`)              |
| Vector Store           | FAISS                                         |
| Transcript Extraction  | youtube-transcript-api                        |

---

## Features

- Paste any YouTube URL with available captions/transcripts
- Automatic transcript chunking and embedding
- Semantic search over video content (not just keyword matching)
- Context-grounded answers with source chunk transparency
- Clean, minimal Streamlit interface
- Session-based caching to avoid reprocessing the same video

---

## Local Setup

### Prerequisites

- Python 3.12 (recommended — some ML dependencies do not yet support newer versions)
- A [Groq API key](https://console.groq.com) (free tier available)
- Git

### 1. Clone the repository

```bash
git clone https://github.com/tstnikhil4356/youtube-rag-assistant.git
cd youtube-rag-assistant
```

### 2. Create and activate a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set up your environment variables

Create a `.env` file in the project root:

```bash
touch .env
```

Add your Groq API key inside it:

GROQ_API_KEY=your_actual_groq_api_key_here

### 5. Run the app

```bash
streamlit run Langchain_YoutubeChatBot.py
```

The app will open automatically in your browser at `http://localhost:8501`.

### 6. Try it out

- Paste a YouTube video URL (must have captions/transcript available)
- Wait for the transcript to be processed and embedded
- Ask a question about the video content in the text box
- View the generated answer, along with the retrieved transcript chunks used to answer it

---

## Project Structure

youtube-rag-assistant/
├── Langchain_YoutubeChatBot.py    # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .env                             # Environment variables (not committed)
├── .gitignore
├── assets/
│   └── 1.png                       # Architecture flow diagram
└── README.md

---

## Notes

- This app only works on YouTube videos that have captions/transcripts available (either auto-generated or manually added).
- The vector store is created fresh for each video and stored in-memory for the session — it does not persist across app restarts.
- Answers are generated strictly from the video's transcript context; the assistant is instructed not to use outside knowledge, reducing hallucination risk.

---

## License

This project is open source and available for personal and educational use.
