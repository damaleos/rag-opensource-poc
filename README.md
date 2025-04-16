# rag-opensource-poc

This is a proof-of-concept project to explore the integration of Retrieval-Augmented Generation (RAG) systems using open-source Large Language Models (LLMs).

## 📌 Goals

- Evaluate the behavior of a basic RAG system.
- Identify integration challenges with external knowledge sources.
- Lay the groundwork for the final system development.

## 🚀 Tentative Technologies

- Python
- LangChain
- Chroma
- Ollama
- FastAPI (for the web prototype)
- Docker (to simplify environment setup) - PENDING

## 📁 Project Structure

```
rag-opensource-poc/
├── README.md
├── data/               # Base knowledge files
├── notebooks/          # Experiments and exploratory code
├── tests/
├── src/                # Source code
│   ├── llm/            # Código para interacción con LLMs
│   ├── retriever/      # Implementación del pipeline de recuperación
│   ├── routers/
│   └── main.py           
├── .env                      # API keys y variables de entorno (NO subir a Git)
├── .gitignore     
├── README.md
└── requirements.txt

```

## 🧠 Current Status

Currently in early design stage. This project will serve as a foundation for experimentation while the final system is being specified.