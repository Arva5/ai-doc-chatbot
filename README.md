# AI Doc Chatbot 📄

An AI-powered document chatbot that allows users to upload PDF documents and ask questions about them using natural language.

## Features

- 📄 **PDF Upload**: Upload multiple PDF documents
- 🤖 **AI-Powered Q&A**: Ask questions about your documents using natural language
- ⚡ **Fast Processing**: Powered by Groq's fast LLM inference
- 🎯 **Accurate Answers**: Uses RAG (Retrieval Augmented Generation) with FAISS vector search
- 🆓 **Free to Use**: Built with open-source technologies

## Tech Stack

- **Frontend**: Streamlit
- **LLM**: Groq API (Llama 3.1 8B)
- **Vector Database**: FAISS
- **Embeddings**: HuggingFace sentence-transformers
- **Framework**: LangChain

## How to Use

1. **Upload Documents**: Use the sidebar to upload PDF files
2. **Process Documents**: Click "Process Documents" to create the vector index
3. **Ask Questions**: Type your questions in the main chat area

## Deployment

This app is deployed on Streamlit Cloud and can be accessed at: [Your URL will appear here after deployment]

## Local Development

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Groq API key:
   ```
   GROQ_API_KEY=your_api_key_here
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Getting a Groq API Key

1. Go to [Groq Console](https://console.groq.com/)
2. Sign up for a free account
3. Create an API key
4. Add it to your `.env` file

## Project Structure

```
ai-doc-chatbot/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (not in repo)
├── .gitignore         # Git ignore file
└── README.md          # This file
```

## License

MIT License