# MLflow Tracing & LangChain RAG Project

This project demonstrates how to use MLflow for tracing, logging, and serving LangChain and RAG (Retrieval-Augmented Generation) models, with integration for OpenAI and Pinecone.

## Project Structure

- `mlflow-langchain.py`: Log and load LangChain models with MLflow.
- `mlflow-rag-retriver.py`: RAG pipeline with Pinecone vector store and MLflow tracking.
- `mlflow-trace-tags.py`: Example of tracing OpenAI chat completions with MLflow tags.
- `requirements.txt`: Python dependencies.
- `mlartifacts/`, `mlruns/`: MLflow artifacts and run tracking directories.

## Prerequisites

- Python 3.10+
- [pip](https://pip.pypa.io/en/stable/)
- OpenAI API key
- Pinecone API key (for RAG retriever)

## Setup

1. **Clone the repository**
   ```sh
   git clone <your-repo-url>
   cd mlflow-tracing
   ```

2. **Create and activate a virtual environment**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   Create a `.env` file in the project root with the following content:
   ```
   OPENAI_API_KEY=your-openai-api-key
   PINECONE_API_KEY=your-pinecone-api-key
   MLFLOW_TRACKING_URI=http://localhost:5000
   MLFLOW_EXPERIMENT_NAME=your-experiment-name
   ```

5. **Start MLflow Tracking Server (if not already running)**
   ```sh
   mlflow ui
   ```
   By default, it runs at [http://localhost:5000](http://localhost:5000).

## Usage

### 1. Log and Load a LangChain Model

- To log a LangChain model to MLflow:
  ```sh
  python mlflow-langchain.py
  ```
  This will log a model and print its URI.

- To load and invoke a logged model:
  Edit the `load_langchain_model_and_invoke` call in `mlflow-langchain.py` with your model URI and run:
  ```sh
  python mlflow-langchain.py
  ```

### 2. Run RAG Retriever Example

- To run the RAG retriever pipeline:
  ```sh
  python mlflow-rag-retriver.py
  ```

### 3. Run OpenAI Tracing Example

- To trace OpenAI chat completions with MLflow:
  ```sh
  python mlflow-trace-tags.py
  ```

## Notes

- Ensure your API keys are valid and have sufficient quota.
- MLflow UI will show experiment runs, traces, and logged models.
- Artifacts and run metadata are stored in `mlartifacts/` and `mlruns/`.

## License

MIT License (add your license here)

---

For more details, see the code in:
- [`mlflow-langchain.py`](mlflow-langchain.py)
- [`mlflow-rag-retriver.py`](mlflow-rag-retriver.py)
- [`mlflow-trace-tags.py`](mlflow-trace-tags.py)