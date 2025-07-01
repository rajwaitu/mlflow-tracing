import mlflow
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from dotenv import load_dotenv
import os

load_dotenv()
mlflow.langchain.autolog()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("langchain-hm-rag-experiment")

template_instruction = """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.
    Question: {question} 
    Context: {context} 
    Answer:"""


def get_pinecone_as_retriever():
    """
    Get a Pinecone vector store.
    """
    index_name = os.getenv("PINECONE_INDEX_NAME")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(index_name)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 1, "fetch_k": 2, "lambda_mult": 0.5},
     )
    return retriever

#used by mlfow framework
def load_retriever(persist_dir=None):
    # Load OpenAI embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

    # Initialize Pinecone
    index_name = os.getenv("PINECONE_INDEX_NAME")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(index_name)

    # Return retriever
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    return vector_store.as_retriever(search_type="mmr",search_kwargs={"k": 1, "fetch_k": 2, "lambda_mult": 0.5})

def log_rag_chain_model():
    print("Loading OpenAI model...")
    llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0.1, max_tokens=1000,api_key=os.getenv("OPENAI_API_KEY"))
    #prompt = ChatPromptTemplate.from_template(template_instruction)

    system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentence maximum and keep the answer concise. "
    "Context: {context}"
)
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

    print("Creating Doc Chain...")
    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt, output_parser=StrOutputParser())

    print("Creating RAG Chain...")
    retrieval_chain = create_retrieval_chain(get_pinecone_as_retriever(), combine_docs_chain)

    print("Logging LangChain RAG model to MLflow...")
    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(retrieval_chain, name="hotel_mgmt_rag_model",loader_fn=load_retriever)
        print(f"Model logged with run ID: {model_info.run_id}")
        print(f"Model logged with model uri: {model_info.model_uri}")
        return model_info
    
def load_rag_model_and_invoke(model_uri,question):
    """
    Load a LangChain model from MLflow.
    """
    print("Loading RAG model from MLflow...")
    loaded_chain = mlflow.langchain.load_model(model_uri)
    print("Model loaded successfully.")
    print("calling invoke on model.")
    response_text = loaded_chain.invoke({"input": question})
    #response_text = loaded_chain.invoke(question)
    print(response_text)
    
if __name__ == "__main__":
    #log_rag_chain_model()
    load_rag_model_and_invoke("models:/m-7778e2cfffe445498ac1809e4721354c", "What is the age requirement for booking a hotel")