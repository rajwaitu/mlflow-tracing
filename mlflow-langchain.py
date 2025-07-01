import mlflow
import time
from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("langchain-experiment")

#mlflow.openai.autolog()
mlflow.langchain.autolog()

template_instruction = (
    "Imagine you are a fine dining sous chef. Your task is to meticulously prepare for a dish, focusing on the mise-en-place process."
    "Given a recipe, your responsibilities are: "
    "1. List the Ingredients: Carefully itemize all ingredients required for the dish, ensuring every element is accounted for. "
    "2. Preparation Techniques: Describe the techniques and operations needed for preparing each ingredient. This includes cutting, "
    "processing, or any other form of preparation. Focus on the art of mise-en-place, ensuring everything is perfectly set up before cooking begins."
    "3. Ingredient Staging: Provide detailed instructions on how to stage and arrange each ingredient. Explain where each item should be placed for "
    "efficient access during the cooking process. Consider the timing and sequence of use for each ingredient. "
    "4. Cooking Implements Preparation: Enumerate all the cooking tools and implements needed for each phase of the dish's preparation. "
    "Detail any specific preparation these tools might need before the actual cooking starts and describe what pots, pans, dishes, and "
    "other tools will be needed for the final preparation."
    "Remember, your guidance stops at the preparation stage. Do not delve into the actual cooking process of the dish. "
    "Your goal is to set the stage flawlessly for the chef to execute the cooking seamlessly."
    "The recipe you are given is for: {recipe} for {customer_count} people. "
)

def log_langchain_model():
    """
    Log a LangChain model to MLflow.
    """
    print("Loading OpenAI model...")
    llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0.1, max_tokens=1000,api_key=os.getenv("OPENAI_API_KEY"))
    prompt = ChatPromptTemplate.from_template(template_instruction)

    print("Creating LLMChain...")
    chain = prompt | llm | StrOutputParser()

    print("Logging LangChain model to MLflow...")
    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(chain, name="langchain_model")
        print(f"Model logged with run ID: {model_info.run_id}")
        print(f"Model logged with model uri: {model_info.model_uri}")
        return model_info

def load_langchain_model_and_invoke(model_uri,recipe, customer_count):
    """
    Load a LangChain model from MLflow.
    """
    print("Loading LangChain model from MLflow...")
    loaded_chain = mlflow.langchain.load_model(model_uri)
    print("Model loaded successfully.")
    print("calling invoke on model.")
    response_text = loaded_chain.invoke({"recipe": recipe, "customer_count": customer_count})
    print(response_text)


if __name__ == "__main__":
    #model_info = log_langchain_model()

    load_langchain_model_and_invoke("models:/m-bee45dbcd7f54c109a8ec8556e755446", "chicken curry", "4")

