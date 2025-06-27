import mlflow
from openai import OpenAI
import time
from dotenv import load_dotenv
import os

load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "datageeks-genai-experiment"))

mlflow.openai.autolog()

@mlflow.trace
def enhanced_chat_completion(user_message, conversation_history=None):
    start_time = time.time()

    # Add context to the trace
    mlflow.update_current_trace(
        tags={
            "application": "customer_support_chat",
            "user_type": "premium",
            "conversation_length": len(conversation_history or []),
        }
    )

    # Prepare messages with history
    messages = conversation_history or []
    messages.append({"role": "user", "content": user_message})

    api_token = os.getenv("OPENAI_API_KEY")

    client = OpenAI(api_key=api_token)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages, temperature=0.7, max_tokens=500
    )

    # Add performance metrics
    mlflow.update_current_trace(
        tags={
            "response_time_seconds": time.time() - start_time,
            "token_count": response.usage.total_tokens,
            "model_used": response.model,
        }
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    user_message = "explain the concept of backpropagation in neural networks"
    conversation_history = [
    ]

    response = enhanced_chat_completion(user_message, conversation_history)
    print("Response from OpenAI:", response)