import gradio as gr
import os
import uuid

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_aws import ChatBedrock
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Set AWS profile
os.environ["AWS_PROFILE"] = "bedrock"

# Claude 3 (Sonnet) with streaming enabled
model_id = "anthropic.claude-3-sonnet-20240229-v1:0" 
llm = ChatBedrock(
    model_id=model_id,
    streaming=True,
)

# Define a chat prompt with history support
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful chatbot in {country}."),
    MessagesPlaceholder(variable_name="chat_history"),  # For memory injection
    ("human", "{question}")
])

# Combine prompt and model
chain = prompt | llm

# Wrap with memory capability
chat_chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: ChatMessageHistory(),  # Use in-memory history for each session
    input_messages_key="question",
    history_messages_key="chat_history"
)

# Session manager for Gradio
sessions = {}

def prompt_bot(country, question, chat_history=[], session_id=None):
    """
    Streams Claude response with memory enabled.
    """
    if not session_id:
        session_id = str(uuid.uuid4())  # Create unique session
        sessions[session_id] = chat_history

    inputs = {
        "country": country,
        "question": question
    }

    # Stream Claude response chunk by chunk
    stream = chat_chain_with_history.stream(
        inputs,
        config={"configurable": {"session_id": session_id}}
    )

    # Accumulate streamed tokens
    full_response = ""
    for chunk in stream:
        full_response += chunk.content
        yield full_response, session_id  # Stream back to Gradio

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("### Claude Chatbot (via Amazon Bedrock)")
    country = gr.Dropdown(["USA", "Mexico", "Canada"], label="Country", value="USA")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Your message", placeholder="Ask me anything...")
    session_id = gr.State()  # Store session ID

    def respond(message, chat_history, country, session_id):
        full_response = ""
        for chunk, session in prompt_bot(country, message, chat_history, session_id):
            full_response = chunk  # update with latest chunk (streaming not handled directly here)
        chat_history.append((message, full_response))
        return chat_history, session

    msg.submit(
        respond,
        inputs=[msg, chatbot, country, session_id],
        outputs=[chatbot, session_id]
    )

if __name__ == "__main__":
    demo.launch()