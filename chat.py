import os
import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.messages.ai import AIMessage
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory


from dotenv import load_dotenv
load_dotenv()


st.title("Chatbot")

azure_version = "2024-06-01"
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_key = os.getenv("AZURE_OPENAI_KEY")

llm = AzureChatOpenAI(
    temperature=0.1,
    top_p=1.0,
    azure_deployment=azure_deployment,
    api_key=azure_key,
    azure_endpoint=azure_endpoint,
    api_version=azure_version,
)

# Set up memory
if 'memory' not in st.session_state:
    message = AIMessage("What can I help you with today?")
    st.session_state.memory = ConversationBufferWindowMemory(
        chat_memory=ChatMessageHistory(messages=[message]),
        memory_key='chat_history',
        k=20,
        return_messages=True,
        human_prefix="Human",
        ai_prefix="AI",
    )

chat_prompt = ChatPromptTemplate.from_template(
    """
Answer the question.
{input}
with the history
{chat_history}
""")

chain = chat_prompt | llm
chain_memory = RunnableWithMessageHistory(
    chain,
    get_session_history=lambda: st.session_state.memory.chat_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Render current messages from StreamlitChatMessageHistory
for msg in st.session_state.memory.buffer_as_messages:
    st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    # Note: new messages are saved to history automatically by Langchain during run
    response = chain_memory.invoke({
        "input": prompt,
        "chat_history": st.session_state.memory.buffer_as_messages,
    })
    st.chat_message("ai").write(response.content)
