import os
import json
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

##st.write("app is running")
# -------------------------------------------------------------
# Load environment variables (.env file)
# -------------------------------------------------------------
load_dotenv()

# Read GROQ API key from environment variables
ENV_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()


# -------------------------------------------------------------
# Initialize default system prompt in session state
# This ensures the prompt persists across Streamlit reruns
# -------------------------------------------------------------
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "You are a helpful AI Assistant. Be clear, correct and concise."


# -------------------------------------------------------------
# Streamlit Page Configuration
# -------------------------------------------------------------
st.set_page_config(
    page_title="🤖 Groq Conversational AI)",
    page_icon="🧠",
    layout="centered"
)

# CSS styling for dark aesthetic UI
st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #0f172a;
}

/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #020617;
}

/* Titles */
h1, h2, h3 {
    color: #38bdf8;
    font-weight: 700;
}

/* Text */
p, label, span {
    color: #e2e8f0;
}

/* Download buttons */
.stDownloadButton button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    padding: 10px 18px;
    font-weight: 600;
    border: none;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.4);
}

/* Download hover */
.stDownloadButton button:hover {
    background-color: #1d4ed8;
    transform: scale(1.03);
}

/* Other buttons */
.stButton button {
    background-color: #22c55e;
    color: white;
    border-radius: 10px;
    padding: 8px 16px;
    font-weight: 600;
    border: none;
}

/* Button hover */
.stButton button:hover {
    background-color: #16a34a;
}

/* Divider styling */
hr {
    border: 1px solid #334155;
}

</style>
""", unsafe_allow_html=True)

# Page title and caption
st.title("🧠 Muhammad Salem | AI Assistant")
st.caption("🚀 Powered by Groq • LangChain • Streamlit | Memory-Enabled Conversational AI")


# -------------------------------------------------------------
# Sidebar Controls (Model settings and UI configuration)
# -------------------------------------------------------------
with st.sidebar:

    st.header("Controls")

    # Optional API key input (overrides .env key if provided)
    api_key_input = st.text_input(
        "Groq API key (optional)",
        type="password"
    )

    # Determine which API key to use
    GROQ_API_KEY = api_key_input.strip() if api_key_input.strip() else ENV_GROQ_API_KEY


    # ---------------------------------------------------------
    # Model Selection Dropdown
    # ---------------------------------------------------------
    model_name = st.selectbox(
        "Choose Model",
        [
            "openai/gpt-oss-20b",
            "llama-3.3-70b-versatile",
            "qwen/qwen3-32b",
            "mixtral-8x7b-32768",  # Added model for assignment requirement
            "gemma2-9b-it"         # Added model for assignment requirement
        ],
        index=1
    )


    # ---------------------------------------------------------
    # Tone Selection Dropdown
    # Controls how the chatbot responds
    # ---------------------------------------------------------
    tone = st.selectbox(
        "Response Tone",
        ["Friendly", "Strict", "Teacher"],
        index=0
    )


    # ---------------------------------------------------------
    # Model Creativity (Temperature)
    # Higher values → more creative responses
    # ---------------------------------------------------------
    temperature = st.slider(
        "Temperature (Creativity)",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.1
    )


    # ---------------------------------------------------------
    # Maximum Tokens (Length of AI response)
    # ---------------------------------------------------------
    max_tokens = st.slider(
        "Maximum Token (Reply length)",
        min_value=64,
        max_value=1024,
        value=256,
        step=64
    )


    # ---------------------------------------------------------
    # Editable System Prompt
    # Defines the main behavior/rules for the chatbot
    # ---------------------------------------------------------
    system_prompt = st.text_area(
        "System Prompt (Rules for the bot)",
        value=st.session_state.system_prompt,
        height=140
    )

    # Save updated prompt to session state
    st.session_state.system_prompt = system_prompt


    # ---------------------------------------------------------
    # Reset System Prompt Button
    # Restores the default assistant instruction
    # ---------------------------------------------------------
    if st.button("Reset System Prompt"):
        st.session_state.system_prompt = "You are a helpful AI Assistant. Be clear, correct and concise."
        st.rerun()


    # ---------------------------------------------------------
    # Typing Effect Toggle (simulates typing animation)
    # ---------------------------------------------------------
    typing_effect = st.checkbox("Enabling typing effect", value=True)

    st.divider()


    # ---------------------------------------------------------
    # Clear Chat Button
    # Deletes stored conversation history
    # ---------------------------------------------------------
    if st.button("Clear chat"):
        st.session_state.pop("history_store", None)
        st.session_state.pop("download_cache", None)
        st.rerun()


# -------------------------------------------------------------
# API Key Guard
# Prevents running the chatbot without a valid key
# -------------------------------------------------------------
if not GROQ_API_KEY:
    st.error("Groq API Key is missing")
    st.stop()


# -------------------------------------------------------------
# Chat History Storage (per session)
# -------------------------------------------------------------
if "history_store" not in st.session_state:
    st.session_state.history_store = {}

SESSION_ID = "default_session"


# Function to retrieve or create chat history
def get_history(session_id: str) -> InMemoryChatMessageHistory:

    if session_id not in st.session_state.history_store:
        st.session_state.history_store[session_id] = InMemoryChatMessageHistory()

    return st.session_state.history_store[session_id]


# -------------------------------------------------------------
# Initialize LLM (Groq Model)
# -------------------------------------------------------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model=model_name,
    temperature=temperature,
    max_tokens=max_tokens
)


# -------------------------------------------------------------
# Prompt Template
# Defines how messages are structured for the LLM
# -------------------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])


# -------------------------------------------------------------
# Chain: Prompt → LLM → Output Parser
# -------------------------------------------------------------
chain = prompt | llm | StrOutputParser()


# -------------------------------------------------------------
# Add Memory to the Chain
# Allows conversation history to be preserved
# -------------------------------------------------------------
chat_with_history = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history"
)


# -------------------------------------------------------------
# Display Previous Messages in Chat Interface
# -------------------------------------------------------------
history_obj = get_history(SESSION_ID)

for msg in history_obj.messages:

    role = getattr(msg, "type", "")

    if role == "human":
        st.chat_message("user").write(msg.content)

    else:
        st.chat_message("assistant").write(msg.content)


# -------------------------------------------------------------
# Chat Input Box
# -------------------------------------------------------------
user_input = st.chat_input("Type your message...")


# -------------------------------------------------------------
# Process User Message and Generate AI Response
# -------------------------------------------------------------
if user_input:

    # Display user message
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):

        placeholder = st.empty()

        try:

            # Invoke the LLM chain with memory
            response_text = chat_with_history.invoke(
                {
                    "input": user_input,
                    "system_prompt": st.session_state.system_prompt
                },
                config={"configurable": {"session_id": SESSION_ID}}
            )

        except Exception as e:

            st.error(f"Model Error: {e}")
            response_text = ""


        # -----------------------------------------------------
        # Typing animation effect
        # -----------------------------------------------------
        if typing_effect and response_text:

            typed = ""

            for ch in response_text:

                typed += ch
                placeholder.markdown(typed)
                time.sleep(0.005)

        else:

            placeholder.write(response_text)


# -------------------------------------------------------------
# Download Chat History Section
# -------------------------------------------------------------
st.divider()
st.subheader("⬇  Download Chat History")


# -------------------------------------------------------------
# Convert conversation to JSON format
# -------------------------------------------------------------
export_data = []

for m in get_history(SESSION_ID).messages:

    role = getattr(m, "type", "")

    if role == "human":
        export_data.append({"role": "user", "text": m.content})

    else:
        export_data.append({"role": "assistant", "text": m.content})


# JSON download button
st.download_button(
    label="⬇ Download chat_history.json",
    data=json.dumps(export_data, ensure_ascii=False, indent=2),
    file_name="chat_history_json",
    mime="application/json",
)


# -------------------------------------------------------------
# Convert conversation to plain text format
# -------------------------------------------------------------
chat_text = ""

for m in get_history(SESSION_ID).messages:

    role = getattr(m, "type", "")

    if role == "human":
        chat_text += f"User: {m.content}\n\n"

    else:
        chat_text += f"Assistant: {m.content}\n\n"


# Text download button
st.download_button(
    label="⬇ Download chat_history.txt",
    data=chat_text,
    file_name="chat_history.txt",
    mime="text/plain",
)