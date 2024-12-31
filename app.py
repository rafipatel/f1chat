import streamlit as st
from mlx_lm import load, generate
from huggingface_hub import login
import os
from langchain.memory import ConversationBufferMemory

# @st.cache_resource
# def init_model():
#     token = os.getenv("HF_TOKEN")
#     if token:
#         login(token=token)
#     return load("Rafii/f1llama")
    # return load("mlx-community/Mixtral-8x7B-Instruct-v0.1")

#changees
# Add background image
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://wallpapers.com/images/featured/f1-hfubqbf4vngbuqur.webp");
             background-attachment: fixed;
             background-size: cover;
             background-position: center;
         }}
         
         # Make text more readable with semi-transparent dark overlay
         .stApp::before {{
             content: "";
             position: fixed;
             top: 0;
             left: 0;
             width: 100%;
             height: 100%;
             background-color: rgba(0,0,0,0.7);
             z-index: -1;
         }}
         
         # Style text elements for better visibility
         .stMarkdown, .stTextInput, div[data-testid="stText"] {{
             color: white !important;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
# Add background
add_bg_from_url()

token = os.getenv("HF_TOKEN")
model, tokenizer = load("Rafii/f1llama")

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

def format_chat_history(messages):
    formatted = ""
    for msg in messages:
        if "input" in msg:
            formatted += f"Human: {msg['input']}\n"
        if "output" in msg:
            formatted += f"Assistant: {msg['output']}\n"
    return formatted

def generate_response(user_input, max_tokens=100):
    try:
        # Get chat history
        chat_history = st.session_state.memory.load_memory_variables({})
        history = chat_history.get("history", "")
        
        # Create contextual prompt
        context = format_chat_history(history)
        full_prompt = f"""Previous conversation:
{context}
Human: {user_input}
Assistant:"""

        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": full_prompt}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = full_prompt
        
        response = generate(
            model, 
            tokenizer, 
            prompt=prompt,
            verbose=True
        )
        return response
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "Sorry, I encountered an error."

st.title("F1 Chatbot 🏎️")

user_input = st.text_input("Ask me anything:", key="user_input")

# Add debug prints and modified display logic
if st.button("Send", key="send"):
    if user_input:
        with st.spinner("Thinking..."):
            response = generate_response(user_input)
            # Debug print
            st.write(f"Debug - Response: {response}")
            
            st.session_state.memory.save_context(
                {"input": user_input}, 
                {"output": response}
            )
            # Debug print
            st.write("Debug - Context saved")

# Modified display section
if "memory" in st.session_state:
    st.write("### Conversation")
    try:
        chat_history = st.session_state.memory.load_memory_variables({})
        st.write(f"Debug - Full history: {chat_history}")  # Debug print
        
        if "history" in chat_history:
            for msg in chat_history["history"]:
                st.write(f"Debug - Message: {msg}")  # Debug print
                if isinstance(msg, dict):
                    if "input" in msg:
                        st.info(f"You: {msg['input']}")
                    if "output" in msg:
                        st.success(f"Assistant: {msg['output']}")
    except Exception as e:
        st.error(f"Error displaying conversation: {str(e)}")