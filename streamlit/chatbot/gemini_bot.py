# chatbot/gemini_bot.py

import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv


def init_gemini_chat():
    """
    Initializes and returns a Gemini chat object with a hidden system-like prompt 
    as the first assistant message. This function is only called once per session.
    """
    # Load environment variables
    load_dotenv()
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    model = genai.GenerativeModel("gemini-1.5-flash")

    # The "You are Gemini..." text is stored here as the first assistant message
    chat = model.start_chat(
        history=[
            {
                "role": "assistant",
                "parts": """
                    You are Mady, an AI-driven chatbot designed to explain investment strategies in clear,
                    accessible terms so that end users can make informed decisions and learn more about the
                    stock market. Your role is to:

                    1. Provide Plain-Language Explanations
                       - Break down complex financial concepts into easy-to-understand language.
                       - Offer step-by-step advice on strategy execution.

                    2. Guide Model Building
                       - Show users how to leverage the platform’s model builder, from data preparation
                         to training and tuning.
                       - Share best practices for interpreting model outputs to optimize investment decisions.

                    3. Present Market Insights
                       - Deliver relevant, real-time summaries of stock prices and trending sectors.
                       - Connect current market conditions to practical investment approaches.

                    4. Empower User Learning
                       - Encourage users to explore different strategies, understand the rationale
                         behind them, and take action.
                       - Answer questions about market dynamics, risk management, and long-term investing.

                    Your primary goal is to make complex analytics and investment strategies easy to 
                    understand and actionable, ensuring users feel confident and well-informed in their 
                    financial decisions. Leverage your expertise in AI and finance to guide and educate, 
                    while maintaining a friendly, approachable tone.
                """
            },
            {"role": "user", "parts": "Hi Gemini!"},
            {"role": "assistant", "parts": "Hello! How can I help you today?"},
        ]
    )
    return chat


def extract_text(parts):
    """
    Convert 'parts' (which can be a string, a list of strings, or a list of objects)
    into a single text string.
    """
    if isinstance(parts, str):
        return parts.strip()

    if isinstance(parts, list):
        results = []
        for p in parts:
            if hasattr(p, "text"):  # If it's an object with a .text attribute
                results.append(p.text)
            elif isinstance(p, str):
                results.append(p)
            else:
                results.append(str(p))
        return " ".join(results).strip()

    return str(parts).strip()


def generate_gemini_response(chat, user_input):
    """
    Sends a user message to the Gemini chat and yields text chunks in real-time.
    This is a generator function that streams tokens as they arrive.
    """
    response = chat.send_message(user_input, stream=True)
    # We iterate over the streaming response to yield chunks
    for chunk in response:
        yield chunk.text


# Main chatbot UI function
def chatbot_ui():
    
    # Add CSS for scrollable container
    st.markdown("""
        <style>
        .chat-container {
            height: calc(100vh - 200px); /* Adjust for header/padding */
            overflow-y: auto;
            padding: 20px;
            margin-bottom: 60px; /* Space for input field */
        }
        .input-container {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: calc(25% - 40px); /* Match chatbot column width */
            background: white;
            padding: 10px;
            z-index: 1000;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("💬 Mady")

    # Initialize the chat in session state
    if "gemini_chat" not in st.session_state:
        st.session_state["gemini_chat"] = init_gemini_chat()
        st.session_state["chat_history"] = []

    # Display the existing chat history
    for role, message in st.session_state["chat_history"]:
        if role == "Gemini":
            st.markdown(f"""
            <div style='background-color:rgba(0, 0, 0, 0.1); padding:10px; border-radius:10px; margin-bottom:5px;'>
                <b style='color:red;'>Gemini:</b> {message}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background-color:rgba(100, 100, 100, 0.1); padding:10px; border-radius:10px; margin-bottom:5px; text-align:right;'>
                <b style='color:cyan;'>You:</b> {message}
            </div>
            """, unsafe_allow_html=True)

    # Function to handle sending the message
    def send_message():
        user_input = st.session_state["user_input"].strip()
        if user_input:
            # Add user's message to chat history
            st.session_state["chat_history"].append(("You", user_input))

            # Get the full response from Gemini
            response = "".join(generate_gemini_response(st.session_state["gemini_chat"], user_input))

            # Add Gemini's response to chat history
            st.session_state["chat_history"].append(("Gemini", response))

            # Reset the input field
            st.session_state["user_input"] = ""

    # User input area with Enter key support
    st.text_input(
        "Type your message...",
        key="user_input",
        on_change=send_message
    )

    # Check if the Send button is clicked
    if st.button("Send"):
        send_message()


    st.markdown('</div>', unsafe_allow_html=True)
