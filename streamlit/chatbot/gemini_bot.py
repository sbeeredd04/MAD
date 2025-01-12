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
                    You are Gemini, an AI-driven chatbot designed to explain investment strategies in clear,
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


def chatbot_ui():
    """
    Renders the Streamlit UI for the Gemini chatbot with chunk-wise streaming.
    
    1. We skip the very first assistant message in the history (the system-like prompt).
    2. We display conversation as "Gemini: ..." vs. "You: ...".
    3. On new user input, we stream the assistant's reply chunk by chunk in real-time.
    """
    st.subheader("Gemini AI Chatbot")

    # Initialize the chat in session state if not already there
    if "gemini_chat" not in st.session_state:
        st.session_state["gemini_chat"] = init_gemini_chat()

    # Display existing conversation (except the hidden prompt)
    for i, entry in enumerate(st.session_state["gemini_chat"].history):
        role = entry.role.lower()
        text_content = extract_text(entry.parts)

        # Skip the first assistant message (the system-like instructions)
        if role == "assistant" and i == 0:
            continue

        # Display user vs. assistant
        if role == "assistant":
            st.markdown(f"<span style='color: red'>**Gemini:**</span> {text_content}", unsafe_allow_html=True)        
        else:
            st.markdown(f"<span style='color: cyan'>**You:**</span> {text_content}", unsafe_allow_html=True)        


    st.write("---")

    # Get user input
    user_query = st.text_input("Type your message here and click Send:", "")

    if st.button("Send") and user_query.strip():
        # Display the user's message right away
        st.markdown(f"**You:** {user_query}")

        # Prepare a placeholder to update chunk-by-chunk for Gemini's response
        response_placeholder = st.empty()

        # We accumulate chunks to build the final response
        full_response = ""

        # Stream chunks from generate_gemini_response(...)
        for chunk_text in generate_gemini_response(st.session_state["gemini_chat"], user_query):
            full_response += chunk_text
            # Update the placeholder with the progressively built text
            response_placeholder.markdown(f"**Gemini:** {full_response}")
