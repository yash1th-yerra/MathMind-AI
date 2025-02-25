import streamlit as st
import google.generativeai as genai
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import sympy as sp
import time
from PIL import Image
import cv2
import numpy as np

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(page_title="MathMind AI", layout="centered", page_icon="🧮")

st.markdown("""
    <style>
        .title {text-align: center; font-size: 2.5rem; font-weight: bold; color: #4a90e2; margin-bottom: 20px;}
        .stButton>button {background: linear-gradient(135deg, #4a90e2, #0082c8); color: white; font-size: 1.1rem; border-radius: 10px;}
        .stButton>button:hover {background: linear-gradient(135deg, #0082c8, #0056b3); transform: scale(1.05);}
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>📘 MathMind AI: Text & Image Processing</h1>", unsafe_allow_html=True)

# Initialize session state
if 'use_streaming' not in st.session_state:
    st.session_state.use_streaming = True
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'last_question' not in st.session_state:
    st.session_state.last_question = ""
if 'last_answer' not in st.session_state:
    st.session_state.last_answer = ""
if 'text_output' not in st.session_state:
    st.session_state.text_output = ""

# Sidebar settings
with st.sidebar:
    st.title("Settings")
    st.info("💡 Tip: Turn off streaming for faster solving of complex problems.")
    st.session_state.use_streaming = st.toggle(
        "Use Streaming Output", 
        value=st.session_state.use_streaming, 
        help="Toggle between streaming character-by-character output or normal output"
    )

# Load API keys
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
chain = LLMChain(
    llm=ChatGroq(model="Gemma2-9b-It", groq_api_key=os.getenv("GROQ_API_KEY")),
    prompt=PromptTemplate(
        input_variables=['question'], 
        template="You are an expert math tutor. Solve problems step-by-step with clear explanations.\n\nQuestion: {question}\n\nAnswer:"
    )
)

# Streaming Response Function
def reasoning_solver_stream(question):
    response_container = st.empty()
    response = chain.run(question)
    output_text = ""
    for char in response:
        output_text += char
        response_container.markdown(output_text + "▌")
        time.sleep(0.02)
    response_container.markdown(output_text)
    return output_text

# Solve based on user preference
def solve_with_preference(question):
    if st.session_state.use_streaming:
        return reasoning_solver_stream(question)
    else:
        with st.spinner("🔍 Solving..."):
            response = chain.run(question)
            st.markdown(response)
    return response

# Extract Text from Image
def extract_text_from_image(image_data):
    try:
        img = Image.open(image_data) if isinstance(image_data, (bytes, str)) else image_data
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        prompt = """Extract and format math expressions and question text from the image clearly."""
        response = model.generate_content([prompt, img])
        return response.text.strip()
    except Exception as e:
        return f"Error extracting text: {str(e)}"

# Tabs for input methods
tab1, tab2, tab3, tab4 = st.tabs(["📝 Text Input", "🖼 Upload Image", "📸 Capture Image", "💬 Chat"])

# Text Input Tab
with tab1:
    st.subheader("🔢 Type Your Math Problem")
    user_question = st.text_input("Enter math expression (e.g., x^2 - 4x + 4 = 0):")
    if st.button("Solve Text Problem"):
        if user_question.strip():
            st.session_state.last_question = user_question
            st.session_state.text_output = solve_with_preference(user_question)
    if st.session_state.text_output:
        st.markdown(st.session_state.text_output)

# Image Upload Tab
with tab2:
    st.subheader("🖼 Upload an Image")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Extract & Solve"):
            with st.spinner("Extracting text..."):
                extracted_text = extract_text_from_image(image)
                st.session_state.last_question = extracted_text
                solve_with_preference(extracted_text)

# Camera Capture Tab
with tab3:
    st.subheader("📸 Capture a Math Problem")
    use_camera = st.checkbox("Enable Camera", value=True)
    if use_camera:
        camera_image = st.camera_input("Take a picture")
        if camera_image is not None:
            image = Image.open(camera_image)
            if st.button("Extract & Solve"):
                with st.spinner("Extracting text..."):
                    extracted_text = extract_text_from_image(image)
                    st.session_state.last_question = extracted_text
                    solve_with_preference(extracted_text)
    else:
        st.write("Camera is disabled. Enable it to capture an image.")

# Chat Tab
with tab4:
    st.subheader("💬 Chat with MathMind AI")
    user_chat_input = st.text_input("Ask a question:", key="chat_input")
    if st.button("Send"):
        if user_chat_input.strip():
            st.session_state.chat_messages.append({"role": "user", "content": user_chat_input})
            context_question = f"Previous Problem: {st.session_state.last_question}\n\nUser Query: {user_chat_input}"
            response = solve_with_preference(context_question)
            st.session_state.chat_messages.append({"role": "ai", "content": response})
            st.markdown(response)
    for message in st.session_state.chat_messages:
        st.divider()
        st.markdown(f"**{message['role'].capitalize()}**: {message['content']}")
