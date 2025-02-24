import streamlit as st
import google.generativeai as genai
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import Tool
from dotenv import load_dotenv
import os
import sympy as sp
import io
from PIL import Image
import cv2
import numpy as np

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(page_title="MathMind AI", layout="centered",page_icon="🧮")

# Custom CSS for Styling
st.markdown("""
    <style>
        body {
            background-color: #f4f4f4;
            font-family: 'Poppins', sans-serif;
        }
        .title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: bold;
            color: #4a90e2;
            margin-bottom: 20px;
        }
        .stTabs [data-baseweb="tab-list"] {
            justify-content: center;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 1.2rem;
            padding: 10px;
        }
        .stButton>button {
            background: linear-gradient(135deg, #4a90e2, #0082c8);
            color: white;
            padding: 12px;
            font-size: 1.1rem;
            border-radius: 10px;
            transition: 0.3s;
            border: none;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #0082c8, #0056b3);
            transform: scale(1.05);
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>📘 MathMind AI: Text & Image Processing</h1>", unsafe_allow_html=True)

# Load API keys
groq_api_key = os.getenv("GROQ_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    st.error("Gemini API key not found. Please set GEMINI_API_KEY as an environment variable.")
    st.stop()

# Configure Gemini API
genai.configure(api_key=gemini_api_key)
llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# Wikipedia Tool
wiki_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wiki_wrapper.run,
    description="A tool for searching Wikipedia for information on various topics."
)

# Math Solver Function
def solve_math(expression):
    try:
        if "=" in expression:
            lhs, rhs = expression.split("=")
            x = sp.Symbol('x')
            equation = sp.Eq(sp.sympify(lhs), sp.sympify(rhs))
            solution = sp.solve(equation, x)
            return f"Solution: x = {solution}"
        else:
            return f"Result: {sp.sympify(expression).evalf()}"
    except Exception as e:
        return f"Error: {str(e)}"

calculator = Tool(
    name="Math Solver",
    func=solve_math,
    description="Solves algebraic equations or numerical expressions."
)


# Image Text Extraction Function
def extract_text_from_image(image_data):
    try:
        if isinstance(image_data, (np.ndarray, bytes)):
            if isinstance(image_data, np.ndarray):
                img = Image.fromarray(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))
            else:
                img = Image.open(io.BytesIO(image_data))
        elif isinstance(image_data, Image.Image):
            img = image_data
        else:
            img = Image.open(image_data)
        
        # Load Gemini Pro Vision model
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        
        prompt = """
        Analyze this image and:
        1. Extract any mathematical expressions, equations, or problems
        2. Format them in a clear, solvable form
        3. If there's text explaining the problem, include it as context
        
        Return the result in this format:
        Question: [extracted problem context if any]
        Math Expression: [extracted mathematical expression]
        """
        
        response = model.generate_content([prompt, img])
        text = response.text
        question_part, math_part = "", ""
        
        if "Question:" in text:
            parts = text.split("Math Expression:")
            question_part = parts[0].replace("Question:", "").strip()
            math_part = parts[1].strip() if len(parts) > 1 else ""
        else:
            math_part = text.strip()
            
        return question_part, math_part
    except Exception as e:
        return f"Error extracting text: {str(e)}", ""
    

# Reasoning Prompt Template
prompt = """ 
You are an expert math tutor. Solve problems step-by-step with clear explanations.

Question: {question}

Answer:
"""

prompt_template = PromptTemplate(input_variables=['question'], template=prompt)
chain = LLMChain(llm=llm, prompt=prompt_template)

# Reasoning Solver Function
def reasoning_solver(question):
    response = chain.run(question)
    return response

reasoning_tool = Tool(
    name="Reasoning Tool",
    func=reasoning_solver,
    description="Explains math problems step by step."
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! I can help solve math problems."}
    ]

if "camera_image" not in st.session_state:
    st.session_state["camera_image"] = None

# Tabbed Navigation for Inputs
tab1, tab2, tab3 = st.tabs(["📝 Text Input", "📷 Upload Image", "📸 Capture Image"])

# Text Input Tab
with tab1:
    st.subheader("🔢 Type Your Math Problem")
    user_question = st.text_input("Enter math expression (e.g., x^2 - 4x + 4 = 0):")
    
    if st.button("Solve Text Problem"):
        if user_question.strip():
            with st.spinner("🔍 Solving..."):
                st.session_state.messages.append({"role": "user", "content": user_question})
                response = reasoning_solver(user_question)
                st.session_state.messages.append({"role": "assistant", "content": response})

# Image Upload Tab
with tab2:
    st.subheader("🖼️ Upload an Image")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Extract & Solve",key="extract_solve_button_1"):
            with st.spinner("Extracting text..."):
                extracted_text = extract_text_from_image(image)
                st.session_state.messages.append({"role": "user", "content": f"Extracted: {extracted_text}"})
                response = reasoning_solver(extracted_text)
                st.session_state.messages.append({"role": "assistant", "content": response})

# Camera Capture Tab
# Camera Capture Tab
with tab3:
    st.subheader("📸 Capture a Math Problem")

    # Toggle button to enable/disable camera
    use_camera = st.checkbox("Enable Camera", value=True)

    if use_camera:
        camera_image = st.camera_input("Take a picture")
        
        if camera_image is not None:
            image = Image.open(camera_image)
            
            if st.button("Extract & Solve",key="extract_solve_button_2"):
                with st.spinner("Extracting text..."):
                    extracted_text = extract_text_from_image(image)
                    st.session_state.messages.append({"role": "user", "content": f"Extracted: {extracted_text}"})
                    response = reasoning_solver(extracted_text)
                    st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.write("Camera is disabled. Enable it to capture an image.")

# Display Results
st.divider()
if st.session_state.messages[-1]["role"] == "assistant":
    st.subheader("📊 Latest Solution")
    st.markdown(st.session_state.messages[-1]["content"])

st.subheader("📜 Chat History")
for msg in reversed(st.session_state.messages[:-1]):
    st.chat_message(msg["role"]).markdown(msg["content"])
