import streamlit as st
import os

# from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
import google.generativeai as genai
import ast
import pydot

# Configure Streamlit and NLP Environment
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to use CPU

from dotenv import load_dotenv
load_dotenv()
# GEM_API_KEY = os.getenv('GEMINI_KEY')
# DOT_PATH = os.getenv('DOT_PATH')

GEM_API_KEY = st.secrets["GEMINI_KEY"]
DOT_PATH = st.secrets["DOT_PATH"]

genai.configure(api_key=GEM_API_KEY)

# Load the pre-trained model and tokenizer only once
# tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
# model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base")

# Set up NLTK data path
nltk_data_dir = os.path.expanduser("~/nltk_data")
nltk.download("punkt", download_dir=nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

def code_refactor(code):
    gen_model = genai.GenerativeModel("gemini-1.5-flash")
    template = "Refactor the following code and give only the refactored code as the answer: " + str(code)
    response = gen_model.generate_content(template)
    return response.text

# Preprocessing function
def preprocess_code(code):
    tokens = nltk.word_tokenize(code)
    return " ".join(tokens)

# Bug classification using the model
def analyze_code( code):
    

    gen_model = genai.GenerativeModel("gemini-1.5-flash")
    template = "Act as an expert software developer and tell in one word (\"syntax error is there\" or \"syntax error is not there\") if the code has any syntax errors and where it may occur: " + str(code)
    response = gen_model.generate_content(template)
    return response.text



def code_summary(code):
    gen_model = genai.GenerativeModel("gemini-1.5-flash")
    template = "Act as an expert software developer and give a one line summary of the following code: " + str(code)
    response = gen_model.generate_content(template)
    return response.text

def lang_detect(code):
    gen_model = genai.GenerativeModel("gemini-1.5-flash")
    template = "Detect the coding language of the following code, give answer in one word. If it is ambiguous or there is no language give \"Could not Detect\" as output: " + str(code)
    response = gen_model.generate_content(template)
    return response.text

# Streamlit UI
st.title("AI-Based Source Code Analysis")
st.write("Upload a source code file or paste your code below for analysis.")

code_input = st.text_area("Paste your code here:", height=300)

# Analyze button

try:
    if st.button("Analyze"):
        if code_input:
            # 1. Language Detection
            st.write("Language Detected:")
            lang = lang_detect(code_input)
            st.info(lang)

            # 2. Bug Classification
            st.write("Bug Classification:")
            result = analyze_code(code_input)
            st.info(result)

            # 3. Code Structure (AST Visualization)
            # st.write("Abstract Syntax Tree (AST) visualization:")
            # ast_image = visualize_ast(code_input)
            # if ast_image:
            #     st.image(ast_image)
            # else:
            #     st.warning("AST visualization is only available for Python code.")

            # 4. Code Summary
            st.write("Code Summary:")
            summary = code_summary(code_input)
            st.info(summary)

            # 5. Code Refactoring
            st.write("Refactored Code:")
            ref_code = code_refactor(code_input)
            st.info(ref_code)
        else:
            st.error("Please paste your code.")
except Exception as e:
    st.error(f"An API error occurred: {e} Please try after 10 minutes.")