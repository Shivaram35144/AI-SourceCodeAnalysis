import streamlit as st

from transformers import AutoTokenizer, AutoModelForSequenceClassification

import nltk

import ast
import astor

import pydot

import graphviz

from rope.base.project import Project
from rope.refactor.rename import Rename

import os

from dotenv import load_dotenv

load_dotenv()

DOT_PATH = os.getenv('DOT_PATH')

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base")

# Download NLTK's Punkt tokenizer
print(nltk.data.path)
nltk.data.path.append('/path/to/your/nltk_data')
nltk.download('punkt_tab')

# Language detection using guesslang


# Preprocessing function
def preprocess_code(code):
    tokens = nltk.word_tokenize(code)
    return " ".join(tokens)

# Bug classification using the model
def analyze_code(model, tokenizer, code):
    inputs = tokenizer(preprocess_code(code), return_tensors='pt', max_length=512, truncation=True)
    outputs = model(**inputs)
    return outputs.logits.argmax().item()

# AST visualization
def visualize_ast(code):
    tree = ast.parse(code)
    dot = pydot.Dot()
    dot.set('rankdir', 'LR')
    for node in ast.walk(tree):
        label = type(node).__name__
        dot.add_node(pydot.Node(id(node), label=label))
        for child in ast.iter_child_nodes(node):
            dot.add_edge(pydot.Edge(id(node), id(child)))
    output_path = os.path.join(os.getcwd(), 'ast.png')
    dot.write_png(output_path, prog=DOT_PATH)
    return output_path

# Refactoring suggestion (rename variable as an example)
# def suggest_refactoringg(code):
#     project = Project('.')
#     resource = project.get_file('test.py')  # Save code as test.py
#     rename = Rename(resource, 5)  # Example: Rename something at line 5
#     changes = rename.get_changes('new_name')
#     return changes




# Streamlit UI
st.title("AI-Based Source Code Analysis")

st.write("Upload a source code file or paste your code below for analysis.")


st.info("Note: This tool only works for python codes ! Please give code in python only")



#disclaimer
st.sidebar.title("Disclaimer")

st.sidebar.markdown("""
**Disclaimer:**

This AI-based source code analysis tool uses a pre-trained machine learning model for detecting general code quality and potential bugs. However, it has the following limitations:

- The model primarily performs **binary classification** of code (correct or potentially buggy).
- It may not detect **syntax errors** or provide detailed feedback on specific error types.
- This tool is not designed to detect **security vulnerabilities** or **language-specific runtime errors**.
- The analysis is based on patterns learned from training data and may not be accurate for all cases.

For detailed debugging and security auditing, consider using specialized code analysis tools.
""")

# Text area for user to input code
code_input = st.text_area("Paste your code here:", height=300)

# Analyze button
if st.button("Analyze"):
    if code_input:
        # 1. Language Detection
        
        
        
        
        # 2. Bug Classification
        result = analyze_code(model, tokenizer, code_input)
        if result == 1:
            st.success("The code looks good!")
        else:
            st.warning("The code may contain bugs.")

        # 3. Code Structure (AST Visualization)
        visualize_ast(code_input)
        st.write("Abstract Syntax Tree (AST) visualization:")
        st.image("ast.png")

        
    else:
        st.error("Please paste your code.")