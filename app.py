import os
import concurrent.futures
import streamlit as st
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Optional
import requests
from fpdf import FPDF
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("TOGETHER_API_KEY")
API_URL = os.getenv("TOGETHER_API_URL")
def create_pdf(content: str, filename: str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in content.split('\n'):
        pdf.cell(200, 10, txt=line, ln=True)
    pdf.output(filename)
class FileProcessor:
    def __init__(self, concurrency: int = 1, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.concurrency = max(1, concurrency)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def load_file(self, file_path: str) -> Optional[List[str]]:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
        try:
            if file_path.lower().endswith(".txt"):
                loader = TextLoader(file_path)
            elif file_path.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                print(f"Unsupported file type: {file_path}")
                return None
            documents = loader.load()
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    def process_files(self, file_list: List[str]) -> List[List[str]]:
        if not file_list:
            raise ValueError("No files to process.")
        file_count = len(file_list)
        parallel_threshold = 2
        if self.concurrency < parallel_threshold or file_count < parallel_threshold:
            return [self.load_file(file) for file in file_list if self.load_file(file)]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                results = list(executor.map(self.load_file, file_list))
            return [result for result in results if result]
st.title("File Upload and Processing")
def fetch_resume_analysis(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "prompt": prompt,
        "max_tokens": 200,
        "temperature": 0.9
    }
    response = requests.post(API_URL, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json().get("choices", [{}])[0].get("text", "").strip()
    else:
        print(f"Error: {response.text}")
        return None
uploaded_files = st.file_uploader("Choose PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
if uploaded_files:
    temp_dir = "temp_files"
    os.makedirs(temp_dir, exist_ok=True)
    file_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)
    processor = FileProcessor(concurrency=2, chunk_size=500, chunk_overlap=100)
    processed_data = processor.process_files(file_paths)
    resume_description = ""
    for idx, chunks in enumerate(processed_data):
        for chunk in chunks:
            resume_description += chunk.page_content
    prompt=f"""You are an ai helping assistant , Based on the resume description {resume_description} create a roadmap of the skills I need to enhance or acquire for professional growth.
    Additionally, suggest any relevant projects or practical applications that I can undertake to reinforce my learning and showcase my skills.
    only output the roadmap and projects in a list format.
    """
    response_text = fetch_resume_analysis(prompt)
    st.subheader("Resume Analysis")
    st.write(response_text)
    create_pdf(response_text,"roadmap.pdf")
    with open("roadmap.pdf", "rb") as pdf_file:
        st.download_button(
            label="Download Roadmap PDF",
            data=pdf_file,
            file_name="roadmap.pdf",
            mime='application/pdf'
        )
    os.remove("roadmap.pdf")
    for file_path in file_paths:
        os.remove(file_path)