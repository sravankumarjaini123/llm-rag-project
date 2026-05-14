from fastapi import FastAPI
import os

# pdf reader is the predefined class
# pdfreader used to read the data from pdf file
from pypdf import PdfReader

# sentence-transformer is the predefined class
# sentence-transformer,used to create embeddings
from sentence_transformers import SentenceTransformer

#chromadb,used to store the number vectors
import chromadb

# Openapi,used to generate the AI output
from openai import OpenAI


# LOAD EMBEDDING MODEL
# model, will do Tokenization,Token id,and generated embeddings (vectors
model = SentenceTransformer("all-MiniLM-L6-v2")

#initialize the chromadb
#store the embeddings(vectors)

client = chromadb.Client()
collection = client.create_collection("pdf_data")

#read the data from pdf file

def read_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        extract_text = page.extract_text()
        if extract_text:
            text += extract_text
    return text

# chunking
def chunks_text(text):
    chunk_size = 500
    chunks = []
    for i in range(0,len(text),chunk_size):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
    return chunks

# Embeddings
def create_embeddings(chunks):
    embeddings = model.encode(chunks)
    return embeddings

# store in chromadb
def store_in_chromadb(chunks,embeddings):
    collection.add(documents=chunks,embeddings=embeddings.tolist(),ids=[str(i) for i in range(len(chunks))])
    return "Data stored in chromadb successfully"

# search question in chromadb
def search_query(question):
    query_embedding = model.encode([question])
    results = collection.query(query_embeddings = query_embedding.tolist(),n_results=2)
    return results

# generate output
def generate_answer(question,context):
    api_key = os.getenv("OPEN_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable not set. "
            "Set it in your .env file or system environment."
        )
    openai_client = OpenAI(api_key=api_key)
    prompt = f"""
    Answer the question using below context only
    Context:{context}
    Question:{question}
    """
    response = openai_client.chat.completions.create(model="gpt-4.1-mini",messages=[{"role":"user","content":prompt}])
    final_answer = response.choices[0].message.content
    return final_answer




