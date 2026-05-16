from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile
from fastapi import File
from fastapi.responses import JSONResponse
import shutil
from rag import(
    read_pdf,
    chunks_text,
    create_embeddings,
    store_in_chromadb,
    search_query,
    generate_answer,
    collection
)



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)

def dd(data):
    return JSONResponse(
        content={"debug": str(data)},
        status_code=200
    )

### Home API

@app.get("/")
def home():
    return {
        "message":"LLM RAG Project Running"
    }

# PDF upload api
@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    dd(file)
    #save uploaded PDF
    pdf_path = f"../uploads/{file.filename}"
    with open(pdf_path,"wb") as buffer:
        shutil.copyfileobj(file.file,buffer)
        # STEP 1 READ PDF
    text = read_pdf(pdf_path)
    #STEP 2 CHUNK TEXT
    chunks = chunks_text(text)
    #STEP 3 CREATE EMBEDDINGS
    embeddings = create_embeddings(chunks)
    #STEP4 STORE IN CHROMADB
    store_in_chromadb(chunks,embeddings)
    return{
        "message":"PDF Uploaded Successfully",
        "total_chunks":len(chunks)
    }

#### ASK QUESTIONS API
@app.get("/ask/")
def ask_question(question:str):
    # SEARCH RELEVENT CHUNKS
    results = search_query(question)
    documents = results['documents'][0]
    #CREATE CONTEXT
    context = " ".join(documents)
    # GENERATE FINAL ANSWER
    answer = generate_answer(question,context)
    return {
        "question":question,
        "answer":answer
    }

# @app.get("/chromadb-data/")
# def get_chromadb_data():
#     data = collection.get()

#     return {
#         "total_records": len(data["ids"]),
#         "data": data
#     }

@app.get("/view-data/")
def view_data():
    data = collection.get(
        include=["documents"]
    )
    return {
        "total_chunks": len(data["documents"]),
        "documents": data["documents"]
    }

### by Sravan Kumar Jaini ###