from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from transformers import pipeline

app = FastAPI()

templates = Jinja2Templates(directory="templates")
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

print("Loading Embedding Model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading FAISS Index...")
index = faiss.read_index("faiss_index.index")

print("Loading Saved Documents...")
with open("sample_docs.pkl", "rb") as f:
    sample_docs = pickle.load(f)

print("Loading Qwen LLM...")
generator = pipeline("text-generation", model="Qwen/Qwen2-0.5B-Instruct")

def chatbot(question):
    query_embedding = model.encode([question]).astype("float32")
    distances, indices = index.search(query_embedding, 5)
    context = "\n".join([sample_docs[idx] for idx in indices[0]])
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer briefly using only the context.\n"
    response = generator(prompt, max_new_tokens=100)
    answer = response[0]["generated_text"]
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()
    return answer

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/ask")
async def ask(question: str = Form(...)):
    answer = chatbot(question)
    return {"answer": answer}
