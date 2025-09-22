# Updated FastAPI app with advanced RAG: Broad Recall -> Rerank -> Fusion -> Summarize -> Generation
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import re
import pandas as pd
import numpy as np
# from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from PyPDF2 import PdfReader
from huggingface_hub import login
# from transformers import BitsAndBytesConfig
import psutil
import threading
import time
import logging
from llama_cpp import Llama
logging.basicConfig(level=logging.INFO)

app = FastAPI()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    query: str

def resource_monitor(ram_threshold=90, cpu_threshold=90, gpu_threshold=90):
    """Monitor system resources and raise an error if thresholds are exceeded."""
    while True:
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        gpu_msg = ""
        gpu_exceeded = False
        if torch.cuda.is_available():
            try:
                total = torch.cuda.get_device_properties(0).total_memory
                reserved = torch.cuda.memory_reserved(0)
                allocated = torch.cuda.memory_allocated(0)
                free = total - reserved
                used_percent = (reserved / total) * 100
                gpu_msg = (
                    f"GPU: {used_percent:.2f}% used "
                    f"({reserved / (1024 ** 3):.2f} GB reserved / {total / (1024 ** 3):.2f} GB total)"
                )
                if used_percent >= gpu_threshold:
                    gpu_exceeded = True
            except Exception as e:
                gpu_msg = f"GPU: Unable to get usage ({e})"
        if mem.percent >= ram_threshold or cpu >= cpu_threshold or gpu_exceeded:
            raise RuntimeError(
                f"ERROR: Resource usage exceeded threshold!\n"
                f"RAM: {mem.percent:.2f}% used ({mem.used / (1024 ** 3):.2f} GB / {mem.total / (1024 ** 3):.2f} GB)\n"
                f"CPU: {cpu:.2f}% used\n"
                f"{gpu_msg}"
            )
        time.sleep(2) 

# Start resource monitor in background
monitor_thread = threading.Thread(target=resource_monitor, args=(90, 90), daemon=True)
monitor_thread.start()

embedder = SentenceTransformer('paraphrase-MiniLM-L12-v2', device=device)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)

MODEL ='Llama-2'
if MODEL == 'Phi-2':
    param={'layers':35}
    model_path=r"D:\Python\dating coach\text-generation-webui\models\Phi-2\phi-2.Q4_K_M.gguf"
if MODEL == 'Llama-2':
    param={'layers':120}
    model_path=r"D:\Python\dating coach\text-generation-webui\models\Llama-2\Pocket-Llama2-3.2-3B-Instruct.i1-Q4_K_M.gguf"
if MODEL == 'Gemma-2B':
    param={'layers':70}
    model_path=r"D:\Python\dating coach\text-generation-webui\models\Gemma-2B\Vikhr-Gemma-2B-instruct-Q4_K_M.gguf"


llm = Llama(
    model_path=model_path,
    n_ctx=4096,
    n_threads=8,
    n_gpu_layers=param['layers'] # Adjust for your VRAM budget
)
print('llm length:',llm.context_params)
# Load data
csv_docs, pdf_docs = [], []
csv_path = r'D:/Python/dating coach/formatted_data.csv'
pdf_path = r'D:/Python/dating coach/book/book.pdf'

csv_df = pd.read_csv(csv_path)
csv_docs = [f"{row['title']}: {row['text']}" for _, row in csv_df.iterrows()]
# csv_docs = [f"{item['title']}: {item['text']}" for item in csv_data]

def chunk_text(text, max_tokens=300):
    words = text.split()
    for i in range(0, len(words), max_tokens):
        yield " ".join(words[i:i+max_tokens])

def extract_pdf_chunks(pdf_path):
    '''Extract text from PDF and split into chunks.'''
    reader = PdfReader(pdf_path)
    chunks = []
    current = ""
    for page in reader.pages:
        text = page.extract_text()
        if not text:
            continue
        for line in text.split('\n'):
            line = re.sub(r'\s+', ' ', line).strip()
            if len(current) + len(line) < 1000:
                current += " " + line
            else:
                chunks.append(current.strip())
                current = line
    if current:
        chunks.append(current.strip())
    return chunks

pdf_docs = extract_pdf_chunks(pdf_path)

# Embed all
csv_emb = embedder.encode(csv_docs, convert_to_numpy=True)
pdf_emb = embedder.encode(pdf_docs, convert_to_numpy=True)
dim = csv_emb.shape[1]
csv_index = faiss.IndexFlatL2(dim)
pdf_index = faiss.IndexFlatL2(dim)
csv_index.add(csv_emb)
pdf_index.add(pdf_emb)

def retrieve_faiss(index, docs, query, k=15):
    '''Retrieve top-k documents using FAISS and SentenceTransformer embeddings.'''
    query_emb = embedder.encode([query])
    D, I = index.search(query_emb, k)
    return [docs[i] for i in I[0]]

# Utility functions
def rerank(query, candidates):
    '''Rerank candidates using a cross-encoder.'''
    pairs = [(query, c) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked]

def fuse_chunks(chunks):
    '''Fuse similar chunks into a single coherent text.'''
    lines = "\n".join(chunks).split("\n")
    lines = list(dict.fromkeys([l.strip() for l in lines if l.strip()]))
    return "\n".join(lines)

def phi_summarize(query, context):
    prompt = (
        "Instruct: You are a friendly, confident dating coach.\n"
        f"User question: {query}\n"
        f"Context: {context}\n"
        "Write exactly 3 bullet points with practical examples of what to say or do. "
        "Be specific, clear, helpful. End with one friendly tip to keep the conversation going.\n"
        "Output:"
    )
    return generate_response(prompt)

def shrink_context(context):
    """Use the model itself to reduce context to fit."""
    shrink_prompt = (
        "Instruct: Please summarize the following text in 5-6 sentences "
        "while keeping all key details clear and practical.\n\n"
        f"{context}\n\n"
        "Output:"
    )
    logging.info(f"[SHRINK] Summarizing oversized context of length {len(context)}")
    result = llm(
        shrink_prompt,
        max_tokens=200,
        temperature=0.3,
        top_p=0.9,
        echo=False
    )
    text = result["choices"][0]["text"].strip()
    logging.info(f"[SHRINK] Reduced context to length {len(text)}")
    return text


def generate_response(prompt, max_total_length=5000):
    logging.info(f"[SAFE GENERATE] Prompt length: {len(prompt)} chars")
    start_time = time.time()

    # Heuristic: 1 token ~ 4 chars
    estimated_tokens = len(prompt) / 4

    if estimated_tokens > 480:  # Close to 512 token limit
        logging.warning(f"[SAFE GENERATE] Prompt too big ({estimated_tokens:.0f} tokens). Shrinking context.")

        # Try splitting on "Context:" to extract and shrink
        parts = prompt.split("Context:", 1)
        if len(parts) == 2:
            head, context = parts
            reduced_context = shrink_context(context)
            prompt = f"{head}Context: {reduced_context}\nOutput:"
        else:
            logging.warning("[SAFE GENERATE] No context detected to shrink. Truncating prompt.")
            prompt = prompt[:1500]

    result = llm(
        prompt,
        max_tokens=300,
        temperature=0.7,
        top_p=0.95,
        echo=False
    )

    text = result["choices"][0]["text"].strip()
    logging.info(f"[SAFE GENERATE] Took {time.time() - start_time:.2f}s")
    return text


@app.post("/chat")
async def chat(request: ChatRequest):
    query = request.query
    logging.info(f"[INPUT QUERY]: {query}")
    if len(query.split()) < 5:
        K_RETRIEVE  = 8
    else:
        K_RETRIEVE  = 10
    total_start = time.time()

    # # ========== CSV Branch ==========
    # step_start = time.time()
    # csv_hits = retrieve_faiss(csv_index, csv_docs, query, k=K_RETRIEVE )
    # logging.info(f"[CSV] Retrieved {len(csv_hits)} hits in {time.time() - step_start:.2f}s")

    # step_start = time.time()
    # csv_reranked = rerank(query, csv_hits)
    # csv_fused = fuse_chunks(csv_reranked)
    # logging.info(f"[CSV] Reranked and fused in {time.time() - step_start:.2f}s")

    # step_start = time.time()
    # csv_summary = phi_summarize(query, csv_fused)
    # logging.info(f"[CSV] Summarized with Hermes in {time.time() - step_start:.2f}s")
    # logging.info(f"[CSV] Summary: {csv_summary}")


    # ========== Book Branch ==========
    step_start = time.time()
    book_hits = retrieve_faiss(pdf_index, pdf_docs, query, k=K_RETRIEVE )
    logging.info(f"[Book] Retrieved {len(book_hits)} hits in {time.time() - step_start:.2f}s")

    step_start = time.time()
    book_reranked = rerank(query, book_hits)
    book_fused = fuse_chunks(book_reranked)
    logging.info(f"[Book] Reranked and fused in {time.time() - step_start:.2f}s")

    step_start = time.time()
    book_summary = phi_summarize(query, book_fused)
    logging.info(f"[Book] Summarized with Hermes in {time.time() - step_start:.2f}s")
    logging.info(f"[Book] Summary: {book_summary}")

    # ========== Final Merge ==========
    final_prompt = (
        f"You are a confident, friendly dating coach. The user asked:\n\n"
        f"\"{query}\"\n\n"
        # f"Here is situational advice from a CSV-trained model:\n{csv_summary}\n\n"
        f"Here is book-based advice:\n{book_summary}\n\n"
        "Now combine these into clear bullet points. "
        "Each bullet point MUST include a *practical example* of what to say or do. "
        "Avoid being too generic. Make it sound natural and real. "
        "End with a follow up question."
    )

    step_start = time.time()

    answer = generate_response(final_prompt)
    logging.info(f"[Final Merge] Generated answer in {time.time() - step_start:.2f}s")
    logging.info(f"[FINAL ANSWER]: {answer}")

    logging.info(f"[TOTAL RUNTIME]: {time.time() - total_start:.2f}s")

    return {"response": answer}

