from fastapi import FastAPI
from pydantic import BaseModel
import torch
import re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from PyPDF2 import PdfReader
import psutil
import threading
import time
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True  # critical fix!
)

class ChatRequest(BaseModel):
    query: str

def resource_monitor(ram_threshold=90, cpu_threshold=90):
    """Monitor system resources and raise an error if thresholds are exceeded."""
    while True:
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        if mem.percent >= ram_threshold or cpu >= cpu_threshold:
            raise RuntimeError(
                f"ERROR: Resource usage exceeded threshold!\n"
                f"RAM: {mem.percent:.2f}% used ({mem.used / (1024 ** 3):.2f} GB / {mem.total / (1024 ** 3):.2f} GB)\n"
                f"CPU: {cpu:.2f}% used"
            )
        time.sleep(2)  # Check every 2 seconds

# Start resource monitor in background
monitor_thread = threading.Thread(target=resource_monitor, args=(90, 90), daemon=True)
monitor_thread.start()

embedder = SentenceTransformer('paraphrase-MiniLM-L12-v2', device=device)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)

# rewriter_tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
# rewriter_model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base').to(device).eval()

final_model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#change this to use mistral from local cache
final_tokenizer = AutoTokenizer.from_pretrained(final_model_path)
final_model = AutoModelForCausalLM.from_pretrained(
                                                    final_model_path, 
                                                    quantization_config=bnb_config,  # if supported
                                                    device_map="auto", 
                                                    local_files_only=True
                                                )

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

# def generate_with_model(prompt):
#     logging.info(f"[GENERATION] Prompt length: {len(prompt)} chars")
#     start_time = time.time()
#     output = final_model(
#         prompt,
#         max_new_tokens=300,
#         min_length=100,  
#         temperature=0.7,
#         top_p=0.95,
#         repetition_penalty=2.0,
#     )
#     logging.info(f"[GENERATION] Took {time.time() - start_time:.2f}s")
#     return output

# def generate_response(prompt):
    # prompt = (
    #     "Instruct: You are a friendly, confident dating coach.\n"
    #     f"User question: {user_question}\n"
    #     f"Context: {context}\n"
    #     "Write exactly 3 bullet points with practical examples of what to say or do. "
    #     "Be specific, clear, helpful. End with one friendly tip to keep the conversation going.\n"
    #     "Output:"
    # )
    # for chunk in chunk_text(prompt, 300):
    #     result = llm(f"Instruct: {chunk}\nOutput:", max_tokens=100)
    # result = llm(
    #     prompt,
    #     max_tokens=300,
    #     temperature=0.7,
    #     top_p=0.95,
    #     echo=False
    # )
    # return result["choices"][0]["text"]

def generate_with_model(tokenizer, model, prompt):
    logging.info(f"[GENERATION] Prompt length: {len(prompt)} chars")
    start_time = time.time()

    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024).to(device)
    logging.info(f"[GENERATION] Tokenized input: {inputs['input_ids'].shape}")
    logging.info(f"[PROMPT SENT]: {prompt}")
    gen_start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            min_length=100,  
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            repetition_penalty=2.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    gen_time = time.time() - gen_start
    logging.info(f"[GENERATION] Model.generate() took {gen_time:.2f}s")

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logging.info(f"[GENERATION] Output length: {len(decoded)} chars")

    total_time = time.time() - start_time
    logging.info(f"[GENERATION] Total time: {total_time:.2f}s")
    # Strip out the original prompt from the decoded result
    response_only = decoded.replace(prompt, "").strip()
    return response_only

# ---- Summarization with Nous-Hermes ----
# def summarize(query, context):
#     prompt = (
#         "<|system|>You are a friendly, confident dating coach. ...</s>\n"
#         f"<|user|>{query}\n{context}</s>\n"
#         "<|assistant|>"
#     )
#     # return generate_with_model(rewriter_tokenizer, rewriter_model, prompt)
#     return generate_with_model(prompt)

def shrink_context_llm(context, shrink_prompt_base=None):
    shrink_prompt = shrink_prompt_base or (
        "Instruct: Please summarize the following text in 5-6 sentences "
        "while keeping all key details clear and practical.\n\n"
        f"{context}\n\n"
        "Output:"
    )
    logging.info(f"[SHRINK] Shrinking oversized context: {len(context)} chars")
    return generate_with_model(final_tokenizer, final_model, shrink_prompt)


def summarize(query, context):
    
    prompt = (
    "Instruct: You are a friendly, confident dating coach.\n"
    f"User question: {query}\n"
    f"Context: {context}\n"
    "Write exactly 3 bullet points with practical examples of what to say or do. "
    "Be specific, clear, helpful. End with one friendly tip to keep the conversation going.\n"
    "Output:"
    )
    return generate_with_model(final_tokenizer, final_model, prompt)
    # return generate_with_model(prompt)
    # return generate_response(prompt)


@app.post("/chat")
async def chat(request: ChatRequest):
    query = request.query
    logging.info(f"[INPUT QUERY]: {query}")
    if len(query.split()) < 5:
        K_RETRIEVE  = 8
    else:
        K_RETRIEVE  = 10
    total_start = time.time()

    # ========== CSV Branch ==========
    # step_start = time.time()
    # csv_hits = retrieve_faiss(csv_index, csv_docs, query, k=K_RETRIEVE )
    # logging.info(f"[CSV] Retrieved {len(csv_hits)} hits in {time.time() - step_start:.2f}s")

    # step_start = time.time()
    # csv_reranked = rerank(query, csv_hits)
    # csv_fused = fuse_chunks(csv_reranked)
    # logging.info(f"[CSV] Reranked and fused in {time.time() - step_start:.2f}s")
    # MAX_CONTEXT_LEN = 3000
    # if len(csv_fused) > MAX_CONTEXT_LEN:
    #     csv_fused = shrink_context_llm(csv_fused)

    # step_start = time.time()
    # csv_summary = summarize(query, csv_fused)
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
    MAX_CONTEXT_LEN = 3000

    if len(book_fused) > MAX_CONTEXT_LEN:
        book_fused = shrink_context_llm(book_fused)
    step_start = time.time()
    book_summary = summarize(query, book_fused)
    logging.info(f"[Book] Summarized with Hermes in {time.time() - step_start:.2f}s")
    logging.info(f"[Book] Summary: {book_summary}")

    # ========== Final Merge ==========
    final_prompt = (
        f"You are a confident, friendly dating coach. The user asked:\n\n"
        f"\"{query}\"\n\n"
        # f"Here is situational advice from a CSV-trained model:\n{csv_summary}\n\n"
        f"Here is book-based advice:\n{book_summary}\n\n"
        "Now combine these into exactly 3 short, clear bullet points. "
        "Each bullet point MUST include a *practical example* of what to say or do. "
        "Avoid being too generic. Make it sound natural and real. "
        "End with one *friendly tip* to keep the vibe going and make the user feel confident."
    )

    step_start = time.time()
    answer = generate_with_model(final_tokenizer, final_model, final_prompt)
    # answer = generate_with_model(final_prompt)
    # answer = generate_response(final_prompt)
    logging.info(f"[Final Merge] Generated answer in {time.time() - step_start:.2f}s")
    logging.info(f"[FINAL ANSWER]: {answer}")
    logging.info(f"[TOTAL RUNTIME]: {time.time() - total_start:.2f}s")
    return {"response": answer}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.6,
                repetition_penalty=2.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        cleaned_response = clean_response(response, request.query)
        return {"response": cleaned_response}

    except Exception as e:
        return {"error": str(e)}
