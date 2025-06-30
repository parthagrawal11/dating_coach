from fastapi import FastAPI
from pydantic import BaseModel
import torch
import re
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
from PyPDF2 import PdfReader

class ChatRequest(BaseModel):
    query: str

def rewrite_chunk_local(chunk: str) -> str:
    instruction = (
        "Rewrite the following book advice in a casual tone like a dating coach, "
        "as 2-3 short bullet points:\n\n" + chunk
    )
    inputs = rewriter_tokenizer(instruction, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = rewriter_model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
    return rewriter_tokenizer.decode(outputs[0], skip_special_tokens=True)


def clean_response(text, query):
    text = re.sub(re.escape(query), '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'\[.*?\]', '', text).strip()
    text = re.sub(r'<\|endoftext\|>', '', text).strip()
    irrelevant = r'\b(bhaiya|yaar|bro|good night|welcome back|thank god|complaints|subscribe|social media|channel|followers|mentorship|geekhoopermusic|eyeembrace|blueprint|clicking here|online dating)\b'
    text = re.sub(irrelevant, '', text, flags=re.IGNORECASE).strip()
    sentences = [s.strip() for s in text.split('. ') if s.strip()]
    if sentences and not sentences[-1].endswith('.'):
        sentences[-1] += '.'
    required_phrases = ['texting too much', 'overly eager', 'following up']
    response_lower = text.lower()
    missing_phrases = [phrase for phrase in required_phrases if phrase not in response_lower]
    if len(sentences) < 2:
        additions = [
            "Don’t text too much! Send one chill text in 24 hours, like: 'Hey, fun meeting you! Coffee soon?'",
            "Stay cool, don’t seem too eager. Text once a day, like: 'Yo, loved our chat! What’s good?'",
            "Follow up in 1–2 days to keep it real, like: 'Hey, how’s it hangin’? Had a blast talking!'",
            "Next, if she replies, ask about her day to keep the vibe going."
        ]
        sentences = additions
    negative = r'\b(negative|stuck|struggling|reject|pressure|less interested)\b'
    text = re.sub(negative, 'positive', text, flags=re.IGNORECASE)
    return '\n'.join([f"- {s}" for s in sentences if s]) + '\nLet’s keep that connection vibing!'

def extract_chunks_from_pdf(pdf_path, max_chunk_len=1000):
    reader = PdfReader(pdf_path)
    chunks = []
    current = ""
    for page in reader.pages:
        text = page.extract_text()
        if not text:
            continue
        lines = text.split('\n')
        for line in lines:
            line = re.sub(r'\s+', ' ', line).strip()
            if len(current) + len(line) < max_chunk_len:
                current += " " + line
            else:
                chunks.append(current.strip())
                current = line
    if current:
        chunks.append(current.strip())
    return [{'title': f"BookChunk-{i}", 'text': chunk} for i, chunk in enumerate(chunks)]



app = FastAPI()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

# Load locally instead of calling OpenAI
rewriter_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
rewriter_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)

try:
    tokenizer = AutoTokenizer.from_pretrained(r'D:/Python/dating coach/dating_coach/dating_coach_gpt2/final_new')
    model = AutoModelForCausalLM.from_pretrained(r'D:/Python/dating coach/dating_coach/dating_coach_gpt2/final_new')
    model.to(device)
    model.eval()
except Exception as e:
    print(f"Model load failed: {e}")
    exit(1)

csv_path = r'D:/Python/dating coach/formatted_data.csv'
pdf_path = r'D:/Python/dating coach/book/book.pdf'

try:
    csv_df = pd.read_csv(csv_path)
    csv_data = [{'title': row['title'], 'text': str(row['text'])} for _, row in csv_df.iterrows()]
    csv_docs = [f"{item['title']}: {item['text']}" for item in csv_data]

    pdf_data = extract_chunks_from_pdf(pdf_path)
    pdf_docs_raw = [item['text'] for item in pdf_data[:10]]  # limit for quick rewrite test
    pdf_docs_rewritten = [rewrite_chunk_local(text) for text in pdf_docs_raw]
    for chunk in pdf_docs_rewritten[:3]:
        print("--- REWRITTEN BOOK CHUNK ---")
        print(chunk)
    embedder = SentenceTransformer('paraphrase-MiniLM-L12-v2')

    csv_embeddings = embedder.encode(csv_docs, convert_to_numpy=True)
    pdf_embeddings = embedder.encode(pdf_docs_rewritten, convert_to_numpy=True)

    dim = csv_embeddings.shape[1]
    csv_index = faiss.IndexFlatL2(dim)
    pdf_index = faiss.IndexFlatL2(dim)

    csv_index.add(csv_embeddings)
    pdf_index.add(pdf_embeddings)

except Exception as e:
    print(f"RAG init failed: {e}")
    csv_docs, pdf_docs_rewritten = [], []
    embedder, csv_index, pdf_index = None, None, None

def filter_context_notes(raw_context: str) -> str:
    instruction = (
        "Summarize this dating advice into 3-5 bullet points or notes for a coach to use in a reply. "
        "Keep it short and relevant. Avoid repeating questions, irrelevant text, or instructions."
        "\n\n" + raw_context
    )
    inputs = rewriter_tokenizer(instruction, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = rewriter_model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
    return rewriter_tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        context_parts = []
        if embedder and csv_index and pdf_index:
            query_embedding = embedder.encode([request.query])
            D_csv, I_csv = csv_index.search(query_embedding, k=3)
            D_pdf, I_pdf = pdf_index.search(query_embedding, k=2)
            top_csv = [csv_docs[i] for i in I_csv[0]]
            top_pdf = [pdf_docs_rewritten[i] for i in I_pdf[0]]
            context_parts = top_csv + top_pdf

        raw_context = "\n".join(context_parts)
        filtered_context = filter_context_notes(raw_context)
        example = (
                        "User: How can I make a strong impression on a first date?\n"
                        "Coach:\n"
                        "- Pick a simple plan, like coffee or a walk. Show confidence in the plan.\n"
                        "- Keep the vibe fun: “So what’s your superpower?” is better than boring job talk.\n"
                        "- Notice her energy — if she’s smiling or leaning in, you’re on the right track.\n"
                        "Let’s keep that connection vibing!\n\n"
                    )

        prompt = (
                    f"You are a confident, cool dating coach. Use the notes below to write exactly 3 short bullet points giving casual advice with examples. End with one final tip to keep things going.\n"
                    f"Notes:\n{filtered_context}\n"
                    f"User: {request.query}\n"
                    f"Coach:\n-"
                )


        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

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
