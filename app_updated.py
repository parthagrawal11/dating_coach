from fastapi import FastAPI
from pydantic import BaseModel
import torch
import re
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
from PyPDF2 import PdfReader

class ChatRequest(BaseModel):
    query: str

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
    if missing_phrases or not sentences:
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

try:
    tokenizer = AutoTokenizer.from_pretrained(r'D:/Python/dating coach/dating_coach/dating_coach_gpt2/final_new')
    model = AutoModelForCausalLM.from_pretrained(r'D:/Python/dating coach/dating_coach/dating_coach_gpt2/final_new')
    model.to(device)
    model.eval()
except Exception as e:
    print(f"Model load failed: {e}")
    exit(1)

# Load RAG data separately
csv_path = r'D:/Python/dating coach/formatted_data.csv'
pdf_path = r'D:/Python/dating coach/book/book.pdf'

try:
    csv_df = pd.read_csv(csv_path)
    csv_data = [{'title': row['situation'], 'text': str(row['text'])} for _, row in csv_df.iterrows()]
    csv_docs = [f"{item['title']}: {item['text']}" for item in csv_data]

    pdf_data = extract_chunks_from_pdf(pdf_path)
    pdf_docs = [f"{item['title']}: {item['text']}" for item in pdf_data]

    embedder = SentenceTransformer('paraphrase-MiniLM-L12-v2')

    # Separate indices
    csv_embeddings = embedder.encode(csv_docs, convert_to_numpy=True)
    pdf_embeddings = embedder.encode(pdf_docs, convert_to_numpy=True)

    dim = csv_embeddings.shape[1]
    csv_index = faiss.IndexFlatL2(dim)
    pdf_index = faiss.IndexFlatL2(dim)

    csv_index.add(csv_embeddings)
    pdf_index.add(pdf_embeddings)

except Exception as e:
    print(f"RAG initialization failed: {e}")
    csv_docs, pdf_docs = [], []
    embedder, csv_index, pdf_index = None, None, None

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        context_parts = []
        if embedder and csv_index and pdf_index:
            query_embedding = embedder.encode([request.query])

            # Retrieve from both sources
            D_csv, I_csv = csv_index.search(query_embedding, k=3)
            D_pdf, I_pdf = pdf_index.search(query_embedding, k=2)

            top_csv = [csv_docs[i] for i in I_csv[0]]
            top_pdf = [pdf_docs[i] for i in I_pdf[0]]

            context_parts = top_csv + top_pdf

        context = "\n".join(context_parts)

        prompt = (
            f"Context: {context}\n"
            f"{request.query}\n"
            f"Give a short, friendly, and conversational response with bullet points. For each mistake (texting too much, being overly eager, not following up), explain briefly how to avoid it and suggest a fun text example. End with a next-step tip to keep the chat going. Skip unrelated stuff like social media."
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
 