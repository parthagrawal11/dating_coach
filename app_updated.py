from fastapi import FastAPI
from pydantic import BaseModel
import torch
import re
import pandas as pd
import os
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

def is_heading(line):
    """Identify if a line is a heading based on simple heuristics."""
    return line.isupper() or len(line.split()) < 5

def extract_sections_from_pdf(pdf_path):
    """Extract sections from a PDF using PyPDF2."""
    sections = []
    reader = PdfReader(pdf_path)
    current_section = None
    
    for page in reader.pages:
        text = page.extract_text()
        if text:
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Clean line: remove extra whitespace and special characters
                line = re.sub(r'\s+', ' ', line)
                if is_heading(line):
                    if current_section:
                        sections.append(current_section)
                    current_section = {'title': line, 'text': ''}
                elif current_section:
                    current_section['text'] += line + ' '
    
    if current_section:
        sections.append(current_section)
    
    # Log sections for debugging
    print(f"Extracted {len(sections)} sections from PDF")
    for i, section in enumerate(sections):
        print(f"Section {i+1}: Title={section['title'][:50]}, Text={section['text'][:100]}...")
    
    return sections

app = FastAPI()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

# Load model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(r'D:/Python/dating coach/dating_coach/dating_coach_gpt2/final_new')
    model = AutoModelForCausalLM.from_pretrained(r'D:/Python/dating coach/dating_coach/dating_coach_gpt2/final_new')
    model.to(device)
    model.eval()
except Exception as e:
    print(f"Model load failed: {e}")
    exit(1)

# Load RAG components (CSV and PDF)
csv_path = r'D:/Python/dating coach/formatted_data.csv'  # From format_situational_data.py
pdf_path = r'D:/Python/dating coach/book.pdf'  # Replace with your PDF path
try:
    # Load CSV data
    csv_df = pd.read_csv(csv_path)
    csv_data = [{'title': row['situation'], 'text': str(row['text'])} for _, row in csv_df.iterrows()]
    
    # Load PDF data
    pdf_data = extract_sections_from_pdf(pdf_path)
    
    # Combine CSV and PDF data
    documents = []
    for item in csv_data:
        documents.append(f"{item['title']}: {item['text']}")
    for item in pdf_data:
        documents.append(f"{item['title']}: {item['text']}")
    
    # Initialize RAG
    embedder = SentenceTransformer('paraphrase-MiniLM-L12-v2')
    embeddings = embedder.encode(documents, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
except Exception as e:
    print(f"RAG initialization failed: {e}")
    documents = []
    embedder = None
    index = None

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Retrieve context with RAG
        context = ""
        if documents and embedder and index:
            query_embedding = embedder.encode([request.query])
            D, I = index.search(query_embedding, k=5)
            context = "\n".join([documents[i] for i in I[0]])

        prompt = (
            f"Context: {context}\n"
            f"{request.query}\n"
            f"Give a short, friendly, and conversational response with bullet points. For each mistake (texting too much, being overly eager, not following up), explain briefly how to avoid it and suggest a fun text example. End with a next-step tip to keep the chat going. Skip unrelated stuff like social media."
        )
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=256)
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