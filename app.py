
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

app = FastAPI()
device = torch.device('cuda:0')
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

# Load RAG components
csv_path = r'D:/Python/dating coach/formatted_data.csv'  # Adjust
try:
    df = pd.read_csv(csv_path)
    data = [{'title': row['title'], 'text': str(row['text'])} for _, row in df.iterrows()]
    documents = [f"{item['title']}: {item['text']}" for item in data]
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
            D, I = index.search(query_embedding, k=3)
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
                max_new_tokens=200,  # Increased
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
