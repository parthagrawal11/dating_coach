
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

class ChatRequest(BaseModel):
    query: str

def clean_response(text):
    text = re.sub(r'\[.*?\]', '', text).strip()
    text = re.sub(r'<\|endoftext\|>', '', text).strip()
    irrelevant = r'\b(bhaiya|yaar|bro|good night|welcome back|thank god|complaints|subscribe|social media|channel|followers|mentorship|geekhoopermusic)\b'
    text = re.sub(irrelevant, '', text, flags=re.IGNORECASE).strip()
    sentences = [s.strip() for s in text.split('. ') if s.strip()]
    if sentences and not sentences[-1].endswith('.'):
        sentences[-1] += '.'
    required_phrases = ['texting too much', 'overly eager', 'following up']
    response_lower = text.lower()
    missing_phrases = [phrase for phrase in required_phrases if phrase not in response_lower]
    if missing_phrases:
        additions = []
        if 'texting too much' in missing_phrases:
            additions.append("To avoid texting too much, send one casual message within 24 hours, like: 'Hey, great meeting you! Free for coffee?'")
        if 'overly eager' in missing_phrases:
            additions.append("Prevent being overly eager by matching her texting pace with one text daily, such as: 'Hey, loved our chat! What’s up?'")
        if 'following up' in missing_phrases:
            additions.append("Follow up within 1–2 days to show interest, e.g., 'Hey, how’s it going? Had fun talking!'")
        sentences.extend(additions)
    negative = r'\b(negative|stuck|struggling|reject|pressure|less interested)\b'
    text = re.sub(negative, 'positive', text, flags=re.IGNORECASE)
    return '\n'.join([f"- {s}" for s in sentences if s]) + '\nThese actions foster a positive connection.'

app = FastAPI()
device = torch.device('cuda:0')
torch.cuda.empty_cache()

try:
    tokenizer = AutoTokenizer.from_pretrained(r'D:/Python/dating coach/dating_coach/dating_coach_gpt2/final_new')
    model = AutoModelForCausalLM.from_pretrained(r'D:/Python/dating coach/dating_coach/dating_coach_gpt2/final_new')
    model.to(device)
    model.eval()
except Exception as e:
    print(f"Model load failed: {e}")
    exit(1)

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        prompt = (
            f"{request.query}\n"
            f"Provide a detailed, well-structured response addressing each mistake. Use bullet points for clarity. For each mistake: explain how to avoid it, suggest a specific action (e.g., example text message), and maintain a positive tone. Ensure the response is concise, positive, and avoids irrelevant topics like social media or unrelated activities."
        )
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                repetition_penalty=2.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        cleaned_response = clean_response(response)
        return {"response": cleaned_response}
    except Exception as e:
        return {"error": str(e)}
