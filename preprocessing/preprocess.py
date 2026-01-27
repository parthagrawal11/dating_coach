import os
import re
import json
import nltk
import pdfplumber
from perplexity import Perplexity
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util

nltk.download("punkt")

# -------------------------------
# CONFIG
# -------------------------------
api_key = os.getenv("PERPLEXITY_API_KEY")
if not api_key:
    raise EnvironmentError("PERPLEXITY_API_KEY environment variable not set. Please set it before running.")
else:
    print("PERPLEXITY_API_KEY found.")
client = Perplexity(api_key=api_key)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# 1. Extract and chunk all text sources
# -------------------------------
def extract_pdf_text(pdf_path: str) -> str:
    all_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                all_text.append(page_text)
    return " ".join(all_text)

def chunk_text(text: str, max_sentences: int = 3) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i:i+max_sentences])
        if chunk:
            chunks.append(chunk)
    return chunks

# -------------------------------
# 2. Generate or load questions dynamically
# -------------------------------
def generate_dynamic_questions(title: str, raw_text: str, n: int = 5) -> List[str]:
    prompt = f"""
    You are a dating coach dataset builder.
    Given the book titled "{title}", generate {n} distinct, meaningful, human-like questions that a user might ask based on its content.
    Example topics: dating, texting, relationships, confidence, first date, rejection.
    Return only the questions as a numbered list.
    """
    # Limit API calls to avoid exceeding credit (e.g., max 100 calls per run)
    if not hasattr(generate_dynamic_questions, "call_count"):
        generate_dynamic_questions.call_count = 0
    MAX_CALLS = 2
    if generate_dynamic_questions.call_count >= MAX_CALLS:
        return ValueError("Max API call limit reached for question generation.")
    try:
        resp = client.chat.completions.create(
            model="sonar-pro",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=50
        )
        generate_dynamic_questions.call_count += 1
        # For non-streaming, use .choices[0].message.content
        text = resp.choices[0].message.content if hasattr(resp, "choices") else str(resp)
        questions = re.findall(r'\d+\.\s*(.+)', text)
        return questions if questions else [line.strip("-• ") for line in text.split("\n") if line.strip()]
    except Exception as e:
        print("Error generating questions:", e)
        return ValueError("Max API call limit reached for question generation.")

# -------------------------------
# 3. Retrieve relevant chunks
# -------------------------------
def find_relevant_chunks(question: str, chunks: List[str], top_k: int = 3) -> List[str]:
    if not chunks or top_k <= 0:
        return []
    q_emb = embedder.encode(question, convert_to_tensor=True)
    c_embs = embedder.encode(chunks, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(q_emb, c_embs)[0]
    if similarities.numel() == 0:
        return []
    top_indices = np.argsort(similarities.cpu().numpy())[::-1]
    top_indices = top_indices[:top_k] if top_k > 0 else top_indices
    return [chunks[i] for i in top_indices]

# -------------------------------
# 4. Optionally LLM-frame coherent answer
# -------------------------------
def frame_answer_llm(question: str, selected_chunks: List[str]) -> str:
    joined = " ".join(selected_chunks)
    prompt = f"""
    You are a dating coach content refiner.
    Question: "{question}"
    The following text snippets are related answers or advice from a dating guide:
    {joined}
    Combine and rewrite them into one coherent, human-sounding answer suitable for a chatbot coach.
    """
    try:
        resp = client.chat.completions.create(
            model="sonar-pro",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=50
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("Error framing answer:", e)
        return joined

# -------------------------------
# 5. Build dataset (Question-first)
# -------------------------------
def build_question_first_dataset(pdf_path: str, n_questions: int = 10, top_k: int = 3) -> List[Dict[str, str]]:
    title = pdf_path.split("/")[-1] if "/" in pdf_path else pdf_path.split("\\")[-1]
    raw_text = extract_pdf_text(pdf_path)
    chunks = chunk_text(raw_text)

    print("Generating questions...")
    questions = generate_dynamic_questions(title, raw_text, n=n_questions)
    dataset = []

    for q in questions:

        relevant_chunks = find_relevant_chunks(q, chunks, top_k=top_k)
        response = frame_answer_llm(q, relevant_chunks)
        dataset.append({
            "instruction": q,
            "response": response
        })
    return dataset

# -------------------------------
# 6. Save to JSONL
# -------------------------------
def save_jsonl(data: List[Dict[str, str]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# -------------------------------
# 7. Run Example
# -------------------------------
if __name__ == "__main__":
    pdf_path = "D:/Python/dating coach/book/book.pdf"
    out_path = "qa_question_first.jsonl"
    qa_dataset = build_question_first_dataset(pdf_path, n_questions=50, top_k=4)
    save_jsonl(qa_dataset, out_path)
    print(f"✅ Saved {len(qa_dataset)} Q/A pairs to {out_path}")


'''| Step   | Logic                                                              | Example                                             |
| ------ | ------------------------------------------------------------------ | --------------------------------------------------- |
| **1.** | Extract & chunk text                                               | Paragraphs like “When she called…”                  |
| **2.** | Generate 50 **dynamic questions** using GPT (based on the book) | “How should a guy respond when a girl calls first?” |
| **3.** | Compute **similarity** between question & chunks using embeddings  | Top 3–4 most relevant text blocks selected          |
| **4.** | Send those to GPT to **rewrite as one coherent answer**            | Combined answer flows naturally                     |
| **5.** | Save as JSONL with `instruction` → `response` pairs                | Ready for finetuning                                |
'''