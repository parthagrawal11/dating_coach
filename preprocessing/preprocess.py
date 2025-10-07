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
# 1. Extract text from PDF
# -------------------------------
def extract_pdf_text(pdf_path: str) -> str:
    all_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                all_text.append(page_text)
    return " ".join(all_text)

# -------------------------------
# 2. Chunk text
# -------------------------------
def chunk_text(text: str, max_sentences: int = 3) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i:i+max_sentences])
        if chunk:
            chunks.append(chunk)
    return chunks

# -------------------------------
# 3. Generate questions dynamically (LLM)
# -------------------------------
def generate_dynamic_question(title: str, chunk: str) -> str:
    prompt = f"""
    You are a dating coach content transformer.
    Given this section from a book titled "{title}", create one natural question that this text answers.
    Text: "{chunk}"
    Question:
    """
    # Limit API calls to avoid exceeding credit (e.g., max 100 calls per run)
    if not hasattr(generate_dynamic_question, "call_count"):
        generate_dynamic_question.call_count = 0
    MAX_CALLS = 100
    if generate_dynamic_question.call_count >= MAX_CALLS:
        return "What advice does this section provide?"
    try:
        resp = client.chat.completions.create(
            model="sonar-pro",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=50
        )
        generate_dynamic_question.call_count += 1
        question = resp.choices[0].message.content.strip()
        return question
    except Exception as e:
        print("Error generating question:", e)
        return "What advice does this section provide?"

# -------------------------------
# 4. Combine related chunks using embeddings
# -------------------------------
def merge_related_chunks(chunks: List[str], similarity_threshold: float = 0.65) -> List[str]:
    embeddings = embedder.encode(chunks, convert_to_tensor=True)
    merged_responses = []
    used = set()
    for i, chunk in enumerate(chunks):
        if i in used:
            continue
        related = [chunk]
        for j in range(i + 1, len(chunks)):
            if j in used:
                continue
            sim = util.pytorch_cos_sim(embeddings[i], embeddings[j]).item()
            if sim > similarity_threshold:
                related.append(chunks[j])
                used.add(j)
        merged_responses.append(" ".join(related))
    return merged_responses

# -------------------------------
# 5. Build dataset
# -------------------------------
def create_qa_pairs_from_pdf(pdf_path: str) -> List[Dict[str, str]]:
    title = pdf_path.split("/")[-1] if "/" in pdf_path else pdf_path.split("\\")[-1]
    raw_text = extract_pdf_text(pdf_path)
    chunks = chunk_text(raw_text)
    merged_chunks = merge_related_chunks(chunks)

    qa_pairs = []
    for chunk in merged_chunks:
        question = generate_dynamic_question(title, chunk)
        qa_pairs.append({
            "instruction": question,
            "response": chunk.strip()
        })
    return qa_pairs

# -------------------------------
# 6. Save to JSONL
# -------------------------------
def save_qa_pairs_jsonl(pairs: List[Dict[str, str]], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

# -------------------------------
# 7. Run Example
# -------------------------------
if __name__ == "__main__":
    pdf_path = "D:/Python/dating coach/book/book.pdf"
    output_path = "qa_dynamic_output.jsonl"
    qa_pairs = create_qa_pairs_from_pdf(pdf_path)
    save_qa_pairs_jsonl(qa_pairs, output_path)
    print(f"✅ Saved {len(qa_pairs)} dynamic Q/A pairs → {output_path}")



'''| Step                            | Description                                                                      | Example                                             |
| ------------------------------- | -------------------------------------------------------------------------------- | --------------------------------------------------- |
| **Dynamic question generation** | LLM reads each chunk and writes a natural question.                              | “What should a man do when a girl calls him first?” |
| **Semantic merging**            | Nearby or similar chunks are combined using **Sentence-BERT cosine similarity**. | 2–3 related paragraphs merged into one response.    |
| **Flexible**                    | Works for PDFs, can extend for transcripts later.                                | Yes.                                                |
'''