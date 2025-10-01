import re
import json
import nltk
import pdfplumber
from pathlib import Path

nltk.download("punkt")

# -------------------------------
# 1. Extract PDF text
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
# 2. Clean and split into chunks
# -------------------------------
def chunk_text(text: str, max_words: int = 60):
    sentences = nltk.sent_tokenize(text)
    chunks, current, count = [], [], 0
    for sent in sentences:
        words = sent.split()
        if count + len(words) > max_words:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sent)
        count += len(words)
    if current:
        chunks.append(" ".join(current))
    return chunks

# -------------------------------
# 3. Heuristic Question Generator
# -------------------------------
def generate_question(chunk: str) -> str:
    """
    Rule-based: pick keywords to form generic questions.
    Can be replaced with GPT for smarter Q-generation.
    """
    chunk_lower = chunk.lower()
    if "call" in chunk_lower:
        return "What to do on calls?"
    elif "date" in chunk_lower:
        return "How to plan a date?"
    elif "drink" in chunk_lower:
        return "Should you buy drinks for someone?"
    elif "text" in chunk_lower or "message" in chunk_lower:
        return "How to handle texting?"
    else:
        return "What should you do in this situation?"

# -------------------------------
# 4. Build Q–A pairs
# -------------------------------
def build_pairs_from_pdf(pdf_path: str, output="pdf_finetune.jsonl"):
    raw_text = extract_pdf_text(pdf_path)
    chunks = chunk_text(raw_text)

    pairs = []
    for chunk in chunks:
        question = generate_question(chunk)
        # Remove all newlines from the response
        response = chunk.replace('\n', ' ').replace('\r', ' ').strip()
        pairs.append({
            "instruction": question,
            "response": response
        })

    with open(output, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return pairs

# -------------------------------
# Example Run
# -------------------------------
if __name__ == "__main__":
    pdf_dir = "D:/Python/dating coach/book/book.pdf"
    pdf_file = pdf_dir
    dataset = build_pairs_from_pdf(pdf_file)
    print("Sample:", dataset[:3])

# # -------------------------------
# # Example usage
# # -------------------------------
# if __name__ == "__main__":
#     pdf_dir = "D:/Python/dating coach/book"
#     transcript_dir = "D:/Python/dating coach/transcripts"
#     dataset = build_dataset(
#         pdf_dir=pdf_dir, 
#         transcript_dir=transcript_dir,
#         output_path="dating_coach.jsonl"
#     )
#     print(dataset)
