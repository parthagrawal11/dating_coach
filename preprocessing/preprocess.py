
import random
from typing import List, Dict
from itertools import islice
import re
import nltk
import pdfplumber

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

def chunk_text(text: str, max_sentences: int = 3) -> List[str]:
    """
    Split text into chunks of sentences for combining later.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    window = 2  # smaller window for more overlap
    stride = 1  # overlap chunks for more Q/A
    for i in range(0, len(sentences) - window + 1, stride):
        chunk = " ".join(sentences[i:i+window])
        if chunk:
            chunks.append(chunk)
    # Add any remaining sentence as a chunk
    if len(sentences) % stride != 0 and len(sentences) > 0:
        chunk = " ".join(sentences[-window:])
        if chunk and chunk not in chunks:
            chunks.append(chunk)
    return chunks

def generate_questions(title: str, text: str) -> List[str]:
    """
    Generate candidate questions from title and some heuristics.
    """
    title_clean = re.sub(r'[_:]', '', title).strip()
    questions = [
        f"What is the main idea of '{title_clean}'?",
        f"Summarize the advice in '{title_clean}'.",
        f"What can we learn from '{title_clean}'?",
        f"What about {title_clean.lower()}?"
    ]
    # Add more generic and situational questions
    generic_qs = [
        "What to do on calls?",
        "How should men behave?",
        "What happens in this situation?",
        "What should I say or do?",
        "What is the best approach?",
        "What is the key takeaway?",
        "How to handle this scenario?",
        "What is the recommended action?"
    ]
    questions.extend(random.sample(generic_qs, k=min(4, len(generic_qs))))
    return questions

def combine_chunks(chunks: List[str], window: int = 2) -> List[str]:
    """
    Combine nearby chunks into longer responses.
    """
    responses = []
    it = iter(chunks)
    while True:
        part = list(islice(it, window))
        if not part:
            break
        responses.append(" ".join(part))
    return responses

def create_qa_pairs(title: str, raw_text: str) -> List[Dict[str, str]]:
    """
    Create Q/A pairs from raw text + title.
    """
    # Step 1: chunk text
    chunks = chunk_text(raw_text, max_sentences=2)
    # Step 2: generate questions
    questions = generate_questions(title, raw_text)
    # Step 3: combine nearby chunks for richer answers
    responses = combine_chunks(chunks, window=1)  # use window=1 for more granular answers

    # Step 4: build pairs (one question per chunk, plus some cross-pairing)
    qa_pairs = []
    for i, r in enumerate(responses):
        # Remove all newlines from the response
        clean_response = r.replace('\n', ' ').replace('\r', ' ').strip()
        for q in questions:
            qa_pairs.append({
                "question": q,
                "response": clean_response
            })
    return qa_pairs


# ------------------ Save to JSONL ------------------
def save_qa_pairs_jsonl(pairs: List[Dict[str, str]], output_path: str):
    import json
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")


# ------------------ Example ------------------
if __name__ == "__main__":
    # Use the provided PDF file path and extract text using pdfplumber
    file_path = "D:/Python/dating coach/book/book.pdf"
    title = file_path.split("/")[-1] if "/" in file_path else file_path.split("\\")[-1]
    # Extract text from PDF
    all_text = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                all_text.append(page_text)
    raw_text = " ".join(all_text)
    qa_dataset = create_qa_pairs(title, raw_text)
    save_qa_pairs_jsonl(qa_dataset, "qa_output.jsonl")
    print(f"Saved {len(qa_dataset)} Q/A pairs to qa_output.jsonl")

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
