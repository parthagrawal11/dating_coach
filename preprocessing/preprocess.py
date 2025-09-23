import os
import re
import json
from pathlib import Path
import pdfplumber
import nltk
from datasets import Dataset
nltk.download("punkt")

# -------------------------------
# 1. Helpers for text cleaning
# -------------------------------
def clean_text(text: str) -> str:
    text = text.replace("\n", " ").strip()
    text = re.sub(r'\s+', ' ', text)  # collapse spaces
    text = re.sub(r'http\S+', '', text)  # remove URLs
    text = re.sub(r'[^A-Za-z0-9.,!?\'" ]+', '', text)  # keep readable chars
    return text

# -------------------------------
# 2. Extract text from PDFs
# -------------------------------
def extract_pdf_text(pdf_path: str) -> str:
    all_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            all_text.append(page.extract_text() or "")
    return clean_text(" ".join(all_text))

# -------------------------------
# 3. Process transcripts
# -------------------------------
def process_transcript(transcript_path: str) -> list:
    """
    Transcript format assumption:
    Each line = "Speaker: text"
    Example: "User: I need help" / "Coach: Try this"
    """
    dialogues = []
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            speaker, text = line.split(":", 1)
            dialogues.append({"speaker": speaker.strip(), "text": clean_text(text)})
    return dialogues

# -------------------------------
# 4. Convert to instruction-response pairs
# -------------------------------
def build_instruction_pairs(dialogues: list) -> list:
    """
    Convert User–Coach into {"instruction": ..., "response": ...}
    """
    pairs = []
    for i in range(len(dialogues) - 1):
        if dialogues[i]["speaker"].lower() == "user" and dialogues[i+1]["speaker"].lower() == "coach":
            pairs.append({
                "instruction": dialogues[i]["text"],
                "response": dialogues[i+1]["text"]
            })
    return pairs

# -------------------------------
# 5. Chunk freeform text (from PDFs)
# -------------------------------
def chunk_text(text: str, max_tokens: int = 200) -> list:
    # naive sentence-based chunking
    sentences = nltk.sent_tokenize(text)
    chunks, current_chunk = [], []
    count = 0
    for sent in sentences:
        words = sent.split()
        if count + len(words) > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk, count = [], 0
        current_chunk.append(sent)
        count += len(words)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# -------------------------------
# 6. Putting it all together
# -------------------------------
def build_dataset(pdf_dir: str, transcript_dir: str, output_path: str = "dating_coach.jsonl"):
    data = []

    # PDFs → chunked text
    for pdf_file in Path(pdf_dir).glob("*.pdf"):
        print("Processing PDF:", pdf_file)
        pdf_text = extract_pdf_text(str(pdf_file))
        print("Extracted text length:", len(pdf_text))
        print("Extracted text (first 500 chars):", pdf_text[:500])
        chunks = chunk_text(pdf_text)
        print("Number of chunks:", len(chunks))
        for c in chunks:
            data.append({"text": c})

    # Transcripts → instruction/response
    for txt_file in Path(transcript_dir).glob("*.txt"):
        dialogues = process_transcript(str(txt_file))
        pairs = build_instruction_pairs(dialogues)
        data.extend(pairs)

    # Save as JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    # Also return HF Dataset for training
    return Dataset.from_list(data)

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    pdf_dir = "D:/Python/dating coach/book"
    transcript_dir = "D:/Python/dating coach/transcripts"
    print("PDF dir absolute:", os.path.abspath(pdf_dir))
    print("Transcript dir absolute:", os.path.abspath(transcript_dir))
    dataset = build_dataset(
        pdf_dir=pdf_dir, 
        transcript_dir=transcript_dir,
        output_path="dating_coach.jsonl"
    )
    print("Total records:", len(dataset))

