import re
import json
import pandas as pd
from pathlib import Path

# Optional grammar correction
try:
    import language_tool_python
    tool = language_tool_python.LanguageTool('en-US')
except:
    tool = None

def clean_sentence(text: str) -> str:
    """Clean grammar lightly"""
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    if tool:
        matches = tool.check(text)
        text = language_tool_python.utils.correct(text, matches)
    return text.capitalize()

def split_question_answer(text: str):
    """
    Try to split text into question and response parts
    Example: "should men buy drink drinks they should"
    → ("should men buy drink drinks", "they should")
    """
    words = text.split()
    # naive heuristic: split at last wh-word or '?'
    for i, w in enumerate(words):
        if w.lower() in ["should", "why", "what", "how", "when", "where", "who"]:
            # assume response starts after last 2 words
            if len(words) > i+3:
                return " ".join(words[:i+3]), " ".join(words[i+3:])
    return None, None

def build_pairs_from_txt(txt_path: str, title: str):
    pairs = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text: 
                continue

            # Split into Q/A if possible
            q, a = split_question_answer(text)
            if q and a:
                pairs.append({
                    "instruction": clean_sentence(q),
                    "response": clean_sentence(a)
                })
            else:
                # fallback: use title as question, line as response
                if len(text.split()) > 3:  # ignore too short lines
                    pairs.append({
                        "instruction": clean_sentence(title),
                        "response": clean_sentence(text)
                    })
    return pairs

def process_transcripts(txt_dir: str, output="finetune.jsonl"):
    data = []
    for txt_file in Path(txt_dir).glob("*.txt"):
        # Extract title from filename (or metadata)
        title = txt_file.stem.replace("_", " ")
        pairs = build_pairs_from_txt(str(txt_file), title)
        data.extend(pairs)

    with open(output, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return data

# ------------------------
# Example Run
# ------------------------
if __name__ == "__main__":
    dataset = process_transcripts("data/transcripts")
    print("Sample:", dataset[:5])
