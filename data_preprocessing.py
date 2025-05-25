import os
import re
from datasets import Dataset
import pandas as pd
from uuid import uuid4

def parse_text_file(file_path,filename):
    if filename.endswith(".txt"):
        """Parse a single text file to extract title and content."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Assume first line is the title (scenario)
        lines = content.strip().split('\n')
        if len(lines) < 2:
            return None, None  # Skip malformed files
        title = filename.split(".")[0]
        patterns_to_remove = [
        r'🔴 \[LIVE Webinar\]_ ',
        r'📌',
        r'🤯',
        r'🥳',
        r'😤',
        r'💸',
        r'🔥',
        r'😉',
        r'\[500K Offer\]_',
        r'_ Hindi _',
        r'_ Hindi',
    ]
        for pattern in patterns_to_remove:
            title = re.sub(pattern, '', title)
        # Remove #01, #shorts, or anything immediately following #
        title = re.sub(r'#[^ ]*', '', title)
        
        # Remove any pattern like [...]_ (e.g., [500K Offer]_, [50K Special]_)
        title = re.sub(r'\[.*?\]_', '', title)
        
        # Remove any bracketed text like [Client Results], [500K Offer], etc.
        title = re.sub(r'\[.*?\]', '', title)
        
        # Remove any parenthetical text like (Approach Anxiety)
        title = re.sub(r'\(.*?\)', '', title)
        
        # Remove leading/trailing underscores but preserve content between them
        title = re.sub(r'^_+|_+$', '', title)
        text = ' '.join(line.strip() for line in lines[1:] if line.strip())
        # Ensure title is unique
        return title, text

def identify_examples(text):
    """Identify positive and negative examples in text using heuristics."""
    sentences = re.split(r'[.!?]\s+', text)
    positive_examples = []
    negative_examples = []
    
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        # Heuristics for positive examples
        if any(keyword in sent.lower() for keyword in ['do this', 'try this', 'example', 'good', 'recommended', 'should', 'great']):
            positive_examples.append(sent)
        # Heuristics for negative examples
        elif any(keyword in sent.lower() for keyword in ['avoid', 'don’t', 'not', 'bad', 'wrong', 'mistake']):
            negative_examples.append(sent)
        # Default: Assume positive if it looks like a direct quote/action
        elif sent.startswith('"') or 'text:' in sent.lower():
            positive_examples.append(sent)
    
    return positive_examples, negative_examples

def format_sample(title, text, positive_examples, negative_examples):
    """Format a single sample for training."""
    # Concept is the first 1-2 sentences of text (or fallback to text)
    concept_sentences = re.split(r'[.!?]\s+', text)[:2]
    concept = ' '.join(concept_sentences).strip()
    if not concept:
        concept = text[:200] + '...' if len(text) > 200 else text
    
    # Limit examples to avoid over-length samples
    pos_example = positive_examples[0] if positive_examples else "No positive example provided."
    neg_example = negative_examples[0] if negative_examples else "No negative example provided."
    
    # Dialogue format
    sample = (
        f"User: What advice do you have for {title.lower()}?\n"
        f"Assistant: For {title.lower()}, {concept.lower()}. "
        f"Try something like: {pos_example} "
        f"Avoid things like: {neg_example}"
    )
    
    return sample

def preprocess_dataset(data_dir, output_file='formatted_data.csv'):
    """Process all text files in the directory and save formatted samples."""
    samples = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(data_dir, filename)
            title, text = parse_text_file(file_path,filename)
            if not title or not text:
                continue
            
            # Identify positive/negative examples
            positive_examples, negative_examples = identify_examples(text)
            
            # Format sample
            sample = format_sample(title, text, positive_examples, negative_examples)
            samples.append({'text': sample, 'title': title})
    
    # Save to CSV for inspection
    df = pd.DataFrame(samples)
    df.to_csv(output_file, index=False)
    
    # Convert to Dataset
    dataset = Dataset.from_pandas(df[['text']])
    return dataset, len(samples)

# Example usage
if __name__ == "__main__":
    DATA_DIR = "./hinglish"  # Replace with your dataset directory
    OUTPUT_FILE = "formatted_data.csv"
    
    dataset, sample_count = preprocess_dataset(DATA_DIR, OUTPUT_FILE)
    print(f"Processed {sample_count} samples. Saved to {OUTPUT_FILE}")
    
