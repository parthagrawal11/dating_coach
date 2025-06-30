import pytest
import torch
import re
import pandas as pd
from fastapi.testclient import TestClient
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader
from ..app_updated import app, clean_response, extract_chunks_from_pdf, rewrite_chunk_local

client = TestClient(app)

MODEL_PATH = r'D:/Python/dating coach/dating_coach/dating_coach_gpt2/final_new'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH)
MODEL = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE).eval()

REWRITER_TOKENIZER = AutoTokenizer.from_pretrained("google/flan-t5-base")
REWRITER_MODEL = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(DEVICE)

CSV_PATH = r'D:/Python/dating coach/formatted_data.csv'
PDF_PATH = r'D:/Python/dating coach/book/book.pdf'
EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2')

def load_rag_data():
    csv_df = pd.read_csv(CSV_PATH)
    csv_data = [{'title': row['title'], 'text': str(row['text'])} for _, row in csv_df.iterrows()]
    csv_docs = [f"{item['title']}: {item['text']}" for item in csv_data]
    
    pdf_data = extract_chunks_from_pdf(PDF_PATH)
    pdf_docs_raw = [item['text'] for item in pdf_data[:10]]
    pdf_docs_rewritten = [rewrite_chunk_local(text) for text in pdf_docs_raw]
    
    return csv_docs, pdf_docs_rewritten

@pytest.fixture
def rag():
    csv_docs, pdf_docs_rewritten = load_rag_data()
    csv_embeddings = EMBEDDER.encode(csv_docs, convert_to_numpy=True)
    pdf_embeddings = EMBEDDER.encode(pdf_docs_rewritten, convert_to_numpy=True)
    
    dim = csv_embeddings.shape[1]
    csv_index = faiss.IndexFlatL2(dim)
    pdf_index = faiss.IndexFlatL2(dim)
    csv_index.add(csv_embeddings)
    pdf_index.add(pdf_embeddings)
    
    return csv_docs, pdf_docs_rewritten, csv_index, pdf_index

@pytest.mark.asyncio
async def test_model_inference():
    query = "How can I strategize a pickup line to ask her out at a coffee shop?"
    inputs = TOKENIZER(query, return_tensors='pt', truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = MODEL.generate(**inputs, max_new_tokens=50)
    response = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    print(f"Model output: {response}")
    assert len(response.strip()) > 0, "Model output is empty"
    assert any(word in response.lower() for word in ['coffee', 'pickup', 'ask']), "Model output lacks relevant keywords"

def test_rewrite_chunk_local():
    sample_chunk = "Maintain confidence and avoid excessive communication."
    rewritten = rewrite_chunk_local(sample_chunk)
    print(f"Rewritten chunk: {rewritten}")
    assert rewritten.strip(), "Rewritten chunk is empty"
    assert rewritten.count('- ') >= 2, f"Expected at least 2 bullet points, got {rewritten.count('- ')}"
    assert len(rewritten.split('\n')) <= 4, "Rewritten chunk should be concise"
    assert any(word in rewritten.lower() for word in ['confident', 'chill', 'follow up']), "Rewritten chunk lacks casual tone"

def test_extract_chunks_from_pdf():
    chunks = extract_chunks_from_pdf(PDF_PATH)
    print(f"Extracted {len(chunks)} PDF chunks")
    assert len(chunks) > 0, "No chunks extracted from PDF"
    assert all('title' in chunk and 'text' in chunk for chunk in chunks), "Chunks missing title or text"
    assert all(len(chunk['text']) <= 1000 for chunk in chunks), "Chunks exceed max length"
    print(f"Sample chunk: {chunks[0]['text'][:200]}")

def test_rag_retrieval_relevance(rag):
    csv_docs, pdf_docs_rewritten, csv_index, pdf_index = rag
    query = "How can I strategize a pickup line to ask her out at a coffee shop?"
    query_embedding = EMBEDDER.encode([query])
    
    D_csv, I_csv = csv_index.search(query_embedding, k=3)
    D_pdf, I_pdf = pdf_index.search(query_embedding, k=2)
    retrieved_csv = [csv_docs[i] for i in I_csv[0]]
    retrieved_pdf = [pdf_docs_rewritten[i] for i in I_pdf[0]]
    
    print(f"Retrieved CSV: {retrieved_csv}")
    print(f"Retrieved PDF: {retrieved_pdf}")
    relevant_count = sum(1 for doc in retrieved_csv + retrieved_pdf if 'coffee' in doc.lower() or 'pickup' in doc.lower())
    assert relevant_count >= 2, f"Expected at least 2 relevant documents, got {relevant_count}"
    assert len(retrieved_csv) == 3, f"Expected 3 CSV documents, got {len(retrieved_csv)}"
    assert len(retrieved_pdf) == 2, f"Expected 2 PDF documents, got {len(retrieved_pdf)}"

def test_response_structure():
    query = "How can I strategize a pickup line to ask her out at a coffee shop?"
    dummy_response = (
        "Don’t text too much! Send one chill text like: 'Hey, coffee soon?' "
        "Stay cool, don’t be too eager. Try: 'Feeling a connection?' "
        "Follow up in 1–2 days, like: 'How’s your day?'"
    )
    cleaned = clean_response(dummy_response, query)
    print(f"Cleaned response: {cleaned}")
    assert cleaned.count('- ') >= 3, f"Expected at least 3 bullet points, got {cleaned.count('- ')}"
    assert 'texting too much' in cleaned.lower(), "Missing 'texting too much'"
    assert 'overly eager' in cleaned.lower(), "Missing 'overly eager'"
    assert 'following up' in cleaned.lower(), "Missing 'following up'"
    assert 'Let’s keep that connection vibing!' in cleaned, "Missing next-step tip"

@pytest.mark.asyncio
async def test_fastapi_endpoint():
    query = {"query": "How can I strategize a pickup line to ask her out at a coffee shop?"}
    response = client.post("/chat", json=query)
    assert response.status_code == 200, f"Expected HTTP 200, got {response.status_code}"
    
    response_text = response.json()['response']
    print(f"Endpoint response: {response_text}")
    assert len(response_text.strip()) > 0, "Response is empty"
    assert any(word in response_text.lower() for word in ['coffee', 'pickup', 'ask']), "Response lacks relevant keywords"
    assert response_text.count('- ') >= 3, f"Expected at least 3 bullet points, got {response_text.count('- ')}"
    assert 'Let’s keep that connection vibing!' in response_text, "Missing next-step tip"

def test_rag_weightage(rag):
    csv_docs, pdf_docs_rewritten, csv_index, pdf_index = rag
    queries = [
        "How can I strategize a pickup line to ask her out at a coffee shop?",
        "How to ask her out at a party?",
        "What’s a good way to start a conversation at the gym?"
    ]
    csv_counts = []
    pdf_counts = []
    
    for query in queries:
        query_embedding = EMBEDDER.encode([query])
        D_csv, I_csv = csv_index.search(query_embedding, k=3)
        D_pdf, I_pdf = pdf_index.search(query_embedding, k=2)
        csv_counts.append(len([csv_docs[i] for i in I_csv[0]]))
        pdf_counts.append(len([pdf_docs_rewritten[i] for i in I_pdf[0]]))
    
    avg_csv = sum(csv_counts) / len(csv_counts)
    avg_pdf = sum(pdf_counts) / len(pdf_counts)
    print(f"RAG weightage: CSV={avg_csv}, PDF={avg_pdf}")
    assert avg_csv == 3, f"Expected 3 CSV documents per query, got {avg_csv}"
    assert avg_pdf == 2, f"Expected 2 PDF documents per query, got {avg_pdf}"