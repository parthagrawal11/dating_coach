when using transformers_env
run in same directory where app.py is present.
powershell: uvicorn app:app --host 127.0.0.1 --port 8000

bash: curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d "{\"query\":\"Mistakes Guys Make After Getting Her Number\nCommon mistakes include texting too much too soon, being overly eager, or not following up quickly enough.\"}"

when using book_env
run in same directory wheer app.py is present.
uvicorn app:app --host 127.0.0.1 --port 8000
in case of environment conflicts: C:\Users\ASUS\anaconda3\envs\book_env\Scripts\uvicorn.exe app_updated:app --host 127.0.0.1 --port 8000 --reload 

curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d "{\"query\":\"How can I strategize a pickup line to ask her out at a coffee shop?\"}"

##Architecture1##

User Query
   ↓
CSV Branch:
  - Retrieve → Rerank → Summarize 1 (mistral)
  - Summarize 1 + Query → GPT-2 → Output1

Book Branch:
  - Retrieve → Rerank → Summarize 2 (t5)

Final Merger:
  - Output1  + Summarize 2 + Query → Mistral → Final Answer

###Architecture2### Failed

User Query
   ↓
CSV Branch:
  - Retrieve → Rerank → Summarize 1 (mistral)

Book Branch:
  - Retrieve → Rerank → Summarize 2 (mistral)

Final Merger:
  - Summarize 1  + Summarize 2 + Query → Mistral → Final Answer

##Arch3##

1️⃣ CSV branch
  - retrieve -> rerank -> fuse
  - summarize with Nous-Hermes

2️⃣ Book branch
  - retrieve -> rerank -> fuse
  - summarize with Nous-Hermes

4️⃣ Final merge with Nous-Hermes
  - csv summary + book summary + question -> Hermes


Data flow 

hinglish--> formatted Data

transcripts-->hinglish

data preprocessing: to process hinglish to formatted data