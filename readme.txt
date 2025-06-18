when using transformers_env
run in same directory wheer app.py is present.
powershell: uvicorn app:app --host 127.0.0.1 --port 8000

bash: curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d "{\"query\":\"Mistakes Guys Make After Getting Her Number\nCommon mistakes include texting too much too soon, being overly eager, or not following up quickly enough.\"}"

when using book_env
run in same directory wheer app.py is present.
uvicorn app:app --host 127.0.0.1 --port 8000

curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d "{\"query\":\"How can I strategize a pickup line to ask her out at a coffee shop?\"}"