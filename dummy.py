import os
import json
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import nltk

# Download sentence splitter on first run
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

app = FastAPI()

# 1. CORS Middleware (Allows Ashwin's frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load local embedding model
embedder = SentenceTransformer('BAAI/bge-small-en-v1.5')

# 2. API Key from Environment Variable
groq_api_key = os.environ.get("GROQ_API_KEY")
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=groq_api_key
) 

def extract_text(file: UploadFile) -> str:
    if file.filename.lower().endswith('.pdf'):
        reader = PdfReader(file.file)
        return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif file.filename.lower().endswith('.txt'):
        return file.file.read().decode('utf-8')
    return ""

@app.post("/verify")
async def verify_claim(draft: str = Form(...), files: List[UploadFile] = File(...)):
    all_source_sentences = []
    doc_texts = {} 

    # 3. Process Multiple Files (PDF & TXT)
    for file in files:
        full_text = extract_text(file)
        doc_texts[file.filename] = full_text
        sentences = sent_tokenize(full_text)
        
        for sent in sentences:
            all_source_sentences.append({"text": sent, "src": file.filename})
    
    draft_claims = sent_tokenize(draft)
    
    just_sentences = [item["text"] for item in all_source_sentences]
    if not just_sentences:
        return {"cert_id": "vc_error", "verdict": "REVIEW_REQUIRED", "claims": []}
        
    source_embeddings = embedder.encode(just_sentences, convert_to_tensor=True)
    
    results = []
    
    for claim in draft_claims:
        claim_emb = embedder.encode(claim, convert_to_tensor=True)
        hits = util.semantic_search(claim_emb, source_embeddings, top_k=3)[0]
        
        best_hit = hits[0]
        best_match_obj = all_source_sentences[best_hit['corpus_id']]
        best_match_text = best_match_obj["text"]
        best_match_src = best_match_obj["src"]
        
        context = " ".join([all_source_sentences[hit['corpus_id']]["text"] for hit in hits])
        
        prompt = f"""
        Source Context: {context}
        AI Claim: {claim}
        
        Does the Source Context support the AI Claim? 
        Respond in JSON format with exactly two keys:
        "verdict": must be strictly "SUPPORTED" or "DIVERGENT"
        "reason": a 1-sentence explanation
        """
        
        response = client.chat.completions.create(
            model="llama3-8b-8192", 
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        llm_out = json.loads(response.choices[0].message.content)
        
        # 4. Calculate exact character spans for the frontend highlight mapping
        full_doc_text = doc_texts[best_match_src]
        start_idx = full_doc_text.find(best_match_text)
        end_idx = start_idx + len(best_match_text) if start_idx != -1 else 0
        if start_idx == -1: start_idx = 0
        
        results.append({
            "text": claim,
            "verdict": llm_out.get("verdict", "DIVERGENT"),
            "evidence": [{
                "src": best_match_src, 
                "span": [start_idx, end_idx],
                "span_text": best_match_text,
                "match": float(best_hit['score'])
            }]
        })

    doc_verdict = "REVIEW_REQUIRED" if any(r["verdict"] == "DIVERGENT" for r in results) else "VERIFIED"

    return {
        "cert_id": "vc_9f2a41",
        "verdict": doc_verdict,
        "claims": results
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)