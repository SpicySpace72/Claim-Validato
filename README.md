# VeriClaim

Evidence-grounded AI claim verification API.

Large language models produce fluent answers that are difficult to audit. VeriClaim takes an AI-generated draft, breaks it into individual claims, and checks each one against source documents you supply. Every claim comes back with a verdict and a pointer to the exact sentence in the source that justifies it.

## How it works

1. **Ingest** — PDF and TXT sources are parsed and split into sentences.
2. **Retrieve** — each claim from the draft is embedded with `BAAI/bge-small-en-v1.5` and matched against the source sentences by cosine similarity. The top 3 candidates form the evidence context.
3. **Adjudicate** — a Groq-hosted Llama 3.1 model classifies the claim against that evidence as `SUPPORTED`, `DIVERGENT`, or `UNVERIFIED`.
4. **Locate** — the matched sentence is resolved to a character span `[start, end]` in the original document so a frontend can highlight it directly.

Retrieval is local, so the LLM only ever sees the three most relevant sentences rather than the whole corpus.

## Stack

- FastAPI + Uvicorn
- sentence-transformers (`BAAI/bge-small-en-v1.5`) for local embeddings
- Groq API (`llama-3.1-8b-instant`) for verdict reasoning
- pypdf for document extraction

## Setup

```bash
py -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

py -m pip install fastapi uvicorn python-multipart pypdf sentence-transformers openai
```

Set your Groq API key:

```powershell
$env:GROQ_API_KEY = "your_key_here"     # Windows PowerShell
```

```bash
export GROQ_API_KEY="your_key_here"     # macOS / Linux
```

Run the server:

```bash
py -m uvicorn main:app --reload --port 8000
```

Interactive docs are at `http://127.0.0.1:8000/docs`.

## API

### `POST /verify`

`multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `draft` | string | The AI-generated text to verify |
| `files` | file[] | One or more PDF or TXT source documents |

Response:

```json
{
  "cert_id": "vc_9f2a41",
  "verdict": "REVIEW_REQUIRED",
  "claims": [
    {
      "text": "The policy covers water damage from burst pipes.",
      "verdict": "SUPPORTED",
      "evidence": [
        {
          "src": "policy.pdf",
          "span": [1420, 1495],
          "span_text": "Sudden and accidental discharge from plumbing is covered.",
          "match": 0.87
        }
      ]
    }
  ]
}
```

The document-level `verdict` is `VERIFIED` only when no claim is `DIVERGENT`; otherwise it is `REVIEW_REQUIRED`.

### `GET /health`

Liveness probe. Returns `{"status": "healthy"}`.

## Notes

- Only PDF and TXT sources are supported. Scanned PDFs without a text layer will not extract.
- `GROQ_API_KEY` is read from the environment and is never committed to the repository.
- CORS is open during development and should be restricted to known origins before deployment.
