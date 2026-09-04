# VeriClaim

Evidence-grounded AI claim verification API.

Large language models produce fluent answers that are difficult to audit. VeriClaim takes an AI-generated draft, breaks it into individual claims, and checks each one against source documents you supply. Every claim comes back with a verdict, a plain-English reason, and a pointer to the exact character span in the source that justifies it.

## How it works

1. **Ingest** — PDF and TXT sources are parsed, whitespace-normalised, and split into sentences. Fragments under four words (list numbers, headings) are discarded.
2. **Retrieve** — each claim is embedded with `BAAI/bge-small-en-v1.5` and matched against source sentences by cosine similarity. Scores are weighted by source authority; the top 3 form the evidence context.
3. **Adjudicate** — `openai/gpt-oss-20b` on Groq classifies the claim against that evidence as `SUPPORTED`, `DIVERGENT`, or `UNVERIFIED`, at `temperature=0`. It also flags whether the sources disagree with each other.
4. **Locate** — the matched sentence is resolved to a character span `[start, end]` in the original document so a frontend can highlight it directly.

Retrieval runs locally, so source documents never leave the machine — the LLM only ever sees the three most relevant sentences rather than the whole corpus.

## Source authority

Real organisations hold a current policy alongside outdated documents that contradict it. Similarity alone cannot tell which one to trust, so each uploaded file can be assigned an authority tier:

| Tier | Weight |
|---|---|
| `statutory` | 1.00 |
| `policy` | 0.97 |
| `internal` | 0.94 |
| `web` | 0.90 |

Weights affect **ranking only** — the reported `match` score stays the raw, unweighted similarity. When candidates come from different documents and give materially different answers, the claim is flagged `conflict: true` and the document verdict escalates to `REVIEW_REQUIRED`. Disagreement is surfaced rather than silently resolved.

## Reproducibility

`cert_id` is a SHA-256 hash of the draft, the extracted document text, the model name, and the prompt version. Identical inputs always produce an identical certificate, so a reviewer can re-run a verification and confirm nothing was substituted. Every response carries an `audit` block naming the models and prompt version used.

## Stack

- FastAPI + Uvicorn
- sentence-transformers (`BAAI/bge-small-en-v1.5`), 384-dim, runs on CPU
- Groq API (`openai/gpt-oss-20b`) for verdict reasoning
- pypdf for document extraction
- Vanilla HTML/JS frontend, no build step

## Setup

```bash
py -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

py -m pip install fastapi uvicorn python-multipart pypdf sentence-transformers openai requests
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

Interactive docs at `http://127.0.0.1:8000/docs`. Open `frontend/index.html` in a browser for the UI — no build step required.

## API

### `POST /verify`

`multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `draft` | string | The AI-generated text to verify |
| `files` | file[] | One or more PDF or TXT source documents |
| `trust` | string | Optional. Comma-separated authority tiers, positionally matched to `files`. Defaults to `internal`. |

Response:

```json
{
  "cert_id": "vc_9f2a41c8b03d",
  "verdict": "REVIEW_REQUIRED",
  "audit": {
    "model": "openai/gpt-oss-20b",
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "prompt_version": "v1.1",
    "sources": ["Sample.pdf", "old_handbook.txt"]
  },
  "claims": [
    {
      "text": "Leave requests must be submitted at least 7 working days in advance.",
      "verdict": "DIVERGENT",
      "reason": "Source 1 states leave requests must be submitted at least 3 working days in advance, which contradicts the claim of 7 working days.",
      "conflict": true,
      "conflicting_sources": ["Sample.pdf", "old_handbook.txt"],
      "evidence": [
        {
          "src": "Sample.pdf",
          "span": [97, 214],
          "span_text": "Leave Application Employees must submit their leave requests at least 3 working days before the requested leave date.",
          "match": 0.8469
        }
      ]
    }
  ]
}
```

The document verdict is `VERIFIED` only when every claim is `SUPPORTED` and no conflict was detected; otherwise `REVIEW_REQUIRED`.

### `GET /health`

Liveness probe. Returns `{"status": "healthy"}`.

## Evaluation

`eval_dataset.json` contains 18 hand-labelled claims against `Sample.pdf`, spanning all three verdict types. The document and labels were authored by the team; no third-party licensing applies.

```bash
py evaluate.py
```

Reports per-claim grounding accuracy, a confusion matrix, and timing. Current result: 18/18 correct at roughly 1.1 seconds per claim.

This is a single self-authored document, so the figure demonstrates that the pipeline works end to end rather than production-grade accuracy. A larger benchmark on third-party documents is the next step.

## Known limitations

- No OCR, so scanned PDFs without a text layer extract nothing.
- Claims are judged at sentence granularity. A sentence containing two separate facts is treated as one claim.
- Only the top-scoring evidence sentence is returned, though the top three inform the verdict.
- Embeddings are recomputed per request; a vector store would remove that cost.
- `UNVERIFIED` claims still attach their nearest match even when similarity is low.

## Notes

- `GROQ_API_KEY` is read from the environment and is never committed.
- CORS is open for development and should be restricted before deployment.
