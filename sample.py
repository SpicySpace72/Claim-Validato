import os
import re
import uuid
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI


# ============================================================
# APP CONFIGURATION
# ============================================================

app = FastAPI(
    title="VeriClaim API",
    description="Evidence-grounded AI claim verification API",
    version="1.0.0"
)

# Allow the frontend to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# AI MODELS
# ============================================================

print("Loading BGE embedding model...")

embedding_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

print("BGE model loaded successfully.")


# ============================================================
# GROQ CLIENT
# ============================================================

groq_api_key = os.environ.get("GROQ_API_KEY")

if not groq_api_key:
    raise RuntimeError(
        "GROQ_API_KEY is not set. "
        "Set the environment variable before starting the server."
    )

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=groq_api_key
)

# Current Groq production model
GROQ_MODEL = "llama-3.1-8b-instant"


# ============================================================
# HELPER: EXTRACT TEXT FROM PDF / TXT
# ============================================================

async def extract_text(file: UploadFile) -> str:

    filename = file.filename or ""

    if not filename:
        raise ValueError("Uploaded file has no filename.")

    extension = filename.lower().split(".")[-1]

    # -------------------------
    # PDF
    # -------------------------
    if extension == "pdf":

        try:
            contents = await file.read()

            from io import BytesIO

            reader = PdfReader(BytesIO(contents))

            pages = []

            for page in reader.pages:
                text = page.extract_text()

                if text:
                    pages.append(text)

            extracted_text = "\n".join(pages)

            return extracted_text.strip()

        except Exception as e:
            raise ValueError(
                f"Could not read PDF '{filename}': {str(e)}"
            )

    # -------------------------
    # TXT
    # -------------------------
    elif extension == "txt":

        try:
            contents = await file.read()

            return contents.decode(
                "utf-8",
                errors="ignore"
            ).strip()

        except Exception as e:
            raise ValueError(
                f"Could not read TXT '{filename}': {str(e)}"
            )

    # -------------------------
    # Unsupported file
    # -------------------------
    else:

        raise ValueError(
            f"Unsupported file type: '{filename}'. "
            "Please upload PDF or TXT files."
        )


# ============================================================
# HELPER: SPLIT TEXT INTO SENTENCES
# ============================================================

def split_into_sentences(text: str):

    # Basic sentence splitting suitable for this MVP
    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    # Remove empty strings
    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    return sentences


# ============================================================
# HELPER: GET CHARACTER SPAN
# ============================================================

def get_span(full_text: str, sentence: str):

    start = full_text.find(sentence)

    if start == -1:
        return [0, 0]

    end = start + len(sentence)

    return [start, end]


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
async def root():

    return {
        "status": "online",
        "service": "VeriClaim API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():

    return {
        "status": "healthy"
    }


# ============================================================
# VERIFY CLAIMS
# ============================================================

@app.post("/verify")
async def verify_claim(
    draft: str = Form(...),
    files: List[UploadFile] = File(...)
):

    try:

        # ----------------------------------------------------
        # VALIDATE INPUT
        # ----------------------------------------------------

        if not draft.strip():

            raise HTTPException(
                status_code=400,
                detail="AI draft answer cannot be empty."
            )

        if not files:

            raise HTTPException(
                status_code=400,
                detail="At least one PDF or TXT source file is required."
            )


        # ----------------------------------------------------
        # READ SOURCE DOCUMENTS
        # ----------------------------------------------------

        all_sentences = []

        source_documents = {}

        for uploaded_file in files:

            filename = uploaded_file.filename or "unknown"

            try:

                text = await extract_text(uploaded_file)

            except ValueError as e:

                raise HTTPException(
                    status_code=400,
                    detail=str(e)
                )

            if not text:

                continue

            source_documents[filename] = text

            sentences = split_into_sentences(text)

            for sentence in sentences:

                all_sentences.append({
                    "src": filename,
                    "text": sentence
                })


        # ----------------------------------------------------
        # CHECK WHETHER DOCUMENTS CONTAIN TEXT
        # ----------------------------------------------------

        if not all_sentences:

            raise HTTPException(
                status_code=400,
                detail=(
                    "No readable text was found in the uploaded "
                    "PDF/TXT files."
                )
            )


        # ----------------------------------------------------
        # SPLIT AI DRAFT INTO CLAIMS
        # ----------------------------------------------------

        claims = split_into_sentences(draft)

        if not claims:

            raise HTTPException(
                status_code=400,
                detail="Could not extract claims from the AI draft."
            )


        # ----------------------------------------------------
        # CREATE SOURCE EMBEDDINGS
        # ----------------------------------------------------

        source_texts = [
            item["text"]
            for item in all_sentences
        ]

        source_embeddings = embedding_model.encode(
            source_texts,
            convert_to_tensor=True
        )


        # ----------------------------------------------------
        # VERIFY EACH CLAIM
        # ----------------------------------------------------

        verified_claims = []

        for claim in claims:

            # -----------------------------------------------
            # EMBED CLAIM
            # -----------------------------------------------

            claim_embedding = embedding_model.encode(
                claim,
                convert_to_tensor=True
            )


            # -----------------------------------------------
            # SEMANTIC SIMILARITY
            # -----------------------------------------------

            similarity_scores = util.cos_sim(
                claim_embedding,
                source_embeddings
            )[0]


            # -----------------------------------------------
            # GET TOP 3 EVIDENCE CANDIDATES
            # -----------------------------------------------

            top_k = min(
                3,
                len(all_sentences)
            )

            top_results = similarity_scores.topk(
                k=top_k
            )


            candidates = []

            for score, index in zip(
                top_results.values,
                top_results.indices
            ):

                index = int(index)

                candidates.append({
                    "src": all_sentences[index]["src"],
                    "text": all_sentences[index]["text"],
                    "score": float(score)
                })


            # -----------------------------------------------
            # BEST MATCH
            # -----------------------------------------------

            best_match = candidates[0]

            best_match_text = best_match["text"]
            best_match_src = best_match["src"]
            best_score = best_match["score"]


            # -----------------------------------------------
            # LLM VERIFICATION
            # -----------------------------------------------

            evidence_context = "\n\n".join(
                [
                    f"SOURCE {i + 1}:\n{candidate['text']}"
                    for i, candidate in enumerate(candidates)
                ]
            )


            prompt = f"""
You are an evidence-grounded claim verification system.

Your task is to determine whether the AI claim is supported
by the provided source evidence.

AI CLAIM:
{claim}

SOURCE EVIDENCE:
{evidence_context}

Classify the claim using exactly one of these labels:

SUPPORTED
DIVERGENT
UNVERIFIED

Definitions:

SUPPORTED:
The source evidence clearly supports the claim.

DIVERGENT:
The source evidence directly contradicts the claim.

UNVERIFIED:
The source evidence does not provide enough information
to determine whether the claim is true.

Return ONLY the following format:

VERDICT: SUPPORTED
REASON: short explanation

or

VERDICT: DIVERGENT
REASON: short explanation

or

VERDICT: UNVERIFIED
REASON: short explanation
"""


            try:

                response = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a precise factual "
                                "verification assistant."
                            )
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0
                )


                llm_output = (
                    response.choices[0]
                    .message
                    .content
                    .strip()
                )


            except Exception as e:

                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Groq verification request failed: "
                        f"{str(e)}"
                    )
                )


            # -----------------------------------------------
            # PARSE LLM VERDICT
            # -----------------------------------------------

            verdict_match = re.search(
                r"VERDICT\s*:\s*(SUPPORTED|DIVERGENT|UNVERIFIED)",
                llm_output,
                re.IGNORECASE
            )

            reason_match = re.search(
                r"REASON\s*:\s*(.*)",
                llm_output,
                re.IGNORECASE
            )


            if verdict_match:

                claim_verdict = (
                    verdict_match.group(1)
                    .upper()
                )

            else:

                claim_verdict = "UNVERIFIED"


            if reason_match:

                reason = reason_match.group(1).strip()

            else:

                reason = llm_output


            # -----------------------------------------------
            # SOURCE CHARACTER SPAN
            # -----------------------------------------------

            full_source_text = source_documents[
                best_match_src
            ]

            span = get_span(
                full_source_text,
                best_match_text
            )


            # -----------------------------------------------
            # ADD RESULT
            # -----------------------------------------------

            verified_claims.append({

                "text": claim,

                "verdict": claim_verdict,

                "reason": reason,

                "evidence": [

                    {
                        "src": best_match_src,

                        "span": span,

                        "span_text": best_match_text,

                        "match": round(
                            best_score,
                            4
                        )
                    }

                ]

            })


        # ----------------------------------------------------
        # OVERALL VERDICT
        # ----------------------------------------------------

        verdicts = [
            claim["verdict"]
            for claim in verified_claims
        ]


        if "DIVERGENT" in verdicts:

            overall_verdict = "REVIEW_REQUIRED"

        elif "UNVERIFIED" in verdicts:

            overall_verdict = "REVIEW_REQUIRED"

        else:

            overall_verdict = "VERIFIED"


        # ----------------------------------------------------
        # CERTIFICATE ID
        # ----------------------------------------------------

        cert_id = (
            "vc_"
            + uuid.uuid4().hex[:8]
        )


        # ----------------------------------------------------
        # FINAL RESPONSE
        # ----------------------------------------------------

        return {

            "cert_id": cert_id,

            "verdict": overall_verdict,

            "claims": verified_claims

        }


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )