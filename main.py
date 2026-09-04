import os
import re
import uuid
import hashlib
from typing import Annotated

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
from fastapi.openapi.utils import get_openapi


# ============================================================
# APP CONFIGURATION
# ============================================================

app = FastAPI(
    title="VeriClaim API",
    description="Evidence-grounded AI claim verification API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# SWAGGER FILE UPLOAD COMPATIBILITY FIX
# ============================================================

def custom_openapi():

    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Force OpenAPI 3.0.3 for better Swagger file-upload support
    schema["openapi"] = "3.0.3"

    components = schema.get(
        "components",
        {}
    ).get(
        "schemas",
        {}
    )

    for schema_data in components.values():

        properties = schema_data.get(
            "properties",
            {}
        )

        for property_data in properties.values():

            # Single file
            if (
                property_data.get("type") == "string"
                and "contentMediaType" in property_data
            ):
                property_data.pop(
                    "contentMediaType",
                    None
                )

                property_data["format"] = "binary"

            # Multiple files
            if property_data.get("type") == "array":

                items = property_data.get(
                    "items",
                    {}
                )

                if "contentMediaType" in items:

                    items.pop(
                        "contentMediaType",
                        None
                    )

                    items["type"] = "string"
                    items["format"] = "binary"

    app.openapi_schema = schema

    return app.openapi_schema


app.openapi = custom_openapi


# ============================================================
# AI MODEL
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

GROQ_MODEL = "openai/gpt-oss-20b"
PROMPT_VERSION = "v1.1"
TRUST_WEIGHTS = {
    "statutory": 1.00,
    "policy": 0.97,
    "internal": 0.94,
    "web": 0.90
}
DEFAULT_TRUST = "internal"

# ============================================================
# EXTRACT TEXT FROM PDF / TXT
# ============================================================

async def extract_text(file: UploadFile) -> str:

    filename = file.filename or ""

    if not filename:
        raise ValueError("Uploaded file has no filename.")

    extension = filename.lower().split(".")[-1]

    # PDF
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

            return "\n".join(pages).strip()

        except Exception as e:

            raise ValueError(
                f"Could not read PDF '{filename}': {str(e)}"
            )

    # TXT
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

    # Unsupported
    else:

        raise ValueError(
            f"Unsupported file type: '{filename}'. "
            "Please upload PDF or TXT files."
        )


# ============================================================
# SPLIT TEXT INTO SENTENCES
# ============================================================

def split_into_sentences(text: str):

    text = re.sub(r'\s+', ' ', text)

    sentences = re.split(
        r'(?<![A-Z0-9])(?<=[.!?])\s+(?=[A-Z"\'])',
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ============================================================
# GET CHARACTER SPAN
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
    draft: Annotated[str, Form(...)],
    files: Annotated[list[UploadFile], File(...)],
    trust: Annotated[str, Form()] = ""
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
        source_trust = {}

        trust_list = [
            t.strip().lower()
            for t in trust.split(",")
            if t.strip()
        ]

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
            text = re.sub(r'\s+', ' ', text).strip()
            source_documents[filename] = text

            tier = (
                trust_list[len(source_trust)]
                if len(source_trust) < len(trust_list)
                else DEFAULT_TRUST
            )

            if tier not in TRUST_WEIGHTS:
                tier = DEFAULT_TRUST

            source_trust[filename] = tier

            sentences = split_into_sentences(text)

            for sentence in sentences:

                if len(sentence.split()) < 4:
                    continue

                all_sentences.append({
                    "src": filename,
                    "text": sentence
                })


        # ----------------------------------------------------
        # CHECK SOURCE TEXT
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

            # Embed claim
            claim_embedding = embedding_model.encode(
                claim,
                convert_to_tensor=True
            )


            # Semantic similarity
            similarity_scores = util.cos_sim(
                claim_embedding,
                source_embeddings
            )[0]


            # Top 3 candidates
            # Apply source authority weighting to ranking only
            weighted_scores = similarity_scores.clone()

            for i, item in enumerate(all_sentences):
                weighted_scores[i] *= TRUST_WEIGHTS[
                    source_trust.get(item["src"], DEFAULT_TRUST)
                ]

            top_k = min(3, len(all_sentences))

            top_results = weighted_scores.topk(k=top_k)

            candidates = []

            for index in top_results.indices:

                index = int(index)

                candidates.append({
                    "src": all_sentences[index]["src"],
                    "text": all_sentences[index]["text"],
                    "score": float(similarity_scores[index]),
                    "trust": source_trust.get(
                        all_sentences[index]["src"],
                        DEFAULT_TRUST
                    )
                })


            # Best evidence
            best_match = candidates[0]

            best_match_text = best_match["text"]
            best_match_src = best_match["src"]
            best_score = best_match["score"]


            # ------------------------------------------------
            # LLM VERIFICATION
            # ------------------------------------------------

            evidence_context = "\n\n".join(
                [
                    f"SOURCE {i + 1} "
                    f"(document: {candidate['src']}, "
                    f"authority: {candidate['trust']}):\n"
                    f"{candidate['text']}"
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

Some sources carry more authority than others.
Authority ranking, highest first: statutory, policy, internal, web.
If sources disagree, base your verdict on the highest-authority source.

Return ONLY this format:

VERDICT: SUPPORTED or DIVERGENT or UNVERIFIED
REASON: short explanation
CONFLICT: YES or NO

Set CONFLICT to YES only when two or more sources give
materially different answers about this specific claim.
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


            # ------------------------------------------------
            # PARSE VERDICT
            # ------------------------------------------------

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
            conflict_match = re.search(
                r"CONFLICT\s*:\s*(YES|NO)",
                llm_output,
                re.IGNORECASE
            )

            has_conflict = bool(
                conflict_match
                and conflict_match.group(1).upper() == "YES"
                and len({c["src"] for c in candidates}) > 1
            )

            conflicting_sources = (
                sorted({c["src"] for c in candidates})
                if has_conflict
                else []
            )

            if verdict_match:

                claim_verdict = (
                    verdict_match.group(1).upper()
                )

            else:

                claim_verdict = "UNVERIFIED"


            if reason_match:

                reason = reason_match.group(1).strip()

            else:

                reason = llm_output


            # ------------------------------------------------
            # SOURCE CHARACTER SPAN
            # ------------------------------------------------

            full_source_text = source_documents[
                best_match_src
            ]

            span = get_span(
                full_source_text,
                best_match_text
            )


            # ------------------------------------------------
            # ADD CLAIM RESULT
            # ------------------------------------------------

            verified_claims.append({

                "text": claim,

                "verdict": claim_verdict,

                "reason": reason,
                "conflict": has_conflict,

                "conflicting_sources": conflicting_sources,

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


        if any(c["conflict"] for c in verified_claims):

            overall_verdict = "REVIEW_REQUIRED"

        elif "DIVERGENT" in verdicts:

            overall_verdict = "REVIEW_REQUIRED"

        elif "UNVERIFIED" in verdicts:

            overall_verdict = "REVIEW_REQUIRED"

        else:

            overall_verdict = "VERIFIED"


        # ----------------------------------------------------
        # CERTIFICATE ID
        # ----------------------------------------------------

        fingerprint = "|".join([
            draft.strip(),
            "|".join(f"{name}:{text}" for name, text in sorted(source_documents.items())),
            GROQ_MODEL,
            PROMPT_VERSION
        ])

        cert_id = "vc_" + hashlib.sha256(
            fingerprint.encode("utf-8")
        ).hexdigest()[:12]


        # ----------------------------------------------------
        # FINAL RESPONSE
        # ----------------------------------------------------

        return {
            "cert_id": cert_id,
            "verdict": overall_verdict,
            "audit": {
                "model": GROQ_MODEL,
                "embedding_model": "BAAI/bge-small-en-v1.5",
                "prompt_version": PROMPT_VERSION,
                "sources": sorted(source_documents.keys())
            },
            "claims": verified_claims
        }


    except HTTPException:
        raise

    except Exception as e:

        print("UNEXPECTED ERROR:", str(e))

        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


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