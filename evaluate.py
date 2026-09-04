"""
VeriClaim evaluation harness.

Measures per-claim grounding accuracy against a labelled benchmark.

Usage:
    py -m pip install requests
    py evaluate.py

The API must be running on http://127.0.0.1:8000
"""

import json
import sys
import time

import requests


API_URL = "http://127.0.0.1:8000/verify"
DATASET_PATH = "eval_dataset.json"
LABELS = ["SUPPORTED", "DIVERGENT", "UNVERIFIED"]


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_single_claim(claim_text, document_path):
    """Send one claim to the API and return its verdict."""

    with open(document_path, "rb") as document:

        response = requests.post(
            API_URL,
            data={"draft": claim_text},
            files={"files": (document_path, document, "application/pdf")},
            timeout=120
        )

    response.raise_for_status()

    payload = response.json()

    if not payload.get("claims"):
        return None, None

    first = payload["claims"][0]

    return first.get("verdict"), first.get("evidence", [{}])[0].get("match")


def main():

    dataset = load_dataset(DATASET_PATH)
    document = dataset["source_document"]
    claims = dataset["claims"]

    print(f"Benchmark : {dataset['dataset_name']}")
    print(f"Document  : {document}")
    print(f"Claims    : {len(claims)}\n")

    confusion = {
        expected: {predicted: 0 for predicted in LABELS}
        for expected in LABELS
    }

    correct = 0
    failures = []
    started = time.time()

    for claim in claims:

        try:
            predicted, match = verify_single_claim(claim["text"], document)

        except Exception as error:
            print(f"  [{claim['id']:>2}] REQUEST FAILED: {error}")
            continue

        if predicted is None:
            print(f"  [{claim['id']:>2}] NO CLAIMS RETURNED")
            continue

        expected = claim["expected"]

        if predicted in LABELS:
            confusion[expected][predicted] += 1

        if predicted == expected:
            correct += 1
            mark = "PASS"
        else:
            mark = "FAIL"
            failures.append((claim, predicted))

        score = f"{match:.2f}" if match is not None else "  - "

        print(
            f"  [{claim['id']:>2}] {mark}  "
            f"expected={expected:<11} got={predicted:<11} match={score}"
        )

    elapsed = time.time() - started
    total = len(claims)
    accuracy = correct / total * 100 if total else 0

    print(f"\n{'=' * 62}")
    print(f"Per-claim grounding accuracy : {correct}/{total}  ({accuracy:.1f}%)")
    print(f"Elapsed                      : {elapsed:.1f}s "
          f"({elapsed / total:.1f}s per claim)")

    print(f"\nConfusion matrix (rows = expected, columns = predicted)")
    print(f"{'':<13}" + "".join(f"{label:<13}" for label in LABELS))

    for expected in LABELS:
        row = "".join(
            f"{confusion[expected][predicted]:<13}"
            for predicted in LABELS
        )
        print(f"{expected:<13}{row}")

    if failures:
        print(f"\nMisclassified ({len(failures)}):")
        for claim, predicted in failures:
            print(f"  [{claim['id']}] expected {claim['expected']}, "
                  f"got {predicted}")
            print(f"       \"{claim['text']}\"")

    print(f"{'=' * 62}")

    return 0 if accuracy >= 80 else 1


if __name__ == "__main__":
    sys.exit(main())
