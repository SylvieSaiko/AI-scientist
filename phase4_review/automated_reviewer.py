"""
Phase 4 — Automated Reviewer.
Ensemble of 5 independent LLM reviews with a meta-review (area chair).
Cardiology-specific criteria: endpoint validity, assumption checking,
confounding, causal language, missing data.
"""

import os, sys, json

WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORK_DIR)
from claude_client import claude_call

REVIEWER_SYSTEM = (
    "You are a senior cardiologist and biostatistician reviewing a manuscript "
    "for the Journal of the American College of Cardiology (JACC) or "
    "European Heart Journal. Be rigorous and constructive."
)

REVIEWER_PROMPT = """
Review the following cardiology manuscript. Score each dimension 1-10 (10=best).
Return a JSON object with these exact keys:

  soundness       – statistical rigour, correct methods, assumption checks
  clinical_relevance – importance and novelty of the clinical question
  presentation    – clarity, figure quality, table completeness
  contribution    – advance over existing PCI literature
  overall         – weighted overall score
  accept          – true/false (accept if overall >= 6.5)
  strengths       – list of 3 strengths
  weaknesses      – list of 3 weaknesses
  major_concerns  – list of critical issues that must be addressed
  minor_concerns  – list of minor suggestions
  reviewer_confidence – 1 (low) to 5 (high)

Cardiology-specific criteria to check:
1. Is the endpoint (MACE / death) clearly defined and adjudicated?
2. Are time-to-event data handled correctly (censoring, competing risks)?
3. Is the proportional hazards assumption tested (Schoenfeld residuals)?
4. Are confounders adequately controlled (indication bias, selection bias)?
5. Is causal language avoided appropriately for an observational study?
6. Is missing data handling described and appropriate?
7. Is the STROBE checklist followed?
8. Are subgroup analyses pre-specified or clearly labelled exploratory?
9. Are absolute risk differences reported alongside HRs?
10. Is the sample size adequate for the number of predictors?

MANUSCRIPT:
{manuscript}

Return ONLY valid JSON.
"""

META_REVIEW_PROMPT = """
You are an area chair at a top cardiology journal.
Below are {n} independent reviews of the same manuscript.
Synthesise them into a final decision and overall meta-score.

Return JSON:
  meta_score      – float 1-10
  final_decision  – "accept" | "major_revision" | "minor_revision" | "reject"
  consensus_strengths   – top 3 agreed-upon strengths
  consensus_weaknesses  – top 3 agreed-upon weaknesses
  required_revisions    – list of changes the authors MUST make
  summary         – 2-sentence summary of the decision

REVIEWS:
{reviews_json}

Return ONLY valid JSON.
"""


def run_single_review(manuscript: str, reviewer_idx: int) -> dict:
    print(f"  [Reviewer {reviewer_idx + 1}] Reviewing...")
    prompt = f"{REVIEWER_SYSTEM}\n\n{REVIEWER_PROMPT.format(manuscript=manuscript[:12000])}"
    raw = claude_call(prompt, max_tokens=2048)
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
    if raw.endswith("```"):
        raw = "\n".join(raw.split("\n")[:-1])
    try:
        review = json.loads(raw)
        review["reviewer_id"] = reviewer_idx + 1
        return review
    except json.JSONDecodeError:
        return {"reviewer_id": reviewer_idx + 1, "raw": raw, "parse_error": True}


def run_meta_review(reviews: list[dict]) -> dict:
    print("  [Meta-Reviewer] Synthesising reviews...")
    meta_body = META_REVIEW_PROMPT.format(
        n=len(reviews),
        reviews_json=json.dumps(reviews, ensure_ascii=False, indent=2)[:8000],
    )
    prompt = REVIEWER_SYSTEM + "\n\n" + meta_body
    raw = claude_call(prompt, max_tokens=1024)
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
    if raw.endswith("```"):
        raw = "\n".join(raw.split("\n")[:-1])
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw, "parse_error": True}


def review_manuscript(manuscript_path: str = None,
                      manuscript_text: str = None,
                      n_reviewers: int = 5) -> dict:
    if manuscript_text is None:
        if manuscript_path is None:
            manuscript_path = os.path.join(
                WORK_DIR, "phase3_writeup", "manuscript.md"
            )
        with open(manuscript_path, encoding="utf-8") as f:
            manuscript_text = f.read()

    reviews = []

    print(f"\n[Phase 4] Running {n_reviewers} independent reviews...")
    for i in range(n_reviewers):
        review = run_single_review(manuscript_text, i)
        reviews.append(review)

    meta = run_meta_review(reviews)

    # Compute average scores
    valid_reviews = [r for r in reviews if "overall" in r]
    avg_score = (sum(r["overall"] for r in valid_reviews) / len(valid_reviews)
                 if valid_reviews else None)

    result = {
        "individual_reviews": reviews,
        "meta_review": meta,
        "avg_overall_score": avg_score,
        "n_accept": sum(1 for r in valid_reviews if r.get("accept", False)),
        "n_reviewers": n_reviewers,
    }

    out_path = os.path.join(WORK_DIR, "phase4_review", "review_report.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n[Phase 4] Review complete.")
    print(f"  Average score: {avg_score:.1f}/10" if avg_score else "  Score unavailable")
    print(f"  Accept votes: {result['n_accept']}/{n_reviewers}")
    print(f"  Final decision: {meta.get('final_decision', 'unknown')}")
    print(f"  Report saved to: {out_path}")
    return result
