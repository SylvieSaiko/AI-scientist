"""
Phase 4 — Automated Reviewer.
Ensemble of N independent LLM reviews followed by a meta-review (area chair).
Criteria are calibrated for observational biomedical research across any domain.
"""

import os, sys, json

WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORK_DIR)
from claude_client import claude_call

REVIEWER_SYSTEM = (
    "You are a senior biomedical researcher and biostatistician reviewing a manuscript "
    "for a peer-reviewed medical or public health journal. Be rigorous and constructive."
)

REVIEWER_PROMPT = """
Review the following biomedical research manuscript. Score each dimension 1–10 (10 = best).
Return a JSON object with these exact keys:

  soundness          – statistical rigour, correct methods, assumption checks, sample size
  clinical_relevance – importance and novelty of the research question
  presentation       – clarity, figure quality, table completeness, writing
  contribution       – advance over existing literature
  overall            – weighted overall score (0.35*soundness + 0.30*clinical_relevance + 0.20*presentation + 0.15*contribution)
  accept             – true if overall >= 6.5, false otherwise
  strengths          – list of 3 specific strengths
  weaknesses         – list of 3 specific weaknesses
  major_concerns     – list of critical issues that must be addressed before publication
  minor_concerns     – list of minor suggestions
  reviewer_confidence – integer 1 (low) to 5 (high)

Criteria checklist for observational biomedical research:
1. Is the outcome clearly defined and ascertained consistently?
2. Is the study design appropriate for the hypothesis (cohort, case-control, cross-sectional)?
3. For time-to-event data: is censoring handled correctly? Competing risks addressed?
4. For regression: are assumptions tested (PH, linearity, homoscedasticity)?
5. Are confounders adequately controlled? Is indication/selection bias discussed?
6. Is causal language appropriately avoided for observational data?
7. Is missing data handled and described transparently?
8. Does the study follow STROBE (observational) or CONSORT (trial) guidelines?
9. Are subgroup analyses pre-specified or clearly labelled as exploratory?
10. Is the sample size adequate for the number of predictors (events-per-variable rule)?

MANUSCRIPT:
{manuscript}

Return ONLY valid JSON. No prose outside the JSON object.
"""

META_REVIEW_PROMPT = """
You are an area editor at a peer-reviewed biomedical journal.
Below are {n} independent reviews of the same manuscript.
Synthesise them into a final editorial decision.

Return JSON with these exact keys:
  meta_score            – float 1–10 (average of individual overall scores)
  final_decision        – "accept" | "minor_revision" | "major_revision" | "reject"
  consensus_strengths   – top 3 agreed-upon strengths across reviewers
  consensus_weaknesses  – top 3 agreed-upon weaknesses across reviewers
  required_revisions    – list of changes the authors MUST make
  summary               – 2-sentence editorial summary

REVIEWS:
{reviews_json}

Return ONLY valid JSON.
"""


def _strip_code_fences(text: str) -> str:
    lines = text.strip().split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def run_single_review(manuscript: str, reviewer_idx: int) -> dict:
    print(f"  [Reviewer {reviewer_idx + 1}] Reviewing...")
    prompt = (f"{REVIEWER_SYSTEM}\n\n"
              f"{REVIEWER_PROMPT.format(manuscript=manuscript[:12000])}")
    raw = _strip_code_fences(claude_call(prompt, max_tokens=2048))
    try:
        review = json.loads(raw)
        review["reviewer_id"] = reviewer_idx + 1
        return review
    except json.JSONDecodeError:
        return {"reviewer_id": reviewer_idx + 1, "raw": raw, "parse_error": True}


def run_meta_review(reviews: list) -> dict:
    print("  [Meta-Reviewer] Synthesising reviews...")
    prompt = (f"{REVIEWER_SYSTEM}\n\n"
              + META_REVIEW_PROMPT.format(
                  n=len(reviews),
                  reviews_json=json.dumps(reviews, ensure_ascii=False, indent=2)[:8000],
              ))
    raw = _strip_code_fences(claude_call(prompt, max_tokens=1024))
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

    print(f"\n[Phase 4] Running {n_reviewers} independent reviews...")
    reviews = [run_single_review(manuscript_text, i) for i in range(n_reviewers)]

    meta = run_meta_review(reviews)

    valid = [r for r in reviews if "overall" in r and not r.get("parse_error")]
    avg_score = sum(r["overall"] for r in valid) / len(valid) if valid else None

    result = {
        "individual_reviews": reviews,
        "meta_review":        meta,
        "avg_overall_score":  round(avg_score, 2) if avg_score else None,
        "n_accept":           sum(1 for r in valid if r.get("accept", False)),
        "n_reviewers":        n_reviewers,
    }

    out_path = os.path.join(WORK_DIR, "phase4_review", "review_report.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n[Phase 4] Review complete.")
    if avg_score:
        print(f"  Average score:  {avg_score:.1f}/10")
    print(f"  Accept votes:   {result['n_accept']}/{n_reviewers}")
    print(f"  Final decision: {meta.get('final_decision', 'unknown')}")
    print(f"  Report saved:   {out_path}")
    return result
