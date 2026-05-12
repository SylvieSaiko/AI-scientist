"""
Phase 1 — Ideation
Uses Claude to generate ranked research ideas from a dataset schema,
checks novelty via PubMed, and saves the idea archive.
"""

import json, os, sys, time, re
import urllib.request
import urllib.parse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from claude_client import claude_call

WORK_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDEA_ARCHIVE = os.path.join(WORK_DIR, "phase1_ideation", "idea_archive.json")

# ── Replace this with a description of YOUR dataset ───────────────────────────
DATASET_SUMMARY = """
<Replace with a brief description of your dataset>

Example fields to describe:
- Sample size and time span
- Variable groups (demographics, labs, procedures, medications, outcomes)
- Outcome definitions and follow-up windows
- Any special features (longitudinal, multi-site, registry, trial, etc.)
"""
# ──────────────────────────────────────────────────────────────────────────────

IDEA_PROMPT = """
You are an expert clinical researcher.
Below is a description of a dataset. Generate {n} original, high-impact research
ideas that can be fully answered using this dataset alone.

Each idea must be:
- Clinically meaningful and publishable in a top journal
- Statistically feasible (specify the exact statistical method)
- Novel relative to the mainstream literature

Return a JSON array. Each element must have these exact keys:
  title            – concise paper title
  hypothesis       – one sentence stating the testable hypothesis
  design           – study design (retrospective cohort / landmark / propensity / etc.)
  primary_endpoint – exact outcome variable to use
  time_horizon     – follow-up window
  statistical_method – Cox PH / logistic / competing risks / LASSO / etc.
  key_predictors   – list of 3-6 variable names from the dataset
  interestingness  – integer 1-10
  novelty          – integer 1-10
  feasibility      – integer 1-10
  pubmed_query     – a PubMed search string to check if this is already published

DATASET DESCRIPTION:
{dataset}

Return ONLY the JSON array, no prose.
"""


def query_pubmed(search_string: str, max_results: int = 5) -> dict:
    base  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    query = urllib.parse.quote(search_string)
    url   = f"{base}esearch.fcgi?db=pubmed&term={query}&retmax={max_results}&retmode=json"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())
        return {"count": int(data["esearchresult"]["count"]),
                "ids":   data["esearchresult"]["idlist"]}
    except Exception as e:
        return {"count": -1, "ids": [], "error": str(e)}


def check_novelty(idea: dict) -> dict:
    result = query_pubmed(idea.get("pubmed_query", idea["title"]))
    idea["pubmed_hits"] = result["count"]
    idea["novel"]       = result["count"] < 20
    return idea


def generate_ideas(n: int = 10, model: str = "claude-sonnet-4-6") -> list:
    prompt = IDEA_PROMPT.format(n=n, dataset=DATASET_SUMMARY)
    print(f"[Phase 1] Calling {model} to generate {n} ideas...")
    raw = claude_call(prompt, model=model, max_tokens=4096)

    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        raw = match.group(0)
    ideas = json.loads(raw)

    print("[Phase 1] Checking novelty via PubMed...")
    for idea in ideas:
        idea = check_novelty(idea)
        idea["composite_score"] = (
            idea.get("interestingness", 5) * 0.4 +
            idea.get("novelty", 5) * 0.4 +
            idea.get("feasibility", 5) * 0.2
        )
        time.sleep(0.4)

    ideas.sort(key=lambda x: x["composite_score"], reverse=True)

    os.makedirs(os.path.dirname(IDEA_ARCHIVE), exist_ok=True)
    archive = []
    if os.path.exists(IDEA_ARCHIVE):
        with open(IDEA_ARCHIVE) as f:
            archive = json.load(f)
    for idea in ideas:
        idea["generated_at"] = datetime.now().isoformat()
        archive.append(idea)
    with open(IDEA_ARCHIVE, "w", encoding="utf-8") as f:
        json.dump(archive, f, ensure_ascii=False, indent=2)

    print(f"[Phase 1] Saved {len(ideas)} ideas. Archive total: {len(archive)}")
    return ideas


def select_best_idea(min_novelty_hits: int = 20) -> dict:
    if not os.path.exists(IDEA_ARCHIVE):
        raise FileNotFoundError("Run generate_ideas() first.")
    with open(IDEA_ARCHIVE) as f:
        archive = json.load(f)
    candidates = [i for i in archive if i.get("pubmed_hits", 999) < min_novelty_hits]
    if not candidates:
        candidates = archive
    return max(candidates, key=lambda x: x.get("composite_score", 0))


if __name__ == "__main__":
    ideas = generate_ideas(n=10)
    print("\nTop 3 ideas:")
    for idea in ideas[:3]:
        print(f"  [{idea['composite_score']:.1f}] {idea['title']}")
        print(f"         PubMed hits: {idea['pubmed_hits']}")
