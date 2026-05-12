"""
Phase 3 — Manuscript writer.
Reads experiment results and figures from the best nodes of each stage,
generates a structured manuscript (STROBE-compliant for observational studies),
and saves it as a Markdown file with figures.
"""

import os, sys, json, glob, shutil

WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORK_DIR)
from claude_client import claude_call

SECTION_PROMPTS = {
    "abstract": """
Write a structured abstract (Background, Methods, Results, Conclusions) for a
peer-reviewed biomedical research paper. Keep it under 300 words.
Do not fabricate numbers — use only what appears in the data below.

{context}
""",

    "introduction": """
Write a 3-paragraph Introduction for a biomedical research paper.
Do NOT fabricate citations — reference general domain knowledge only.

Para 1: Clinical or scientific burden and context of the research question.
Para 2: Gap in the current literature that this study addresses.
Para 3: The aim of this study, briefly describing the dataset and primary question.

Study title: {title}
Hypothesis: {hypothesis}
Study design: {design}
""",

    "methods": """
Write a Methods section for an observational cohort study.
Follow STROBE reporting guidelines. Cover:
- Study design and setting
- Participants (inclusion and exclusion criteria)
- Outcome definition
- Key predictors and covariates: {predictors}
- Statistical analysis approach: {method}
- Software used (Python: lifelines, statsmodels, pandas; or R as appropriate)
- Ethics statement (institutional approval, consent waiver for registry/administrative data)

Be concise and precise. Do not invent numbers or institutional details.
""",

    "results": """
Write a Results section based on the statistical outputs below.
Structure it as:
1. Study population (sample size, follow-up, outcome frequency)
2. Primary outcome — unadjusted and adjusted results
3. Key secondary or subgroup findings
4. Sensitivity analyses if available

RESULTS DATA:
{results}

Figures available: {figures}
Reference figures as Figure 1, Figure 2, etc. in order.
Do NOT fabricate numbers — use only what appears in RESULTS DATA.
""",

    "discussion": """
Write a Discussion section (5 paragraphs).
Do NOT fabricate citations — reference general biomedical knowledge only.

Para 1: Summary of the principal finding and how it relates to the hypothesis.
Para 2: Comparison with prior literature on this topic.
Para 3: Biological or clinical mechanisms that could explain the finding.
Para 4: Subgroup or secondary findings and their implications.
Para 5: Limitations (observational design, potential confounding, missing data,
  single-centre or single-database, lack of randomisation) and clinical implications.

Principal finding: {main_finding}
Study design: {design}
""",
}


def collect_results(search_summary: dict) -> tuple:
    results_text = ""
    figure_paths = []

    for stage, node_dict in search_summary.get("best_nodes", {}).items():
        node_dir = node_dict.get("node_dir", "")

        results_file = os.path.join(node_dir, "results.txt")
        if os.path.exists(results_file):
            with open(results_file, encoding="utf-8") as f:
                results_text += f"\n--- Stage {stage} results ---\n" + f.read()

        metrics_file = os.path.join(node_dir, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                results_text += f"\nMetrics (stage {stage}): {f.read()}\n"

        figs = sorted(glob.glob(os.path.join(node_dir, "figures", "*.png")))
        figure_paths.extend(figs)

    return results_text, figure_paths


def generate_manuscript(search_summary: dict) -> str:
    idea    = search_summary["idea"]
    results, figures = collect_results(search_summary)

    print("[Phase 3] Generating manuscript sections...")

    design = idea.get("design", "Retrospective observational cohort study")

    sections = {}

    sections["methods"] = claude_call(SECTION_PROMPTS["methods"].format(
        predictors=", ".join(idea.get("key_predictors", [])),
        method=idea.get("statistical_method", "appropriate statistical methods"),
    ), max_tokens=2048)

    sections["results"] = claude_call(SECTION_PROMPTS["results"].format(
        results=results[:6000],
        figures="\n".join(os.path.basename(f) for f in figures) or "None",
    ), max_tokens=2048)

    sections["introduction"] = claude_call(SECTION_PROMPTS["introduction"].format(
        title=idea["title"],
        hypothesis=idea["hypothesis"],
        design=design,
    ), max_tokens=1024)

    main_finding = results[:500] if results else "See results section."
    sections["discussion"] = claude_call(SECTION_PROMPTS["discussion"].format(
        main_finding=main_finding,
        design=design,
    ), max_tokens=2048)

    abstract_context = (
        f"Title: {idea['title']}\n"
        f"Design: {design}\n"
        f"Methods summary: {sections['methods'][:400]}\n"
        f"Results summary: {sections['results'][:400]}\n"
    )
    sections["abstract"] = claude_call(SECTION_PROMPTS["abstract"].format(
        context=abstract_context,
    ), max_tokens=512)

    manuscript = f"""# {idea['title']}

## Abstract
{sections['abstract']}

---

## Introduction
{sections['introduction']}

## Methods
{sections['methods']}

## Results
{sections['results']}

## Discussion
{sections['discussion']}

---
*Generated by AI Scientist | Study design: {design}*
*Figures: {', '.join(os.path.basename(f) for f in figures) or 'None generated'}*
""".strip()

    out_dir = os.path.join(WORK_DIR, "phase3_writeup")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "manuscript.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(manuscript)

    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    for fig in figures:
        shutil.copy2(fig, fig_dir)

    print(f"[Phase 3] Manuscript saved: {out_path}")
    print(f"[Phase 3] {len(figures)} figures copied.")
    return manuscript
