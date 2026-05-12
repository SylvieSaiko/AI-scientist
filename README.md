# AI Scientist — Automated Clinical Research Pipeline

An end-to-end pipeline that generates, executes, and writes up clinical research studies from a tabular dataset — powered by Claude.

Given a dataset and a brief description, the system autonomously:
1. Generates ranked research ideas and checks novelty via PubMed
2. Runs a 4-stage statistical analysis (survival analysis, variable selection, multivariable modelling, temporal trends)
3. Writes a full manuscript and assembles it as a Word document

---

## Pipeline Overview

```
Phase 1 — Ideation
  └── idea_generator.py           Generate & rank research ideas via Claude;
                                  check novelty on PubMed

Phase 2 — Experimentation (agentic tree search)
  └── tree_search.py              4-stage search; N candidate nodes per stage;
                                  best-scoring node seeds the next stage

      Stage 1  Exploratory        Descriptive stats + univariate association screening
      Stage 2  Variable selection Penalised regression (LASSO / elastic net / stepwise)
      Stage 3  Primary analysis   Multivariable model + assumption checks + subgroups
      Stage 4  Sensitivity        Robustness checks, temporal trends, alternative specs

  └── experiment_node.py          Generates analysis code via Claude, executes it,
                                  scores the output, auto-debugs on failure

Phase 3 — Write-up
  └── manuscript_writer.py        Generates manuscript sections via Claude;
                                  saves Markdown + figures

Phase 4 — Automated peer review
  └── automated_reviewer.py       Ensemble of N independent LLM reviews +
                                  meta-review; returns accept/revise/reject
```

Each node saves `metrics.json`, `results.txt`, and figures to its own directory.
The analysis method is determined by the study idea — survival, logistic, or linear
regression — not hardcoded. Example templates for survival analysis are in
`phase2_experimentation/nodes/stage1–4/`.

---

## Requirements

```bash
pip install lifelines pandas numpy matplotlib python-docx statsmodels
```

Claude Code CLI must be installed and authenticated (`claude` binary in PATH).  
See: https://docs.anthropic.com/claude-code

---

## Setup

**1. Add your dataset**

Place your dataset at `data/dataset.csv` (excluded from git).

**2. Configure the pipeline**

Edit `config.py`:
```python
WORK_DIR  = "/path/to/AI_Scientist"
DATA_PATH = "/path/to/your_dataset"
ENDPOINTS = {
    "primary":   ("event_col", "time_col"),
    "secondary": ("secondary_event", "secondary_time"),
}
```

**3. Describe your dataset**

Edit the `DATASET_SUMMARY` string in `phase1_ideation/idea_generator.py` to describe your data (sample size, variables, outcome definitions).

**4. Configure each stage template**

In each `phase2_experimentation/nodes/stageN/analysis.py`, replace the placeholder column names (`event_col`, `time_col`, `exposure_col`, etc.) with your actual column names.

---

## Running

```bash
# Full automated pipeline
python run_pipeline.py

# Or run phases individually
python phase1_ideation/idea_generator.py
python phase2_experimentation/nodes/stage1/analysis.py
python phase2_experimentation/nodes/stage2/analysis.py
python phase2_experimentation/nodes/stage3/analysis.py
python phase2_experimentation/nodes/stage4/analysis.py
python build_manuscript.py
```

---

## Output

- `phase2_experimentation/nodes/stageN/metrics.json` — key statistics
- `phase2_experimentation/nodes/stageN/figures/` — plots (KM curves, forest plots, temporal trends)
- `phase3_writeup/Manuscript.docx` — full Word manuscript

---

## Notes

- Time column must be in days from baseline. Non-events with `TIME=0` should be censored at the study horizon (see comments in each stage template).
- Binary columns coded 1/2 (not 0/1) should be recoded: `df[col] = df[col] - 1`
- NT-proBNP and similar skewed lab values should be log-transformed before modelling.
- Complete-case analysis is used by default; adjust `dropna()` calls as needed.

---

## License

MIT — see [LICENSE](LICENSE)
