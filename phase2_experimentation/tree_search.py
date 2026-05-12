"""
Phase 2 — Agentic tree search over experiment nodes.

Mirrors the AI Scientist's 4-stage structure, adapted for biomedical research:
  Stage 1: Exploratory analysis — descriptive stats + univariate association screening
  Stage 2: Variable selection  — penalised regression to identify key predictors
  Stage 3: Primary analysis    — confirmatory multivariable model + subgroup analysis
  Stage 4: Sensitivity/ablation — robustness checks, alternative specs, temporal trends

At each stage, N candidate nodes are generated with different analytical angles.
The best-scoring node (by ExperimentNode.score()) seeds the next stage.
"""

import os, sys, json

WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORK_DIR)
from phase2_experimentation.experiment_node import ExperimentNode
from claude_client import claude_call

# ── Stage goals: what each stage should accomplish ────────────────────────────
# Each stage has a primary goal and N analytical variants to try.
# Variants steer Claude toward different analytical angles while staying on-goal.
STAGE_CONFIGS = [
    {
        "stage": 1,
        "goal": (
            "Exploratory analysis: describe the study population, "
            "characterise the outcome, and screen all key predictors with "
            "univariate associations (unadjusted). Produce descriptive plots."
        ),
        "variants": [
            "Focus on the primary exposure vs outcome: produce a survival curve "
            "(if time-to-event) or proportion plot (if binary) and a univariate "
            "association test (log-rank, chi-square, or t-test as appropriate).",
            "Screen ALL key predictors univariately: produce a ranked forest plot "
            "of effect sizes (HR, OR, or β) with 95% CIs for every predictor.",
            "Characterise missing data and distributional properties: missingness "
            "map, histograms for continuous variables, bar charts for categorical; "
            "flag variables with >20% missingness.",
        ],
    },
    {
        "stage": 2,
        "goal": (
            "Variable selection: identify which predictors carry independent signal "
            "using a penalised regression approach. Return a ranked list of selected "
            "predictors for use in the primary model."
        ),
        "variants": [
            "LASSO (L1) penalised regression with cross-validated lambda selection. "
            "Report selected predictors and their shrunk coefficients.",
            "Elastic net (L1+L2, alpha=0.5) with cross-validated lambda selection. "
            "Compare selected predictors to LASSO.",
            "Stepwise selection by AIC: start with all candidates, iteratively drop "
            "the least significant predictor until all remaining are p<0.10.",
        ],
    },
    {
        "stage": 3,
        "goal": (
            "Primary confirmatory analysis: fit a multivariable model using the "
            "predictors identified in Stage 2, test model assumptions, produce a "
            "forest plot of adjusted effect sizes, and run pre-specified subgroup analyses."
        ),
        "variants": [
            "Full multivariable model with all Stage-2 selected predictors plus the "
            "primary exposure. Test assumptions (PH test for survival, residuals for "
            "regression). Produce forest plot and subgroup analysis.",
            "Parsimonious model: primary exposure + age + sex + top 3 predictors by "
            "univariate effect size only. Prioritise interpretability.",
            "Interaction model: fit the full model plus a primary-exposure × key "
            "effect-modifier interaction term. Report interaction p-value and "
            "stratum-specific estimates.",
        ],
    },
    {
        "stage": 4,
        "goal": (
            "Sensitivity and ablation analyses: assess robustness of the primary "
            "finding under alternative analytical choices, subpopulations, or "
            "time periods."
        ),
        "variants": [
            "Temporal trend analysis: stratify by time period (early / mid / late), "
            "plot annual event rates or outcome proportions, run era-stratified models.",
            "Covariate sensitivity: refit the primary model (a) with all available "
            "variables regardless of Stage-2 selection, (b) restricted to complete "
            "cases with all covariates, (c) excluding outliers.",
            "Alternative outcome or competing risks: refit using a secondary endpoint "
            "or, for survival data, a Fine-Gray competing risks model if a competing "
            "event column is available.",
        ],
    },
]

EVALUATOR_PROMPT = """
You are a senior biomedical researcher reviewing analysis results.
Given these experiment node outputs, identify the BEST node. Consider:
- Statistical validity (adequate sample size, assumptions tested, no obvious errors)
- Clinical or scientific meaningfulness (effect sizes, confidence intervals, interpretability)
- Completeness (figures produced, full metrics reported)

Nodes:
{nodes_json}

Return ONLY a JSON object: {{"best_id": "...", "reason": "..."}}
"""


class AgenticTreeSearch:
    def __init__(self, idea: dict, max_nodes_per_stage: int = 3,
                 max_debug_retries: int = 3):
        self.idea       = idea
        self.max_nodes  = min(max_nodes_per_stage, 3)  # cap at 3 variants
        self.max_debug  = max_debug_retries
        self.tree: list = []
        self.log_path   = os.path.join(WORK_DIR, "phase2_experimentation",
                                       "logs", "tree_log.jsonl")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _run_node(self, node: ExperimentNode,
                  stage_goal: str, parent_metrics: dict) -> ExperimentNode:
        node.generate_script(stage_goal=stage_goal, parent_metrics=parent_metrics)
        success = node.execute()
        if not success:
            node.debug(self.max_debug)
        self._log(node)
        return node

    def _log(self, node: ExperimentNode):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(node.to_dict(), ensure_ascii=False) + "\n")

    def _select_best(self, nodes: list) -> ExperimentNode:
        non_buggy = [n for n in nodes if n.status == "non-buggy"]
        if not non_buggy:
            return nodes[0]
        if len(non_buggy) == 1:
            return non_buggy[0]

        # Primary: use numeric score
        by_score = sorted(non_buggy, key=lambda n: n.score(), reverse=True)
        top_score = by_score[0].score()

        # If scores are tied (e.g. all 0.0 — no metrics), fall back to Claude evaluator
        if top_score == 0.0:
            nodes_json = json.dumps([n.to_dict() for n in non_buggy],
                                    ensure_ascii=False, indent=2)
            raw = claude_call(EVALUATOR_PROMPT.format(nodes_json=nodes_json),
                              max_tokens=256)
            try:
                result = json.loads(raw)
                best = next((n for n in non_buggy if n.id == result["best_id"]),
                            non_buggy[0])
                print(f"  [Evaluator] Best: {best.id} — {result.get('reason', '')}")
                return best
            except Exception:
                return non_buggy[0]

        print(f"  [Score] Best: {by_score[0].id}  score={top_score:.3f}")
        return by_score[0]

    def run(self) -> dict:
        best_parent_metrics = None
        parent_id           = None
        best_nodes          = {}

        for cfg in STAGE_CONFIGS:
            stage = cfg["stage"]
            print(f"\n{'='*60}")
            print(f"STAGE {stage}: {cfg['goal'][:60]}...")
            print(f"{'='*60}")

            nodes = []
            for i in range(self.max_nodes):
                variant_hint = cfg["variants"][i] if i < len(cfg["variants"]) else ""
                stage_goal   = f"{cfg['goal']}\n\nSpecific approach: {variant_hint}"

                stage_idea = dict(self.idea)
                # Preserve the idea's method but steer the analytical angle
                stage_idea["statistical_method"] = (
                    f"{self.idea.get('statistical_method', 'appropriate method')} — "
                    f"{variant_hint}"
                )

                node = ExperimentNode(stage_idea, stage, parent_id)
                node = self._run_node(node, stage_goal, best_parent_metrics or {})
                self.tree.append(node)
                nodes.append(node)
                print(f"  Node {node.id}: {node.status}  score={node.score():.3f}  "
                      f"metrics={node.metrics}")

            best           = self._select_best(nodes)
            best_nodes[stage] = best
            parent_id      = best.id
            best_parent_metrics = best.metrics
            print(f"  → Best for stage {stage}: {best.id}  score={best.score():.3f}")

        summary = {
            "idea":        self.idea,
            "best_nodes":  {k: v.to_dict() for k, v in best_nodes.items()},
            "total_nodes": len(self.tree),
            "non_buggy":   sum(1 for n in self.tree if n.status == "non-buggy"),
        }
        summary_path = os.path.join(WORK_DIR, "phase2_experimentation",
                                    "logs", "search_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n[Tree Search] Complete — "
              f"{summary['non_buggy']}/{summary['total_nodes']} nodes succeeded.")
        return summary
