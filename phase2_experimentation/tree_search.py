"""
Phase 2 — Agentic tree search over experiment nodes.
Mirrors the AI Scientist's 4-stage structure adapted for PCI outcomes research:
  Stage 1: Baseline model (unadjusted KM + univariate Cox)
  Stage 2: Hyperparameter / model selection (penalised Cox LASSO, variable selection)
  Stage 3: Full adjusted analysis (multivariable Cox + subgroup forest plot)
  Stage 4: Ablation / sensitivity studies (competing risks, propensity matching)
"""

import os, sys, json, time

WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORK_DIR)
from phase2_experimentation.experiment_node import ExperimentNode
from claude_client import claude_call

EVALUATOR_PROMPT = """
You are a senior cardiologist reviewing statistical analysis results.
Given these experiment node results, rank them from best to worst and
return the ID of the BEST node. Consider:
- Statistical validity (model assumptions met, n events adequate)
- Clinical meaningfulness (effect sizes, confidence intervals)
- Completeness (did the node produce figures and a full results summary?)

Nodes:
{nodes_json}

Return ONLY a JSON object: {{"best_id": "...", "reason": "..."}}
"""


class AgenticTreeSearch:
    def __init__(self, idea: dict, max_nodes_per_stage: int = 3,
                 max_debug_retries: int = 4):
        self.idea      = idea
        self.max_nodes = max_nodes_per_stage
        self.max_debug = max_debug_retries
        self.tree: list[ExperimentNode] = []
        self.log_path  = os.path.join(WORK_DIR, "phase2_experimentation",
                                      "logs", "tree_log.jsonl")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _run_node(self, node: ExperimentNode) -> ExperimentNode:
        node.generate_script()
        success = node.execute()
        if not success:
            success = node.debug(self.max_debug)
        self._log(node)
        return node

    def _log(self, node: ExperimentNode):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(node.to_dict(), ensure_ascii=False) + "\n")

    def _select_best(self, nodes: list[ExperimentNode]) -> ExperimentNode:
        non_buggy = [n for n in nodes if n.status == "non-buggy"]
        if not non_buggy:
            return nodes[0]
        if len(non_buggy) == 1:
            return non_buggy[0]

        nodes_json = json.dumps([n.to_dict() for n in non_buggy],
                                ensure_ascii=False, indent=2)
        raw = claude_call(EVALUATOR_PROMPT.format(nodes_json=nodes_json), max_tokens=512)
        try:
            result = json.loads(raw)
            best_id = result["best_id"]
            best = next((n for n in non_buggy if n.id == best_id), non_buggy[0])
            print(f"  [Evaluator] Best node: {best.id} — {result.get('reason','')}")
            return best
        except Exception:
            return non_buggy[0]

    def run(self) -> dict:
        """Run all 4 stages, returning the best node from each."""
        stage_configs = [
            {
                "stage": 1,
                "label": "Baseline (unadjusted KM + univariate Cox)",
                "method_override": "Kaplan-Meier survival curve + log-rank test + univariate Cox PH",
                "n_nodes": self.max_nodes,
            },
            {
                "stage": 2,
                "label": "Model selection (LASSO penalised Cox)",
                "method_override": "LASSO-penalised Cox PH with cross-validated lambda selection",
                "n_nodes": self.max_nodes,
            },
            {
                "stage": 3,
                "label": "Full adjusted analysis",
                "method_override": (
                    "Multivariable Cox PH with clinical confounder adjustment, "
                    "proportional hazards assumption test (Schoenfeld residuals), "
                    "forest plot of HRs"
                ),
                "n_nodes": self.max_nodes,
            },
            {
                "stage": 4,
                "label": "Sensitivity / ablation",
                "method_override": (
                    "Competing risks analysis (Fine-Gray), propensity score matching, "
                    "subgroup analysis (diabetes, ACS, multivessel disease)"
                ),
                "n_nodes": self.max_nodes,
            },
        ]

        best_nodes = {}
        parent_id  = None

        for cfg in stage_configs:
            print(f"\n{'='*60}")
            print(f"STAGE {cfg['stage']}: {cfg['label']}")
            print(f"{'='*60}")

            stage_idea = dict(self.idea)
            stage_idea["statistical_method"] = cfg["method_override"]

            nodes = []
            for i in range(cfg["n_nodes"]):
                node = ExperimentNode(stage_idea, cfg["stage"], parent_id)
                node = self._run_node(node)
                self.tree.append(node)
                nodes.append(node)
                print(f"  Node {node.id}: {node.status} | metrics: {node.metrics}")

            best = self._select_best(nodes)
            best_nodes[cfg["stage"]] = best
            parent_id = best.id
            print(f"  -> Best for stage {cfg['stage']}: {best.id}")

        summary = {
            "idea": self.idea,
            "best_nodes": {k: v.to_dict() for k, v in best_nodes.items()},
            "total_nodes": len(self.tree),
            "non_buggy": sum(1 for n in self.tree if n.status == "non-buggy"),
        }
        summary_path = os.path.join(WORK_DIR, "phase2_experimentation",
                                    "logs", "search_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n[Tree Search] Done. {summary['non_buggy']}/{summary['total_nodes']} nodes succeeded.")
        return summary
