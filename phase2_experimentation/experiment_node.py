"""
Phase 2 — Experiment node.
Each node is one statistical analysis. The agent generates Python code,
executes it in a subprocess, captures metrics + figures, and marks status.
"""

import os, sys, json, traceback, uuid, subprocess

WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORK_DIR)
from claude_client import claude_call

NODE_PROMPT = """IMPORTANT: Respond with ONLY a Python code block. Do NOT use any file tools. Do NOT write files. Just output the code.

You are an expert biostatistician. Produce a self-contained Python analysis script inside a single ```python ... ``` block.

DATASET: CSV at: {csv_path}
STUDY: {title}
HYPOTHESIS: {hypothesis}
ENDPOINT: event column="{endpoint}", time column="{time_col}" (days from baseline)
METHOD: {method}
KEY PREDICTORS: {predictors}
TIME HORIZON: {horizon}
OUTPUT DIRECTORY: {node_dir}

The script must:
1. Load the CSV with pd.read_csv(path, low_memory=False)
2. Coerce relevant columns to numeric: pd.to_numeric(df[col], errors='coerce')
3. Drop rows missing the endpoint or time column; clip time to 0.5 minimum
4. Run the specified statistical method (lifelines for survival, statsmodels for regression)
5. Save to {node_dir}:
   - metrics.json  — keys: n, events, c_index (or auc), main_hr, hr_ci_low, hr_ci_high, p_value
   - figures/      — at least one plot at 150 dpi
   - results.txt   — plain text summary
6. Print STATUS: SUCCESS on the last line, or STATUS: FAILED if an exception is caught

Notes:
- matplotlib non-interactive backend: import matplotlib; matplotlib.use('Agg')
- Binary columns coded 1/2 (not 0/1) should be recoded: df[col] = df[col] - 1
"""

DEBUGGER_PROMPT = """IMPORTANT: Respond with ONLY a corrected Python code block. Do NOT use any file tools.

The script below failed with the following error. Fix it and return ONLY the corrected ```python ... ``` block.

ERROR:
{error}

ORIGINAL SCRIPT:
{script}

Common fixes:
- Ensure all column names exist in the dataframe before accessing
- Drop NaN before model fit
- lifelines CoxPHFitter: fit(df, duration_col='...', event_col='...')
- matplotlib backend: matplotlib.use('Agg') before any other matplotlib import
- Binary cols coded 1/2: convert with (col - 1)
"""


class ExperimentNode:
    def __init__(self, idea: dict, stage: int, parent_id: str = None):
        self.id        = str(uuid.uuid4())[:8]
        self.idea      = idea
        self.stage     = stage
        self.parent_id = parent_id
        self.status    = "pending"
        self.metrics   = {}
        self.script    = ""
        self.error     = None
        self.node_dir  = os.path.join(
            WORK_DIR, "phase2_experimentation", "nodes", self.id
        )
        os.makedirs(os.path.join(self.node_dir, "figures"), exist_ok=True)

    def _csv_path(self) -> str:
        return os.path.join(WORK_DIR, "data", "dataset.csv")

    def _resolve_endpoint(self):
        """
        Map the idea's primary_endpoint / time_horizon to (event_col, time_col).
        Adapt this mapping to your dataset's column names.
        """
        from config import ENDPOINTS
        horizon = self.idea.get("time_horizon", "primary")
        for key, (ev, tm) in ENDPOINTS.items():
            if key in horizon.lower():
                return ev, tm
        # Fall back to first endpoint
        return next(iter(ENDPOINTS.values()))

    def generate_script(self) -> str:
        endpoint, time_col = self._resolve_endpoint()
        prompt = NODE_PROMPT.format(
            csv_path=self._csv_path(),
            title=self.idea["title"],
            hypothesis=self.idea["hypothesis"],
            endpoint=endpoint,
            time_col=time_col,
            method=self.idea["statistical_method"],
            predictors=", ".join(self.idea.get("key_predictors", [])),
            horizon=self.idea.get("time_horizon", "primary"),
            node_dir=self.node_dir,
        )
        script = claude_call(prompt, max_tokens=4096)
        if script.startswith("```"):
            script = "\n".join(script.split("\n")[1:])
        if script.endswith("```"):
            script = "\n".join(script.split("\n")[:-1])
        self.script = script
        script_path = os.path.join(self.node_dir, "analysis.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)
        return script

    def execute(self, timeout: int = 7200) -> bool:
        script_path = os.path.join(self.node_dir, "analysis.py")
        log_path    = os.path.join(self.node_dir, "run.log")
        print(f"[Node {self.id}] Executing stage {self.stage}...")
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True, text=True, timeout=timeout,
                cwd=self.node_dir,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(result.stdout + "\n" + result.stderr)
            success = "STATUS: SUCCESS" in (result.stdout + result.stderr)
            if not success:
                self.error  = result.stderr[-3000:] or result.stdout[-3000:]
                self.status = "buggy"
            else:
                self.status = "non-buggy"
                self._load_metrics()
            return success
        except subprocess.TimeoutExpired:
            self.error  = "Execution timed out"
            self.status = "buggy"
            return False
        except Exception:
            self.error  = traceback.format_exc()
            self.status = "buggy"
            return False

    def debug(self, max_retries: int = 4) -> bool:
        for attempt in range(max_retries):
            print(f"[Node {self.id}] Debug attempt {attempt + 1}/{max_retries}")
            prompt = DEBUGGER_PROMPT.format(error=self.error, script=self.script)
            fixed  = claude_call(prompt, max_tokens=4096)
            if fixed.startswith("```"):
                fixed = "\n".join(fixed.split("\n")[1:])
            if fixed.endswith("```"):
                fixed = "\n".join(fixed.split("\n")[:-1])
            self.script = fixed
            with open(os.path.join(self.node_dir, "analysis.py"), "w", encoding="utf-8") as f:
                f.write(fixed)
            if self.execute():
                return True
        return False

    def _load_metrics(self):
        path = os.path.join(self.node_dir, "metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                self.metrics = json.load(f)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "stage": self.stage, "parent_id": self.parent_id,
            "status": self.status, "metrics": self.metrics,
            "error": self.error, "node_dir": self.node_dir,
        }
