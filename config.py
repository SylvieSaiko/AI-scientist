"""
AI Scientist — dataset configuration.
Fill in DATA_PATH and the column mappings for your own dataset before running.
"""

WORK_DIR  = "/path/to/AI_Scientist"   # replace with your working directory
DATA_PATH = "/path/to/your_dataset"   # replace with your dataset path

# Outcome column pairs: (event_column, time_column)
# Replace the keys and values with your dataset's actual column names.
ENDPOINTS = {
    "primary":   ("event_col",        "time_col"),
    "secondary": ("secondary_event",  "secondary_time"),
}

# Column groups — replace with your dataset's column names
DEMO_COLS         = []   # e.g. ["age", "sex", "bmi"]
COMORBIDITY_COLS  = []   # e.g. ["hypertension", "diabetes"]
LAB_COLS          = []   # e.g. ["lvef", "egfr", "ldl"]
PROCEDURAL_COLS   = []   # e.g. ["num_stents", "multivessel"]
MEDICATION_COLS   = []   # e.g. ["statin", "acei_arb"]
