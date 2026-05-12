"""
Peer review of:
Ding G et al. "Development and Validation of a Machine Learning Model for
Hepatitis C Virus Exposure: A Demographic Screening Approach for the US Population"
"""
import sys, os
sys.path.insert(0, "/Users/sylviesaiko/Desktop/AI_Scientist")
from claude_client import claude_call

PAPER = """
TITLE: Development and Validation of a Machine Learning Model for Hepatitis C Virus
Exposure: A Demographic Screening Approach for the US Population

AUTHORS: Guanwen Ding¹, Taoyi Chen², Yu Sheng³, Jeffrey S. H. Lin⁴, Ye Yuan⁵

ABSTRACT:
Background: Hepatitis C virus (HCV) remains underdiagnosed in the United States despite
recommendations for universal screening. A simple approach based on readily available
demographic information may help target screening in settings where screening implementation
remains incomplete.

Methods: We analyzed 10 NHANES cycles (1999-2014 and 2017-2023) and defined HCV exposure as
a positive HCV antibody or RNA result. Using sex, birth year, race/ethnicity, birthplace, and
income-to-poverty ratio, we trained and compared logistic regression (LR) and machine learning
models in training and validation cohorts (48,434 and 20,762 participants, respectively). Model
performance was evaluated based on sensitivity, specificity, positive predictive value, negative
predictive value, and AUROC. A web-based calculator was developed to facilitate bedside HCV
screening.

Results: 69,196 participants were included, with 967 showing evidence of HCV exposure. Weighted
HCV prevalence remained relatively stable across cycles (1.22%-1.93%). XGBoost performed better
than LR in the validation cohort (AUROC 0.860 vs 0.762). Predicted risk separated the population
clearly: observed HCV prevalence increased from 0.05% in the lowest-risk decile to 7.95% in the
highest, with the top decile containing 58.3% of participants with HCV exposure and the top three
deciles containing 85.5%.

Conclusions: Five demographic variables were sufficient to build a useful HCV risk model in a
nationally representative US sample. Most HCV-exposed individuals were concentrated in the highest
predicted-risk groups, suggesting that this approach could help prioritize and optimize testing
where universal screening uptake remains incomplete.

---
INTRODUCTION (summary):
- HCV: 57.8 million living with chronic infection globally (2020); 2.2 million infected in US
- ~70% progress to chronic infection; 1/3 develop cirrhosis; 12-fold higher liver-related mortality
- WHO elimination target 2030; HCV infections declined 11% (2015-2020) — insufficient pace
- Universal adult screening now recommended; real-world implementation incomplete (only 40.6%
  of pregnant individuals screened in 2021 despite clear recommendation)
- Prior ML model used 238 predictors (C-stat 0.916, Jang et al.) — impractical for broad use
- Goal: validate a parsimonious 5-variable demographic model using NHANES; develop web calculator

METHODS:
Study Design: Retrospective cross-sectional analysis of 10 NHANES cycles (1999-2014, 2017-2023).
Cycle 2015-2016 excluded (HCV lab data unavailable).

Population: 109,584 distinct participants identified; 31,863 excluded (missing HCV status);
8,525 excluded (missing demographic data) → 69,196 final.

Outcome: HCV exposure = positive HCV antibody OR positive HCV RNA.
  Current infection: RNA positive (regardless of antibody).
  Resolved infection: antibody positive, RNA negative.

Covariates (5 model predictors): sex, birth year, race/ethnicity (White/Black/Hispanic/Asian-Other),
birthplace (US vs non-US), income-to-poverty ratio.
Additional covariates collected for characterization: education, marital status, insurance,
smoking, alcohol, depression (PHQ-9), DM, CAD, BMI, HbA1c, glucose, lipids, liver enzymes,
creatinine, platelets.

Training/Validation Split (fixed-cycle):
  Training: cycles 1999-2000, 2003-2004, 2007-2008, 2009-2010, 2011-2012, 2013-2014,
            2017-March 2020 pre-pandemic (n=48,434)
  Validation: cycles 2001-2002, 2005-2006, 2021-2023 (n=20,762)

Models (6 total): LR (logistic regression), WLR (weighted LR), RF (random forest),
  XGB (XGBoost), GAM (generalized additive model), GLMNET (elastic-net logistic regression).
  All trained on 5 demographic predictors only.

Classification threshold: 80th percentile of predicted risk (top 20% = high risk).
Calibration: Platt scaling on training cohort.
Interpretability: SHAP values (Monte Carlo, random subsample ≤400 from validation cohort).
Sensitivity analysis: outcome redefined as current HCV RNA positivity only.

Statistical analysis: NHANES survey weights; Pearson chi-square (categorical), Wilcoxon
rank-sum (continuous); DeLong test for AUROC comparisons; two-sided α=0.05.

Web calculator: Deployed at https://sylviesaiko.github.io/hcv-risk-calculator/

RESULTS:
Prevalence: Stable across cycles (1.22%-1.93%); no significant pre/post-pandemic difference
(1.71% [95% CI 1.08%-2.69%] in 2017-2020 vs 1.38% [0.98%-1.95%] in 2021-2023). Higher in
males and non-Hispanic Black across all cycles.

Baseline characteristics (HCV-exposed vs unexposed):
  - Male: 62.15% vs 48.65%
  - Birth year: 1958 [1952-1964] vs 1976 [1955-1991]
  - Non-Hispanic Black: 34.85% vs 22.30%
  - US-born: 92.04% vs 81.87%
  - Less than college: 93.24% vs 79.40%
  - Smoking (ever): 73.74% vs 26.08%
  - Probable depression: 20.65% vs 9.12%
  - Income-poverty ratio: 1.3 [0.8-2.4] vs 2.0 [1.0-3.9]
  - ALT: 33.0 vs 19.0 U/L; AST: 33.0 vs 22.0; GGT: 39.0 vs 18.0; LDL: 98 vs 104 mg/dL

Current vs resolved infection (among 888 with RNA results):
  580 current, 308 resolved. Current infection more common in men, non-Hispanic Black;
  less insurance, lower income, higher smoking. Liver enzymes (ALT, AST, GGT, TBil)
  higher in current vs resolved. Proportion currently infected fell from >80% in early
  cycles to 40.9% in 2017-2020 and 31.9% in 2021-2023.

Logistic regression predictors (adjusted OR):
  Birth year (per 10yr): 0.72 [0.70-0.75], p<0.001
  Male sex: 1.91 [1.63-2.23], p<0.001
  Black vs White: 1.87 [1.56-2.24], p<0.001
  US-born vs non-US-born: 3.04 [2.23-4.13], p<0.001
  Income-poverty ratio (per 1 unit): 0.69 [0.65-0.73], p<0.001
  Hispanic vs White: 1.08 [0.85-1.38], p=0.523 (NS)
  Asian/Other vs White: 1.30 [0.90-1.89], p=0.162 (NS)

Model performance (validation cohort):
  LR:      sensitivity 0.527, specificity 0.804, PPV 0.036, NPV 0.992, AUROC 0.762
  GAM:     sensitivity 0.753, specificity 0.808, PPV 0.051, NPV 0.996, AUROC 0.856
  GLMNET:  sensitivity 0.519, specificity 0.804, PPV 0.035, NPV 0.992, AUROC 0.759
  RF:      sensitivity 0.731, specificity 0.807, PPV 0.050, NPV 0.995, AUROC 0.848
  WLR:     sensitivity 0.519, specificity 0.804, PPV 0.035, NPV 0.992, AUROC 0.757
  XGB*:    sensitivity 0.760, specificity 0.808, PPV 0.052, NPV 0.996, AUROC 0.860 (best)

Training AUROC: RF 0.915, GAM 0.868, XGB 0.893, LR 0.783
  RF shows largest train-validation gap (0.915→0.848), XGB more stable (0.893→0.860).

Decile stratification (validation, XGB): prevalence from 0.05% (lowest) to 7.95% (highest);
  top decile captures 58.3% of HCV-exposed; top 3 deciles capture 85.5%.
  LR: 0.14% to 4.14% (~30-fold); top decile captures 30.4%.

SHAP analysis: Birth year is dominant predictor (mean |SHAP|=0.22 across all models);
  income-poverty ratio second (mean |SHAP|=0.075 in tree models).

Sensitivity analysis (current HCV RNA only): similar pattern; XGB best (AUROC 0.867 validation);
  RF 0.906 training → 0.848 validation.

DISCUSSION SUMMARY:
- Stable HCV prevalence; declining proportion currently infected (DAA effect)
- Five demographic variables sufficient for useful risk stratification
- Birth year dominant predictor (birth-cohort effect); income-poverty ratio second
- Models complementary to, not replacements for, universal screening
- Strengths: NHANES nationally representative; SHAP interpretability; web calculator;
  2021-2023 cycle included; sensitivity analysis consistent
- Limitations: NHANES cross-sectional (no incident infections); NHANES excludes high-HCV
  populations (incarcerated, homeless); internal validation only; binary sex variable
"""

REVIEW_PROMPT = f"""You are an expert peer reviewer for a leading epidemiology/infectious disease journal
(e.g., The Lancet Infectious Diseases, Clinical Infectious Diseases, or Journal of Hepatology).
You have deep expertise in:
- Machine learning model development and validation methodology
- NHANES survey design and appropriate use of survey weights
- HCV epidemiology in the US and global context
- Clinical screening tool development and calibration
- TRIPOD and STROBE reporting guidelines

Please write a comprehensive, structured peer review of the manuscript below. Your review should be:
- Rigorous but constructive — identify both strengths and specific, actionable weaknesses
- Specific — cite line numbers, table numbers, or exact claims where relevant
- Clinically grounded — assess whether the findings are meaningful and well-contextualized

Structure your review as:
1. SUMMARY (3-4 sentences: what the study does and its main contribution)
2. MAJOR CONCERNS (numbered list; each must have a specific actionable suggestion)
3. MINOR CONCERNS (numbered list)
4. STATISTICAL AND METHODOLOGICAL COMMENTS
5. CLINICAL RELEVANCE AND IMPACT
6. OVERALL RECOMMENDATION (Accept / Minor Revision / Major Revision / Reject) with 2-sentence justification

Be thorough and critical. This is a pre-submission review intended to strengthen the paper.

--- MANUSCRIPT ---
{PAPER}
"""

print("Running peer review via Claude...")
review = claude_call(REVIEW_PROMPT, max_tokens=4000)

out_path = "/Users/sylviesaiko/Desktop/AI_Scientist/outputs/HCV_Screening_PeerReview.txt"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("PEER REVIEW\n")
    f.write("Manuscript: Development and Validation of a Machine Learning Model for HCV Exposure\n")
    f.write("Ding G, Chen T, Sheng Y, Lin JSH, Yuan Y\n")
    f.write("="*80 + "\n\n")
    f.write(review)

print(f"\nReview saved to: {out_path}")
print("\n" + "="*80)
print(review)
