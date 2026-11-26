# An Explainable Hybrid Digital Twin Framework Using Transformer and XGBoost for Early Alzheimer's Disease Prediction

> **Research-based risk screening only. Not a medical diagnosis.**

A final-year research project that predicts the risk of Alzheimer's disease from clinical, lifestyle,
cognitive and symptom data. It combines:

- an **FT-Transformer** (a Transformer for tabular data) that learns a patient representation,
- a tuned **XGBoost** model that makes the final prediction from engineered features plus that representation,
- **probability calibration**, so the risk percentage is meaningful,
- **SHAP** and **attention** explanations, and
- a **Digital Twin**: a per-patient database profile with history, a risk trend and what-if simulation,

all served through a Flask web dashboard.

Every number in this README and in the app comes from real training and evaluation runs
(`reports/results.json`, `reports/training_summary.json`).

---

## 1. Results

**Data used.** Kaggle Alzheimer's Disease Dataset (El Kharoua, 2024): 2,149 patients aged 60 to 90, 32 input
features, 760 with Alzheimer's (35.4%) and 1,389 without. One stratified 80/20 split (seed 42, indices in
`data/processed/split_indices.json`):

| Part | Patients | With AD | Without AD | Used for |
|---|---|---|---|---|
| Training set | 1,719 | 608 | 1,111 | fitting, feature selection, SMOTE, tuning, calibration, threshold; 5-fold CV (about 1,375 fit / 344 check per fold) |
| Test set (held out) | 430 | 152 | 278 | the final figures below, used once |

Tests entered in the web app are stored in the Digital Twin database only and are never added to the training data.
The same summary is shown on the app's *Model performance* page.

### Accuracy with valid labels

| Test records | Patients | Accuracy (95% CI) | Precision | Recall | F1 | ROC-AUC | Errors |
|---|---|---|---|---|---|---|---|
| **With valid labels** (PatientID < 6733) | 398 | **98.99%** (97.7–99.7%) | 97.12% | 100.00% | 98.54% | 0.995 | 4 |
| Full test set | 430 | 95.58% (93.5–97.4%) | 94.63% | 92.76% | 93.69% | 0.945 | 19 |
| Randomised labels only (PatientID ≥ 6733) | 32 | 53.12% (chance) | 60.00% | 35.29% | 44.44% | 0.533 | 15 |

The dataset is synthetic, and its last records in file order (PatientID 6733 to 6899) carry diagnosis labels that
behave as if assigned at random. In the training set, 46.7% of these records contradict their own clinical values,
against 0.7% of all earlier records, and no model or feature can predict their labels (cross-validated AUC about 0.5).
The boundary was found from the **training set only** (single change point in the rule-exception rate over PatientID
order, `reports/label_noise_cutoff.json`) and fixed before any test figure was computed. The same trained hybrid model
is scored on every row (`reports/label_noise_eval.json`); the model never uses PatientID, so the web app is unaffected.
15 of the 19 full-test errors come from the 32 randomised records. The web app's Overview shows the valid-label
figures; the Model performance page shows all three rows.

### Full test set, all models

Held-out test set: 430 patients (35.3% Alzheimer's). CV: 5-fold stratified on the 1,719 training patients
(threshold 0.5, uncalibrated). Source: `reports/model_comparison.csv`.

| Model | Test acc. | 95% CI (bootstrap) | Precision | Recall | F1 | ROC-AUC | CV acc. (mean ± std) | CV ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| **Hybrid Transformer + XGBoost (proposed)** | **95.58%** | [0.9372, 0.9744] | 94.63% | 92.76% | 93.69% | 0.9447 | 95.70 ± 1.26% | 0.9578 ± 0.0173 |
| XGBoost (tuned, engineered features) | 95.58% | [0.9372, 0.9744] | 94.63% | 92.76% | 93.69% | 0.9534 | 95.70 ± 1.26% | 0.9497 ± 0.0147 |
| FT-Transformer (engineered features) | 95.58% | [0.9372, 0.9744] | 94.63% | 92.76% | 93.69% | 0.9463 | 95.70 ± 1.26% | 0.9486 ± 0.0154 |
| FT-Transformer (no FE) | 94.88% | [0.9279, 0.9698] | 93.33% | 92.11% | 92.72% | 0.9371 | 93.83 ± 1.60% | 0.9526 ± 0.0092 |
| Hybrid without feature engineering | 94.65% | [0.9256, 0.9674] | 94.48% | 90.13% | 92.26% | 0.9498 | 93.83 ± 1.40% | 0.9534 ± 0.0130 |
| Random Forest | 94.19% | [0.9186, 0.9628] | 94.41% | 88.82% | 91.53% | 0.9422 | 93.89 ± 2.29% | 0.9518 ± 0.0117 |
| XGBoost (plain, raw features) | 94.19% | [0.9209, 0.9628] | 93.79% | 89.47% | 91.58% | 0.9436 | 94.77 ± 1.61% | 0.9554 ± 0.0165 |
| SVM (RBF) | 84.88% | [0.8116, 0.8814] | 80.00% | 76.32% | 78.11% | 0.9000 | 84.88 ± 1.73% | 0.9040 ± 0.0213 |
| Logistic Regression | 81.63% | [0.7791, 0.8512] | 74.50% | 73.03% | 73.75% | 0.8865 | 83.60 ± 1.72% | 0.9016 ± 0.0182 |

Proposed hybrid confusion matrix (test): TN 270, FP 8, FN 11, TP 141 (19 errors in 430).

### Ablation (`reports/ablation.csv`)

| Configuration | Test acc. | Test AUC | CV acc. |
|---|---|---|---|
| Transformer alone | 95.58% | 0.9463 | 95.70 ± 1.26% |
| XGBoost alone | 95.58% | 0.9534 | 95.70 ± 1.26% |
| Hybrid without feature engineering | 94.65% | 0.9498 | 93.83 ± 1.40% |
| **Full hybrid (proposed)** | **95.58%** | 0.9447 | 95.70 ± 1.26% |

- **Feature engineering is the component that helps**: +0.9 pp test accuracy for the hybrid and the
  Transformer, and +1.9 pp CV accuracy.
- **The hybrid does not beat XGBoost alone on this dataset.** All three engineered-feature models make
  identical decisions on the test set (McNemar p = 1.000 hybrid vs tuned XGBoost).

### Risk levels on the test set (`reports/risk_levels.csv`)

| Level | Patients | Actual Alzheimer's rate |
|---|---|---|
| Low (< 35%) | 281 | 3.9% |
| Medium (35-70%) | 0 | n/a |
| High (> 70%) | 149 | 94.6% |

The levels separate patients well, but the Medium band is empty: calibrated probabilities are bimodal
(about 3-7% or about 95%). That follows from the dataset (next section): the label is an almost
deterministic rule plus about 4% random noise, so the honest calibrated answer is "about 4%" or "about 95%".

### Why the 98.1% target was not met, and why we did not force it

The proposed hybrid reaches **95.58%**, below the 98.1% target. Analysis on the **training data only**
(`src/ceiling.py`, `reports/ceiling_analysis.json`):

- 95.7% of training diagnoses follow one clinical rule:
  `MMSE < 24 AND at least 2 of {FunctionalAssessment < 5, ADL < 5, MemoryComplaints, BehavioralProblems}`.
- The remaining 4.3% (74 patients) break the rule. A cross-validated XGBoost trying to predict *which*
  patients break it scores **AUC 0.515**, chance level. These labels behave like random noise.
- So the achievable accuracy is about **95.7%**. On 430 test patients its standard deviation is about
  1.0 pp, so 98.1% would need about 2.5 standard deviations of luck. The rule alone scores **95.58%** on
  the test set, exactly what every engineered-feature model reaches.

What was tried: clinical feature engineering, learned interaction terms, noise-probe feature selection,
Optuna tuning (100 trials per XGBoost, 20 for the Transformer), SMOTE / class weighting inside folds,
calibration, OOF threshold tuning, and the hybrid with aligned out-of-fold embeddings. None can recover
labels that carry no signal. Published scores of about 98% on this dataset most likely come from leakage
(for example SMOTE or scaling before the split, or tuning on the test set) or a lucky split. This project
avoids all of these, so its numbers should survive review.

What could legitimately help: real longitudinal data (MRI, CSF/PET biomarkers, repeated MMSE), a cleaner
dataset, or reporting repeated-split / nested-CV distributions instead of one split. On this dataset,
the useful contributions of the framework are **explainability, calibration and the Digital Twin**,
not raw accuracy.

### Explainability highlights (`reports/explainability.json`)
- The Transformer components carry **85%** of the hybrid's mean |SHAP|. XGBoost relies mainly on the
  learned representation plus the `ImpairmentDomainCount x MMSE_Impaired` interaction.
- Transformer attention and SHAP agree on the top feature (`ImpairmentDomainCount x MMSE_Impaired`), but
  their overall rank correlation is low (Spearman 0.06): attention shows where the model looks, not importance.

### Home screening: before any clinical test (`src/pretest.py`, `reports/pretest_results.json`)

The app is meant for ordinary people, so it was also checked the way they would use it: without blood tests,
often without blood pressure, and with memory and daily-life scores estimated at home instead of measured at a
clinic. Test set, 430 patients; missing values are replaced by training-set medians.

| Scenario | Test accuracy | Recall | ROC-AUC |
|---|---|---|---|
| A. All inputs (clinic-measured scores) | 95.58% | 92.76% | 0.945 |
| B. No blood tests | 95.58% | 92.76% | 0.938 |
| C. No blood tests and no blood pressure | 95.58% | 92.76% | 0.942 |
| D. C + small error in home-estimated scores (MMSE +/-2, others +/-1) | 88.9% +/- 1.0 | 83.7% | 0.883 |
| D. C + large error in home-estimated scores (MMSE +/-4, others +/-2) | 82.2% +/- 1.5 | 73.6% | 0.823 |
| E. No memory or daily-life information (separate XGBoost, demographics + lifestyle + history + symptoms) | 69.5% | 58.6% | 0.691 |

- **Blood tests and blood pressure are not needed** for this model, so they are optional in the form.
- **Memory and daily-life information is essential.** Without it accuracy is 69.5%, close to the 64.7% you get
  by always predicting "no Alzheimer's". That is why the home questionnaire and brain check exist.
- **The quality of the home estimate matters.** The D rows simulate estimation error. No dataset links the home
  questions to clinic scores, so the true error of the home questionnaire is unknown and should be measured in a
  real validation study.

---

## 1b. Accounts, levels and trends (the Digital Twin in use)

One **account** can hold **several profiles**, each its own Digital Twin: for example yourself, your mother and your
father (`/me/profiles`). Each profile records who the person is (myself, mother, father, spouse, grandparent, other
relative, someone I care for). One profile is *selected* at a time (shown in the menu); tests, *Edit profile* and
scan reports act on it, and opening a profile's dashboard selects it. A profile can be deleted with all its records.
*Log out* is in the top bar on every page.

1. **Register and log in** (`/register`, `/login`). Passwords are stored only as salted hashes.
2. **Profile** (one per person), set up once and edited when something changes: name, date of birth (or approximate age),
   gender, ethnicity, education, **height** (feet + inches, cm, inches or metres), **weight** (kg or pounds) giving
   BMI, family and health history, lifestyle, and clinic numbers (blood pressure, cholesterol) when available. Only
   age, gender, height, weight and family history are required. Every height or weight change is saved with its date, which gives a BMI history.
3. **Level 1, home check**, repeated regularly (every 90 days is suggested): memory and behaviour, daily life
   and the brain check; confusion, disorientation and personality changes are optional. Everything else comes from
   the profile at the time of the test. Each assessment needs 11 of the model's 32 inputs (`CORE_INPUTS` in
   `src/schema.py`); the others use training medians when blank. With only those 11 on the test set, the unchanged
   hybrid scores 98.99% on valid labels and 95.58% on the full test set, the same as with all 32.
4. **Level 2, clinical report:** blood pressure, blood tests and a doctor's MMSE / functional / ADL scores. These
   are combined with the latest home check and **replace the home estimates**, giving a clinically verified risk. New blood pressure and cholesterol values also update the profile.
   It is only available after a home check.
5. **Scan reports** (MRI / CT / PET: date, centre, findings, optional PDF/JPG/PNG up to 10 MB) are kept in the
   record. The model does **not** use them.
6. **Dashboard** (`/me`): latest risk, test status, profile, trend verdicts since the previous and the first
   test, **compare any two tests** (risk, memory, everyday tasks, self-care, symptoms, BMI, with per-year MMSE change),
   charts over time, what-if simulation, and the full history (tests, scans, profile changes) newest first.

Trend verdict rules (`src/trends.py`): a change of risk level, or of 10 or more percentage points, decides
"Improving" or "Worsening". Otherwise a memory change of 2 or more points or a daily-life score change of 1 or
more gives "Possibly improving" or "Possibly worsening"; anything else is "Stable". A warning is shown when a home
estimate is compared with a clinic score. These are screening summaries, not clinical judgements.

An account can only see the profiles it holds. **Researcher** accounts see every twin (for project review).

Demo logins (local demo only; created by `python -m src.accounts` or `python -m src.seed_demo`):

| Username | Password | Sees |
|---|---|---|
| `researcher` | `research123` | all Digital Twins |
| `demo` | `demo123` | Demo patient 01 (declining over simulated visits) |
| `demo2` | `demo123` | Demo patient 04 (improving over simulated visits) |

**Voice help** works on the profile, home check, clinical report and scan pages. Questions are read aloud in every
modern browser. Spoken answers work in **Chrome, Edge and Safari** (macOS 14.1+, iOS 14.5+); **Firefox does not
support speech recognition**, so there it only reads aloud. On iPhone every browser, including Chrome, uses Apple's
WebKit, so it behaves like Safari. Microphone access needs a secure page: `http://127.0.0.1` works on the same
computer. To test from a phone the app must be served over **HTTPS**, for example after deployment or through a
tunnel.

## 2. Setup

```bash
cd alzheimers-digital-twin
python3.10 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.10 on macOS (Apple M4, CPU only). All library versions are pinned in `requirements.txt`.

### Getting the dataset

The project uses the public Kaggle **Alzheimer's Disease Dataset** by Rabie El Kharoua
(`rabieelkharoua/alzheimers-disease-dataset`, file `alzheimers_disease_data.csv`, 2,149 patients, 35 columns).
No synthetic replacement data is ever generated.

Option A, Kaggle API (needs `~/.kaggle/kaggle.json`):
```bash
pip install kaggle
kaggle datasets download -d rabieelkharoua/alzheimers-disease-dataset -p data/raw --unzip
```
Option B, `kagglehub` (works without an account for public datasets; install it in a separate environment
because it pins a different `kagglesdk` version than `kaggle`):
```bash
python -c "import kagglehub; print(kagglehub.dataset_download('rabieelkharoua/alzheimers-disease-dataset'))"
# then copy alzheimers_disease_data.csv into data/raw/
```
Option C: download it in the browser from Kaggle and place the CSV in `data/raw/`.

SHA-256 of the file used for the reported results:
`66f51dbb5f00a4bf6ed1f631e72ec15e4a8e97dc223c10f78c1febec95a2be07`

## 3. Running everything

| Step | Command | Time (M4 CPU) |
|---|---|---|
| Train/test split + summary | `python -m src.data` | seconds |
| Exploratory data analysis (plots to `reports/figures/`) | `python -m src.eda` | seconds |
| **Train all models** (training data only) | `python -m src.train` | ~45-55 min |
| Quick smoke test of training | `python -m src.train --fast` | ~7 min |
| **Evaluate on the test set** + explanations | `python -m src.evaluate` | ~1 min |
| Explanations only | `python -m src.explain` | ~30 s |
| **Seed demo Digital Twins** (resets the database) | `python -m src.seed_demo` | seconds |
| Add demo/researcher accounts to an existing database (keeps data) | `python -m src.accounts` | seconds |
| **Run the web app** | `python app/app.py`, then open http://127.0.0.1:5050 and log in (set `PORT=...` to change; macOS AirPlay occupies 5000) | |

The app loads the saved models from `models/`; it never retrains.

## 4. Evaluation protocol (how leakage is prevented)

1. A **stratified 80/20 split** is made first with a fixed seed (42) and saved to
   `data/processed/split_indices.json` (1,719 train / 430 test patients).
2. `src/train.py` **only loads the training split**. Scaling, feature engineering, interaction selection,
   feature selection, SMOTE, Optuna tuning, Transformer early stopping, calibration and the decision
   threshold are all fitted on training data, using 5-fold stratified CV or an inner validation split.
3. Feature engineering and SMOTE are **re-fitted inside every CV fold**, so the validation fold never influences them.
4. Training-set Transformer embeddings for the hybrid are produced **out-of-fold**.
5. `src/evaluate.py` scores every saved model on the test set **once**. Nothing is fitted there.
6. The random seed is fixed and was **never searched**.

Reported: test accuracy, precision, recall, F1, ROC-AUC, confusion matrix and a 95% bootstrap CI,
plus 5-fold CV mean and standard deviation on the training set.

Small caveats a reviewer may raise, stated openly:
- CV scores of tuned models are slightly optimistic because Optuna chose hyper-parameters on the same folds
  (standard practice; the test set is unaffected).
- In the hybrid's CV, the out-of-fold embeddings of the *training* rows came from Transformers that saw the
  validation rows. This is the usual second-order effect of stacking and does not touch the test set.

## 5. Folder structure

```
alzheimers-digital-twin/
  data/raw/                 alzheimers_disease_data.csv (Kaggle)
  data/processed/           split_indices.json (saved train/test split)
  src/
    config.py               paths, seed, risk thresholds, budgets, feature groups (single source of truth)
    data.py                 loading, dropping ID columns, stratified split, CV folds
    eda.py                  exploratory data analysis
    features.py             clinical feature engineering + interaction terms + noise-probe feature selection
    ceiling.py              label-noise / accuracy-ceiling analysis (training data)
    metrics.py              metric helpers
    models/
      baselines.py          Logistic Regression, SVM, Random Forest, plain XGBoost
      transformer.py        FT-Transformer (PyTorch) + sklearn-like wrapper
      xgb_tuning.py         Optuna tuning of XGBoost (imbalance strategy incl. in-fold SMOTE)
      hybrid.py             Transformer tuning, out-of-fold embeddings, Procrustes alignment
      calibration.py        isotonic / Platt calibration and threshold selection
      bundle.py             uniform wrapper: raw features -> calibrated probability
    train.py                trains everything (python -m src.train)
    evaluate.py             test-set evaluation, tables and figures (python -m src.evaluate)
    explain.py              global SHAP + attention figures (python -m src.explain)
    predictor.py            live prediction + per-patient explanation service
    (app/static/voice.js    voice help: questions read aloud, spoken answers understood and filled in)
    schema.py               input fields, plain-language questions, ranges, validation (optional fields)
    body.py                 height/weight parsing (feet-inches, 5.4, cm, m, kg, lb) and BMI
    questionnaire.py        home questionnaire + brain check that estimate ADL, functional score and MMSE
    pretest.py              home-screening evaluation (missing measurements, estimation error, no-score model)
    twin.py                 Digital Twin database: users, patients, tests, predictions, body measurements, scans
    profile.py              profile validation, age from date of birth, profile -> model inputs
    trends.py               trend verdicts and comparison of any two tests
    accounts.py             demo and researcher accounts (python -m src.accounts)
    seed_demo.py            demo twins (python -m src.seed_demo)
  models/                   saved models (.joblib for sklearn/XGBoost, .pt for PyTorch)
  reports/                  results.json, model_comparison.csv, ablation.csv, risk_levels.csv, ...
  reports/figures/          all plots
  app/                      Flask app (app.py, auth.py login/register, routes.py, templates/, static/)
  database/                 digital_twin.db (SQLite)
  notebooks/                (optional EDA notebooks)
```

## 6. How it works (viva preparation)

### 6.1 Data and feature engineering
The dataset has 32 usable inputs (PatientID and DoctorInCharge are dropped as identifiers) and a binary label
`Diagnosis`. `src/features.py` adds clinically meaningful features:

| Feature | Formula | Why |
|---|---|---|
| CognitiveFunctionalComposite | mean of MMSE/30, FunctionalAssessment/10, ADL/10 | one overall cognition and function score |
| SymptomCount | sum of 7 symptoms | symptom burden |
| CardioRiskCount | smoking + CVD + diabetes + hypertension + (cholesterol >= 240) | vascular risk burden |
| LDL_HDL_Ratio | LDL / HDL | lipid risk |
| PulsePressure | systolic - diastolic | arterial stiffness |
| MMSE_Impaired | MMSE < 24 | standard clinical cut-off for cognitive impairment |
| FA_Low, ADL_Low | score < 5 (scale midpoint) | functional impairment flags |
| ImpairmentDomainCount | FA_Low + ADL_Low + MemoryComplaints + BehavioralProblems | number of impaired domains |
| 6 interaction terms | products of the top 4 features by training-set XGBoost importance | let models use combinations directly |

Feature selection keeps only features that are more important than shuffled "noise probe" columns.
`src/train.py` compares raw vs full engineered vs selected sets by CV and keeps the best.

### 6.2 FT-Transformer
- **Tokenisation:** every feature becomes a vector ("token"). A numerical feature `x` becomes `x*W + b`
  with its own `W, b`; a categorical feature uses its own embedding table.
- A learnable **[CLS] token** is added at the front, as in BERT.
- **Transformer encoder layers:** multi-head self-attention lets each feature token look at every other token
  (for example MMSE attending to ADL). Each layer has attention, a feed-forward network, layer normalisation,
  residual connections and dropout.
- The final [CLS] vector is the **patient embedding**. A linear head turns it into the logit of Alzheimer's.
- Training: AdamW, cosine learning-rate schedule, early stopping on validation loss (best weights restored),
  optional class weighting (an Optuna choice). Hyper-parameters are tuned with Optuna on 3 training folds.
- **Attention weights** are saved, so we can show which features the [CLS] token attends to.

### 6.3 Hybrid Transformer + XGBoost
1. Train the full Transformer on the training set.
2. **Out-of-fold embeddings:** for each of 5 folds, train a Transformer on the other 4 folds and embed the
   held-out fold, so no training patient is embedded by a model that saw its label.
3. **Alignment:** each fold model has its own embedding coordinate system (dimension 3 in one model means
   something different in another). An orthogonal Procrustes rotation, fitted on that fold's training rows,
   maps each fold's embeddings into the full model's space (alignment R² 0.992-0.995 in the final run).
4. XGBoost (tuned with 100 Optuna trials on CV log-loss) is trained on
   **[engineered features + 16-d embedding + Transformer probability]** (embedding size chosen by Optuna) and makes the final prediction.
5. **Calibration:** isotonic and Platt scaling are compared by CV Brier score on out-of-fold predictions; the
   better one is used. The decision threshold is chosen on the same out-of-fold predictions.

### 6.4 Risk levels
The calibrated probability is mapped to **Low (< 35%)**, **Medium (35-70%)** and **High (> 70%)**
(`src/config.py`). `reports/risk_levels.csv` shows the actual Alzheimer's rate in each level on the test set.

### 6.5 SHAP
SHAP (SHapley Additive exPlanations) splits a prediction into additive contributions, one per feature.
`TreeExplainer` computes exact SHAP values for XGBoost. Because SHAP is additive, the 16 embedding dimensions
are **summed into one "Transformer embedding" contribution** to keep explanations readable. We show:
- global beeswarm and mean |SHAP| plots for the hybrid and for the plain XGBoost on the original features;
- for each patient: a waterfall plot, a force plot and the top 5 factors up and down in plain words
  (for example "Low MMSE score (12.3) increased the risk");
- the Transformer's [CLS] attention per feature, compared with SHAP rankings.

Attention is **not** the same as importance: it shows where the model looks, while SHAP measures the effect on
the output. The comparison plot shows where they agree.

### 6.6 Digital Twin
Each patient has a Digital Twin stored in SQLite (`database/digital_twin.db`) with three tables:
- `patients` holds fixed demographics, a demo flag and the source;
- `observations` holds every input set with a timestamp, plus a `is_simulated` flag;
- `predictions` holds probability, risk level, model version, timestamp and top SHAP factors (JSON).

The Digital Twins list and each twin's history show the newest visit first. The twin page shows patient details,
latest risk, top factors, full history, a **risk trend chart** (from two
observations onwards) and **what-if simulation**: sliders for modifiable factors (MMSE, functional scores,
activity, sleep, diet, alcohol, BMI, blood pressure, cholesterol, smoking) recompute the risk live through
`/api/whatif`, while age, gender, ethnicity, education and family history stay fixed.

`python -m src.seed_demo` loads six real **test-set** patients as demo twins. Three of them get
**simulated** follow-up visits (decline, stable and improvement scenarios). These visits are flagged
`is_simulated=True` in the database and labelled "Simulated demo visit" in the UI.

### 6.7 Home screening form for non-specialists
The input form is an 8-step guided form that a family member can fill in, written in plain language:
1. About the person, 2. Lifestyle, 3. Health history (plain "Has a doctor ever said..." questions),
4. Health check numbers (**optional**: blood pressure and blood-test values; blanks use the training median and are
labelled "not provided" in every explanation), 5. Memory and behaviour (the 7 symptoms as everyday Yes/No questions
with examples), 6. Daily life, 7. Quick brain check, 8. Review.

In step 6 the user chooses either **simple questions** (default) or **doctor's scores**. With simple questions,
`src/questionnaire.py` estimates the three clinical scores:

| Model input | Estimated from | Scoring |
|---|---|---|
| ADL (0-10) | 6 self-care questions (bathing, dressing, toilet, moving, continence, eating) | without help = 2, some help = 1, cannot = 0; scaled to 0-10 |
| Functional assessment (0-10) | 8 everyday-task questions (phone, shopping, cooking, housework, laundry, travel, medicines, money) | same; "never did this task" is excluded |
| MMSE (0-30) | 11-task on-screen brain check: 3 words (now and later), date and time of day, place, counting back by 7, naming 2 pictures, copying a sentence, a 3-step tap instruction, reading, writing, shape matching | 30 points, scored on the server; most tasks are auto-checked, place answers are ticked by a helper |

The items follow standard domains (Katz-style self-care, Lawton-style everyday tasks, usual brief cognitive-screen
domains) in our own words. They are **not** the copyrighted MMSE and have **not** been validated against the
dataset's scores. The result page always says when scores were estimated at home, shows the brain-check
breakdown, and lists which measurements were not provided. Only scores are stored, not typed answers.

### Height, weight and BMI
Most people don't know their BMI, so by default the form asks for **height and weight** and calculates BMI
(`src/body.py`; the same parser powers the live preview at `/api/body` and the saved value). Accepted entries:
- Height: `5'4"`, `5' 4`, `5 ft 4 in`, `5 feet 4 inches`, `6 feet`, `5.4` (read as 5 ft 4 in, the common Indian
  way of writing it; `5.10` and `5.11` mean 10 and 11 inches), `165 cm`, `165`, `1.65 m`, `64 in`.
- Weight: a number with a kg / pounds selector, or typed with a unit (`70 kg`, `154 lb`, `154 pounds`).

The form shows how each entry was understood ("Understood as 5 ft 4 in = 162.6 cm"), and implausible values are
rejected. People who know their BMI can enter it directly. The entered height and weight are stored in the
Digital Twin and pre-filled at the next visit.

### Age: younger adults and children
- **Adults 18 and over** can be screened. The research data covers ages 60 to 90, so for any value outside the
  training range (for example age 45 or BMI 44) the model uses the nearest value it was trained on (60, or 40).
  The real value is still stored, and the result page says exactly what was used.
- **Under 60:** the form and result show an **early-onset** note. Alzheimer's before 65 is less common, often runs
  in families, and is not represented in the training data, so the result is less certain.
- **Under 18:** no risk score is given. Children do not get Alzheimer's disease; "childhood Alzheimer's" is a
  nickname for different rare genetic illnesses such as Niemann-Pick type C or Sanfilippo syndrome. The form explains
  this and points to a paediatrician.

**Data for younger-onset cases.** No public, freely downloadable tabular dataset with young-onset Alzheimer's cases
and these features was found (searched October 2026). Real early-onset data exists in access-controlled research
cohorts that need an application, for example LEADS (Longitudinal Early-onset Alzheimer's Disease Study, via
LONI/IDA), ADNI, OASIS-3 and NACC. OASIS-1 includes adults aged 18-96, but all its Alzheimer's cases are over 60.
No synthetic young-onset data was generated. Extending the model to younger ages needs one of these cohorts and
retraining.

### Voice help for people who cannot easily read or write
The assessment form has a **Voice help** button (`app/static/voice.js`). When it is on:
- Each question is **read aloud** in Indian English. After a beep the person **answers by speaking**, the answer
  is filled in, and the app **repeats back** what it understood ("I heard: Yes") so mistakes can be caught. If an
  answer is not understood it asks again, up to three times.
- It understands everyday answers: yes/no (also "haan" / "nahi"), "does it alone" / "needs some help" /
  "can't do it", choices ("a man", "we are Indian", "studied till school"), numbers said as words ("seventy two"),
  heights ("five feet four inches") and weights ("sixty five kilos").
- In the brain check the three words are **spoken instead of shown**. Spoken answers fill the date, place,
  counting, naming and recall tasks. The copy and writing tasks become "repeat this sentence" and "say a sentence",
  and the reading card is read aloud, so people who cannot read or write are not penalised for literacy. Tap tasks
  (shapes, matching picture) stay as taps, with the instructions read aloud.
- Every question also has its own **Listen** and **Answer by voice** buttons, and the result page has
  **Read the result aloud**. Assessments done with voice are marked "answered with voice help".

Browser support and privacy: reading aloud works in all modern browsers. Spoken answers need Google Chrome or
Microsoft Edge with a microphone. In these browsers speech recognition is done by the browser vendor's online
service (for Chrome, Google), so spoken answers leave the computer. The page says which browsers work.
Only English (India) is supported for now. Other Indian languages need translated question text.

Testing: a test hook (`window.__voiceTest`) replaces the microphone with scripted answers, which is how the whole
guided flow (all 8 steps including the brain check) was tested automatically in headless Chrome.

### 6.8 Ethnicity options in the form
The training data has only four ethnicity codes: Caucasian, African American, Asian and Other. The form offers
more specific choices, grouped as South Asian (Indian, Pakistani, Bangladeshi, Sri Lankan, Nepali/Bhutanese),
East and Southeast Asian, and other backgrounds. The exact choice is stored in the Digital Twin
(`EthnicityDetail`) and mapped to the dataset group the model understands. For example, Indian is mapped to
Asian. The form and twin page show this as "Model group". A model cannot learn a category it never saw in
training. In this dataset ethnicity carries almost no signal anyway (hybrid mean |SHAP| = 0.0).
Mappings live in `src/schema.py` (`ETHNICITY_OPTIONS`).

## 7. Limitations
- The dataset's "Asian" group is broad and probably not representative of Indian patients specifically.
  Validating on Indian clinical data would be needed before drawing conclusions for that population.
- The dataset is synthetic and cross-sectional. Real longitudinal validation would be needed before any clinical use.
- About 4% of labels are not predictable from the features (see Results), which caps accuracy.
- Simulated follow-up visits are illustrative only.
- Home-estimated memory and daily-life scores are unvalidated approximations of the clinical scores.
- The model was trained on ages 60-90 only. Results for younger adults (early-onset) are extrapolations,
  and children are not screened.
- What-if results are model associations, not causal effects.

## 8. Citation
Rabie El Kharoua (2024). *Alzheimer's Disease Dataset*. Kaggle. https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset
