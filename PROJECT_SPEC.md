# CardioScreen — 12-Lead ECG Abnormality Screening

**Build specification.** This document is the single source of truth for the project. Read it fully before writing code. Follow it section by section.

---

## 0. Ground rules (read first)

These are non-negotiable. The value of this project is that it is *real*.

1. **No fabricated data, ever.** No synthetic ECGs, no mock predictions, no hardcoded demo outputs, no simulated "live" streaming. Every number shown in the UI must come from an actual model forward pass or an actual evaluation run.
2. **No placeholder metrics.** If a metric has not been computed yet, the UI shows an empty state. It never shows an invented number.
3. **Fail loudly.** If a record fails to load or the model is missing, raise a clear error. No silent fallbacks to fake values.
4. **Patient-level splits only.** Never split by record or by row. Leakage across patients is the single most common way this class of project gets marked down.
5. **Test set is touched once.** Use validation for all tuning. Run the test set at the very end, and only again if the model genuinely changes.
6. **Every metric in the UI is reproducible** by running one script in the repo.
7. Commit in small, working increments with meaningful messages.

---

## 1. What this project is

A screening tool that takes a real 12-lead ECG recording and returns, for each of five diagnostic categories, a calibrated probability that the category is present — along with a visual indication of which part of the signal drove the prediction.

**Framing for the presentation:** this is a *screening aid* that helps prioritise which ECGs a clinician looks at first. It is not a diagnostic device. State this explicitly in the UI footer and on your slides.

### Why this problem is worth solving
ECGs are high-volume and mostly normal. A reliable triage model that surfaces likely-abnormal recordings first reduces the time to review for the ones that matter. The clinically important failure mode is a **missed abnormality**, not a false alarm — which is why sensitivity is weighted above accuracy throughout.

### Scope
**In scope:** offline inference on stored 12-lead records, five-class multi-label output, explainability overlay, honest evaluation dashboard.

**Explicitly out of scope** (say this if asked — knowing your boundaries earns marks):
- Real-time / streaming ECG from a device
- Beat-level segmentation or R-peak-by-R-peak classification
- Any claim of diagnostic accuracy or clinical deployment readiness
- Patient identity, demographics-based inference, or PHI handling

---

## 2. Data

### Dataset: PTB-XL (PhysioNet)

- Large public clinical 12-lead ECG dataset, 10 seconds per record, available at both 100 Hz and 500 Hz.
- Roughly 21,800 records from roughly 18,900 patients.
- Ships with `ptbxl_database.csv` (metadata + labels), `scp_statements.csv` (label ontology), and records in WFDB format under `records100/` and `records500/`.
- **Use the 100 Hz version.** 1000 samples × 12 leads per record. Trains fast, fits in memory, and is sufficient for superclass prediction.
- Check the current dataset version on PhysioNet when you download and record the exact version in the README.

### Labels — 5 diagnostic superclasses

Derived from the `scp_codes` column, mapped through `scp_statements.csv` using the `diagnostic_class` field:

| Code | Meaning |
|------|---------|
| `NORM` | Normal ECG |
| `MI` | Myocardial infarction |
| `STTC` | ST/T wave changes |
| `CD` | Conduction disturbance |
| `HYP` | Hypertrophy |

**This is multi-label, not multi-class.** A single ECG can carry more than one superclass. Use sigmoid outputs and `BCEWithLogitsLoss` — not softmax, not cross-entropy. Be ready to explain this distinction; it is a very likely viva question.

Records whose `scp_codes` map to no superclass (after filtering by the diagnostic likelihood threshold) are dropped. Log how many were dropped and why.

### Splits

PTB-XL provides a `strat_fold` column with 10 pre-computed, patient-stratified folds. Use them directly:

- **Train:** folds 1–8
- **Validation:** fold 9
- **Test:** fold 10

Do not build your own split. Using the official folds means your results are comparable to published baselines, and it removes any question of leakage. Write an assertion in the data module that verifies no `patient_id` appears in more than one split, and run it.

### Preprocessing

Keep it minimal and justifiable. Each step must have a reason you can state out loud.

1. Load WFDB record → array of shape `(1000, 12)`, transpose to `(12, 1000)`.
2. **Per-lead z-score normalisation** using mean and standard deviation computed **on the training split only**. Persist these statistics to disk and reuse them for validation, test, and inference. Computing them over the whole dataset is leakage.
3. Optional band-pass filter (0.5–40 Hz) to remove baseline wander and high-frequency noise. Implement it, make it a config flag, and report metrics with and without it. If it does not help, leave it off and say so — a negative result you measured is worth more than an unjustified step.
4. No resampling, no padding — every record is already a fixed 10 s.

### Augmentation (training only)

- Random time shift / crop (take a random 8 s window from the 10 s, then pad or resize back — or train on 8 s windows throughout)
- Random amplitude scaling, ×0.8 to ×1.2
- Gaussian noise, small sigma

Do **not** flip the signal in time or invert amplitude. Both destroy clinical meaning — ECG morphology is directional. If asked why you skipped standard augmentations, this is the answer.

### Cache

After first load, cache the full preprocessed arrays as a `.npy` memmap plus a label matrix. Loading 21k WFDB records on every epoch is unnecessarily slow.

---

## 3. Model

### Architecture: 1D residual CNN

```
Input (12, 1000)
  ↓
Stem:  Conv1d(12 → 64, kernel=7, stride=2) → BatchNorm → ReLU → MaxPool
  ↓
Residual stage 1:  2 × ResBlock(64),  stride 1
Residual stage 2:  2 × ResBlock(128), stride 2
Residual stage 3:  2 × ResBlock(256), stride 2
Residual stage 4:  2 × ResBlock(512), stride 2
  ↓
Global Average Pooling  ←  (keep this: Grad-CAM depends on it)
  ↓
Dropout(0.3) → Linear(512 → 5)
  ↓
Output: 5 logits (sigmoid applied at inference)
```

**ResBlock:** `Conv1d(k=5, pad=2) → BN → ReLU → Dropout → Conv1d(k=5, pad=2) → BN`, plus identity skip (1×1 conv on the skip path when channels or stride change), then ReLU.

Roughly 5–10M parameters. Trains on a free Colab T4 in well under an hour.

**Keep global average pooling before the classifier.** Flattening instead will break the Grad-CAM implementation in section 5.

### Baselines to beat (build these first, they take an hour total)

1. **Majority / prior baseline** — always predict the training set base rate. Establishes the floor.
2. **Classical baseline** — extract simple features (heart rate, RR interval statistics, per-lead signal statistics) → gradient boosting. 

Report both alongside the CNN. A deep model with no baseline to compare against is an unfinished experiment, and examiners notice.

### Training configuration

| Setting | Value |
|---|---|
| Loss | `BCEWithLogitsLoss`, with `pos_weight` per class to counter imbalance |
| Optimiser | AdamW, weight decay 1e-2 |
| LR | 1e-3, OneCycle or cosine schedule with warmup |
| Batch size | 64 (drop to 32 if memory-constrained) |
| Epochs | 30–50, early stopping on validation macro AUROC, patience 8 |
| Seed | Fixed and logged. Run the final config on 3 seeds and report mean ± std |
| Checkpoint | Save best-on-validation only |

Log every run: config, git commit hash, seed, per-epoch train/val loss and macro AUROC. A simple CSV or TensorBoard is fine — no need for a heavyweight tracking service.

### Expected performance

Published CNN baselines on PTB-XL superclasses land somewhere around **0.92–0.93 macro AUROC**. Treat that as a reference point, not a target to claim. Report what you actually measure. If you land at 0.89, report 0.89 and discuss the gap — that is a better presentation than an unexplained 0.95.

---

## 4. Evaluation

This section carries the most marks. Do not rush it.

### Metrics — compute all of these

**Per class** (NORM, MI, STTC, CD, HYP) and **macro-averaged**:
- AUROC
- **AUPRC (average precision)** — the headline metric for imbalanced multi-label. Lead with this.
- Sensitivity (recall) at the chosen operating threshold
- Specificity
- Precision / PPV
- F1

**Do not lead with accuracy.** With a dominant NORM class, accuracy is misleading and quoting it invites an easy criticism. If you show it, show it last and explain why it is a poor metric here.

### Threshold selection

Default 0.5 is arbitrary. Choose thresholds **on the validation set** by maximising F1 per class, and additionally report a **high-sensitivity operating point** (threshold tuned so sensitivity ≥ 0.90 for each abnormal class) with the resulting specificity.

Frame it clinically: for a screening tool, a false alarm costs a clinician thirty seconds of review; a missed MI costs considerably more. That asymmetry is the justification for the operating point you pick, and it is the strongest single thing you can say in the viva.

### Calibration

- Reliability diagram per class (predicted probability vs observed frequency, 10 bins)
- **Expected Calibration Error (ECE)**
- If calibration is poor, fit **temperature scaling on the validation set** and report ECE before and after

A model whose "0.8" genuinely means 80% is a meaningfully better tool, and almost no student project does this. It is a differentiator.

### Required plots (saved as PNG/SVG into `reports/figures/`)

1. Per-class ROC curves
2. Per-class precision–recall curves
3. Confusion matrix per class (binary, at the chosen threshold)
4. Reliability diagram, before and after calibration
5. Training curves (loss and macro AUROC vs epoch)
6. Class distribution / support bar chart

### Error analysis — do this, it is 15 minutes and it reads extremely well

Pull the 10 highest-confidence false negatives and 10 highest-confidence false positives. Plot them. Look for a pattern (noisy leads? a specific co-occurring label? short records?). Write 3–4 sentences of findings into the README and put one slide on it. Examiners consistently reward evidence that you looked at your own failures.

### Output artifact

Evaluation writes a single `reports/metrics.json` containing every number above, plus the git commit and seed. **The frontend reads this file.** It never computes or invents metrics itself.

---

## 5. Explainability

**Grad-CAM adapted to 1D.**

Implementation:
1. Register a forward hook on the final residual stage to capture activations `A` of shape `(C, L)`.
2. Backpropagate the logit for the target class to get gradients `∂y_c/∂A`.
3. Channel weights `α_c = GlobalAveragePool(gradients)`.
4. `CAM = ReLU(Σ_c α_c · A_c)` → shape `(L,)`.
5. Upsample `CAM` from `L` to 1000 via linear interpolation.
6. Normalise to [0, 1].

The frontend renders this as a shaded intensity band behind the waveform, aligned on the time axis.

**Be honest about what this shows.** Grad-CAM indicates which time regions the model's activations responded to. It is not a clinical explanation and not proof the model learned correct physiology. Say exactly that in the UI ("model attention — not a clinical rationale") and in the presentation. Overclaiming here is an easy way to lose marks; stating the limitation clearly is an easy way to gain them.

Sanity check to run once: confirm the CAM for a model with randomly initialised weights looks like noise, and that a trained model's CAM does not. If they look the same, your implementation is broken.

---

## 6. Backend API

**FastAPI**, Python 3.11+.

### Endpoints

```
GET  /api/health
     → { "status": "ok", "model_loaded": bool, "model_version": str }

GET  /api/records
     → List of held-out TEST-SET records available for demo.
       [ { "id": "12345", "duration_s": 10, "sampling_rate": 100 } ]
     NOTE: expose record IDs only. No patient metadata, no ground-truth
     labels in this response — the UI must not be able to leak the answer
     before prediction.

GET  /api/records/{id}/signal
     → { "leads": ["I","II",...], "sampling_rate": 100,
         "data": [[...], ...] }   # 12 × 1000 floats, raw (pre-normalisation)

POST /api/predict
     Body: either { "record_id": "12345" }
           or multipart upload of a .hea + .dat pair
     → {
         "predictions": [
           { "class": "NORM", "probability": 0.91, "flagged": false,
             "threshold": 0.53 },
           ...
         ],
         "attention": [ ... ],        # 1000 floats, normalised 0-1
         "attention_class": "MI",     # class the CAM was computed for
         "model_version": "v1.2.0",
         "inference_ms": 34
       }

GET  /api/metrics
     → Contents of reports/metrics.json, served verbatim.
```

### Rules

- Model is loaded **once** at startup, held in memory. Never load per request.
- Uploaded files: validate that the record is 12-lead and can be resampled to 1000 samples at 100 Hz. Reject with a clear 400 and a human-readable message otherwise.
- Cap upload size. Delete uploaded temp files after inference — do not retain them.
- **Ground-truth labels are never sent to the frontend before a prediction.** After prediction, a `/api/records/{id}/truth` endpoint may reveal them so the demo can show "model said X, actual was Y" — this is a strong demo moment precisely because the model could not have seen it.
- CORS restricted to the local dev frontend origin.

---

## 7. Frontend

**React + Vite + TypeScript + Tailwind CSS.** Charting: Recharts, or plain SVG/canvas for the waveform (canvas is smoother for 1000-point traces).

> Time-pressure fallback: Streamlit is acceptable and will still look decent, but you lose most of the layout control. Only fall back if you are behind by day 15.

### Design direction

Restrained, clinical, confident. Think medical software, not a consumer dashboard.

- **Palette:** near-white background (`#FAFAFA`), white cards, `#1A1A1A` text, one accent — a muted clinical teal (`#0F766E`). Semantic colours used *only* for results: amber `#D97706` for flagged, red `#DC2626` for high-confidence abnormal, slate for normal.
- **Type:** Inter or IBM Plex Sans. One family. Weights 400/500/600 only. Numbers in tabular figures.
- **Spacing:** generous. 8px base scale. Let things breathe.
- **Motion:** almost none. A 150ms fade on result load. Nothing else. No spinners that bounce, no animated counters.
- **No gauges, no gradients, no glassmorphism, no dark-mode toggle.** Restraint is the aesthetic.

### Screen 1 — Upload / Select

- Drag-and-drop zone for `.hea` + `.dat`
- "or select a held-out test record" → searchable list of record IDs
- One line of context: *Records listed are from the held-out test set. The model has never seen them during training.*
- Single primary action button. Nothing else on the screen.

### Screen 2 — Result

Top: **the ECG waveform.** Real signal, plotted properly.
- Default view: lead II full width (the standard rhythm lead)
- A lead selector for all 12; optionally a 12-lead grid view
- Proper ECG axes — time in seconds on x, mV on y, with gridlines at clinical intervals
- The attention band rendered *behind* the trace as a soft intensity shade, with a small "model attention — not a clinical rationale" caption

Below: **five result rows**, one per class. Each row:
- Class name and plain-English label (`STTC` → "ST/T change")
- Probability as a number and a slim horizontal bar
- A flag state (Flagged / Not flagged) based on the calibrated threshold
- The threshold in use, shown small

Then a single summary line: *2 of 5 categories flagged at the high-sensitivity operating point.*

Footer, always visible: **Screening aid. Not a diagnostic device. Not for clinical use.**

Optional "Reveal ground truth" button that appears only after prediction — genuinely persuasive in a live demo.

### Screen 3 — Model Performance

This is your marks page. Everything here is read from `/api/metrics`.

- Header block: dataset name and version, number of test records, number of test patients, model version, git commit, date evaluated
- Metrics table: per class × (AUROC, AUPRC, Sensitivity, Specificity, F1), with a macro-average row
- The plots from section 4, laid out in a clean grid
- The calibration section, before/after, with the ECE numbers stated
- A short, plainly written **Limitations** block:
  - Single public dataset; single source institution; results may not generalise
  - Model has not been validated on any prospective or external cohort
  - Attention maps are not clinical explanations
  - Performance on rare classes is limited by support size

Writing your own limitations down is not weakness. It is the clearest signal that you understood what you built, and it pre-empts half the questions you would otherwise get asked.

---

## 8. Repository structure

```
cardioscreen/
├── README.md
├── PROJECT_SPEC.md              ← this file
├── requirements.txt / pyproject.toml
├── .env.example
├── config/
│   └── default.yaml             # all hyperparameters, no magic numbers in code
├── data/
│   ├── raw/                     # PTB-XL download (gitignored)
│   └── cache/                   # preprocessed .npy (gitignored)
├── src/
│   ├── data/
│   │   ├── download.py          # fetch + verify PTB-XL
│   │   ├── labels.py            # scp_codes → 5 superclasses
│   │   ├── dataset.py           # torch Dataset, caching, augmentation
│   │   └── splits.py            # official folds + leakage assertion
│   ├── models/
│   │   ├── resnet1d.py
│   │   └── baselines.py         # prior + gradient-boosting baselines
│   ├── training/
│   │   ├── train.py
│   │   └── calibrate.py         # temperature scaling
│   ├── evaluation/
│   │   ├── metrics.py           # writes reports/metrics.json
│   │   ├── plots.py
│   │   └── error_analysis.py
│   ├── explain/
│   │   └── gradcam1d.py
│   └── api/
│       ├── main.py              # FastAPI app
│       └── schemas.py           # pydantic models
├── frontend/                    # Vite + React + TS
│   └── src/
│       ├── components/
│       ├── pages/
│       └── lib/api.ts
├── reports/
│   ├── metrics.json
│   └── figures/
├── checkpoints/                 # gitignored
├── tests/
└── scripts/
    ├── run_training.sh
    └── run_eval.sh
```

### Tests worth writing (do not skip these three)

1. `test_no_patient_leakage` — assert the patient ID sets of train/val/test are pairwise disjoint
2. `test_normalisation_stats_from_train_only` — assert the persisted stats file was computed on the train fold
3. `test_gradcam_shape_and_range` — CAM output is length 1000 and bounded in [0, 1]

These three cover the failure modes that would actually invalidate your results.

---

## 9. Three-week build order

### Week 1 — Data and baseline (days 1–7)

| Day | Deliverable |
|---|---|
| 1 | Repo scaffold, env, PTB-XL downloaded and verified, one record loaded and plotted |
| 2 | Label mapping to 5 superclasses; class distribution table; drop-count logged |
| 3 | Official fold splits wired in; **leakage test written and passing**; normalisation stats from train only |
| 4 | Torch Dataset + caching + DataLoader; augmentation implemented behind config flags |
| 5 | Prior baseline and gradient-boosting baseline; metrics computed for both |
| 6 | ResNet1D implemented; overfit deliberately on 50 samples to prove the training loop works |
| 7 | First full training run end to end; validation macro AUROC recorded |

**Gate:** by end of day 7 you have a trained model with real validation numbers, and the leakage test passes. If you are behind here, cut augmentation and the band-pass filter — not evaluation.

### Week 2 — Model, evaluation, explainability (days 8–14)

| Day | Deliverable |
|---|---|
| 8 | Hyperparameter tuning on validation; logging in place |
| 9 | Per-class thresholds selected on validation; high-sensitivity operating point defined |
| 10 | Full metrics module → `reports/metrics.json`; all six plots generated |
| 11 | Temperature scaling; reliability diagrams before/after; ECE reported |
| 12 | Grad-CAM 1D implemented; random-weights sanity check passes |
| 13 | Error analysis on top false positives/negatives; findings written into README |
| 14 | Final training run across 3 seeds; **test set evaluated once**; results frozen |

**Gate:** by end of day 14, the model is done. Do not touch it again. Everything after this is product and presentation.

### Week 3 — Product and presentation (days 15–21)

| Day | Deliverable |
|---|---|
| 15 | FastAPI backend: all endpoints working, model loaded at startup, tested with curl |
| 16 | Frontend scaffold, design tokens, routing; Screen 1 (upload/select) complete |
| 17 | Screen 2: waveform rendering with correct ECG axes and lead selector |
| 18 | Screen 2: prediction rows, attention overlay, reveal-ground-truth. **Code freeze.** |
| 19 | Screen 3: metrics page wired to `/api/metrics`, all plots, limitations block |
| 20 | README, setup instructions verified from a clean clone, slide deck drafted |
| 21 | Demo rehearsal ×3, offline fallback prepared, viva answers practised |

### Buffer policy
If you slip, cut in this order: 12-lead grid view → upload flow (keep record selection only) → augmentation → band-pass filter. **Never cut:** the leakage test, calibration, the metrics page, or the limitations block. Those are the marks.

---

## 10. Definition of done

- [ ] A clean clone plus documented setup runs training, evaluation, and the app
- [ ] Leakage test passes and is visible in the repo
- [ ] `reports/metrics.json` is generated by a script, and the UI reads only from it
- [ ] Test set was evaluated exactly once, on the frozen model
- [ ] Baselines reported alongside the CNN
- [ ] Calibration reported before and after temperature scaling
- [ ] Grad-CAM verified against the random-weights sanity check
- [ ] Every screen renders correctly with no data and with real data
- [ ] Disclaimer visible on every screen
- [ ] README documents dataset version, licence, and citation
- [ ] Demo runs fully offline (no network dependency during presentation)

---

## 11. Presentation and viva preparation

### Slide order (10–12 slides)
1. The problem — ECG volume, review burden, cost of a missed abnormality
2. What the tool does — one screenshot of Screen 2
3. Data — PTB-XL, size, label set, class imbalance chart
4. Why patient-level splits matter (this slide alone signals rigour)
5. Model architecture diagram
6. Baselines vs model
7. Results — lead with AUPRC and sensitivity, not accuracy
8. Calibration — reliability diagram before/after
9. Explainability — one Grad-CAM example, with the honest caveat
10. Error analysis — what it gets wrong and your hypothesis why
11. Limitations and what real deployment would require
12. Live demo

### Questions you will be asked — have these ready

**"Why not accuracy?"** — Dominant normal class. A model predicting "normal" always would score high and be clinically useless. AUPRC and per-class sensitivity reflect what actually matters.

**"Why multi-label instead of multi-class?"** — A single ECG can exhibit more than one abnormality simultaneously. Softmax would force a false mutual exclusivity. Hence sigmoid + BCE.

**"How do you know it isn't memorising?"** — Official patient-stratified folds, enforced by an automated test asserting disjoint patient sets. Test set evaluated once, on the frozen model.

**"Is this deployable?"** — No, and here is exactly why: single-source dataset, no external or prospective validation, no regulatory pathway, attention maps are not clinical rationale. It is a research prototype demonstrating a screening approach.

**"What does the attention map actually mean?"** — Which time regions produced the strongest activations for that class. It is a debugging and trust aid, not a clinical explanation. Do not overclaim this.

**"What would you do with three more months?"** — External validation on a second dataset (e.g. a different institution's cohort), per-lead ablation to see which leads carry the signal, a proper prospective reader study comparing triage order with and without the tool.

**"Why did you choose this architecture?"** — Residual 1D CNN is the standard strong baseline for fixed-length ECG classification, trains in under an hour on free hardware, and retains global average pooling which Grad-CAM requires. I compared it against a classical feature-based baseline; the numbers are on slide 6.

### Demo safety
- Run entirely on localhost, no internet dependency
- Pre-select 3 records: one clear normal, one clear abnormal, one the model gets wrong
- **Show the one it gets wrong.** Volunteering a failure and explaining it is far more convincing than three clean successes, and it defuses the "does it ever fail?" question before it is asked
- Record a 90-second screen capture as a fallback in case the live demo breaks

---

## 12. Licensing and attribution

- PTB-XL is released under an open licence on PhysioNet — read it, and cite the dataset paper (Wagner et al., *Scientific Data*, 2020) plus PhysioNet in your README and slides.
- Record the exact dataset version used.
- Do not commit the dataset itself to git.

---

## 13. Common failure modes to avoid

| Mistake | Why it costs you |
|---|---|
| Splitting by record, not patient | Inflated scores; invalidates every result |
| Normalisation stats over full dataset | Subtle leakage; a sharp examiner will catch it |
| Reporting accuracy as headline | Meaningless under class imbalance |
| Tuning on the test set | Your test number no longer means anything |
| Softmax on a multi-label problem | Conceptually wrong; a likely viva question |
| Animated fake "live monitoring" | Reads as theatre and undermines everything real you built |
| Claiming clinical utility | Invites the harshest possible line of questioning |
| Hiding poor metrics | Examiners look for the numbers you did not show |
| Building the UI before the model works | Week 3 panic with nothing real to display |

---

## 14. Stretch goals (only after section 10 is fully checked)

1. **AFib detection** — PTB-XL carries rhythm statements separately from diagnostic ones. Add atrial fibrillation as a sixth output. Rarer, harder, and a good extension to mention even if you only scope it.
2. **Per-lead ablation** — retrain on single leads to show which carry the most signal. Cheap to run, genuinely interesting, one strong slide.
3. **External validation** — evaluate the frozen model on a second public ECG dataset with no retraining. The single most scientifically valuable addition available to you.
4. **Model card** — a one-page structured summary of intended use, training data, metrics, and limitations. Short, professional, and increasingly standard practice.
