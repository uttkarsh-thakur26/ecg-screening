# Error analysis (validation split, fold 9)

The 10 most confident false negatives and the 10 most confident false positives of the release model
(v1.0, run `20261017-215633_no-amp-scale`), at the high-sensitivity operating point, on calibrated
probabilities. Every number below comes from `reports/error_analysis_val.json`, written by
`python -m src.evaluation.error_analysis` at commit `b67124b` (clean tree). Fold 10 was not opened.

Figures: [false negatives](figures/error_analysis_fn_val.png), [false positives](figures/error_analysis_fp_val.png).

## Findings

The confident misses are not noisy recordings (1 of the 10 has any PTB-XL signal-quality note, against
31% of all validation records) but uncertain labels and less common sub-types: all three missed MIs rest
on a single MI statement of likelihood 15, and across the split such MI labels are missed 24% of the time
(27/114) against 5% (14/283) at likelihood 100. Within a class the model has learned the common sub-type
better than the rare ones: LVH is missed in 5% (11/210) of HYP positives against 21% to 40% for atrial
enlargement (LAO/LAE 9/43, RAO/RAE 4/10), all four HYP misses in the top 10 are atrial or right-ventricular,
and non-specific intraventricular conduction delay (IVCD) gives 27 of the 49 CD misses. The confident false
alarms are mostly ST/T flags (7 of 10), and 6 of those 7 records carry LVH or MI statements (3 of them
acute-injury `INJ*` codes), conditions that themselves change the ST segment and T wave; across the split
every abnormal class is flagged 1.8 to 3.5 times as often on records with another abnormality as on records
labelled NORM (STTC on 38% of MI records without STTC, against 11%). So the main error sources are label
uncertainty and overlap between PTB-XL's superclasses rather than signal quality, and a clinician reviewing
a confident false alarm would usually find a real abnormality that carries a different label.

## How the examples were chosen

- Decisions: the frozen high-sensitivity thresholds from `reports/thresholds.json` (MI 0.181, STTC 0.219,
  CD 0.132, HYP 0.073), applied to the frozen calibrated probabilities. NORM has no threshold at this
  operating point, so it has no errors here.
- Confidence: the calibrated probability of the class. A false negative is more confident the lower it
  is, a false positive the higher. Calibration puts all classes on one scale, so errors of different
  classes are ranked together.
- One entry per record: a record wrong for several classes is ranked by its most confident error and
  lists the others, so the 10 examples of each kind are 10 different ECGs.
- Errors at this operating point on the whole split: MI 54 FN / 319 FP (540 positives), STTC 52 / 312
  (528), CD 49 / 363 (495), HYP 26 / 524 (268).

## Most confident false negatives

| # | ecg_id | Error (p vs threshold) | Labels | Flagged | Age, sex | Diagnostic statements (likelihood) | Quality notes |
|---|---|---|---|---|---|---|---|
| 1 | 11671 | MI 0.003 vs 0.181 | MI | - | 51, female | AMI 15 | none |
| 2 | 6058 | STTC 0.003 vs 0.219 | STTC | - | 87, male | NST_ 100 | none |
| 3 | 15995 | HYP 0.004 vs 0.073 | CD, HYP | CD | 49, male | IRBBB 100, RVH 35 | none |
| 4 | 18625 | MI 0.005 vs 0.181 | MI | - | 72, female | IMI 15 | baseline_drift (V1); static_noise (I-AVR) |
| 5 | 10045 | HYP 0.005 vs 0.073; CD 0.012 vs 0.132 | CD, HYP | - | 55, male | IVCD 100, LAO/LAE 100 | none |
| 6 | 15775 | HYP 0.006 vs 0.073 | NORM, HYP | - | 44, female | NORM 100, RAO/RAE 50 | none |
| 7 | 11300 | CD 0.007 vs 0.132 | CD | - | 53, male | WPW 100 | none |
| 8 | 4233 | HYP 0.008 vs 0.073 | MI, HYP | MI | 59, male | IMI 15, LAO/LAE 50 | none |
| 9 | 8233 | CD 0.009 vs 0.132 | CD | MI | 61, male | IVCD 100 | none |
| 10 | 14110 | MI 0.009 vs 0.181 | MI | - | 64, male | IMI 15 | none |

## Most confident false positives

| # | ecg_id | Error (p vs threshold) | Labels | Flagged | Age, sex | Diagnostic statements (likelihood) | Quality notes |
|---|---|---|---|---|---|---|---|
| 1 | 3642 | STTC 0.992 vs 0.219 | HYP | STTC, HYP | 60, male | LVH 50 | none |
| 2 | 1451 | STTC 0.983 vs 0.219 | HYP | STTC, HYP | 69, female | LVH 100 | none |
| 3 | 5006 | CD 0.981 vs 0.132 | STTC | STTC, CD | 68, male | NDT 100 | static_noise (I-AVR) |
| 4 | 6073 | MI 0.974 vs 0.181 | STTC, CD | MI, CD | 74, male | ISCAL 100, LAFB 100 | none |
| 5 | 414 | STTC 0.970 vs 0.219 | MI, HYP | MI, STTC, HYP | 43, male | INJAS 100, INJAL 100, LVH 50 | none |
| 6 | 10306 | STTC 0.966 vs 0.219 | CD | STTC, CD | 73, male | IRBBB 100 | none |
| 7 | 12615 | STTC 0.959 vs 0.219; HYP 0.642 vs 0.073 | MI | MI, STTC, HYP | 72, female | INJIL 100 | none |
| 8 | 4812 | STTC 0.958 vs 0.219; HYP 0.343 vs 0.073; CD 0.261 vs 0.132 | MI | MI, STTC, CD, HYP | 49, male | AMI 100, INJAL 100 | none |
| 9 | 5863 | CD 0.944 vs 0.132; HYP 0.204 vs 0.073 | MI | MI, CD, HYP | 70, male | ALMI 100, ASMI 100 | static_noise (I-AVR) |
| 10 | 7653 | STTC 0.944 vs 0.219; HYP 0.901 vs 0.073 | MI | MI, STTC, HYP | 74, male | ASMI 100, IMI 100 | extra_beats (1ES) |

## The examples against all validation records

| Annotation | False negatives | False positives | All validation records |
|---|---|---|---|
| baseline_drift noted | 1 of 10 (10%) | 0 of 10 (0%) | 226 of 2146 (11%) |
| static_noise noted | 1 of 10 (10%) | 2 of 10 (20%) | 321 of 2146 (15%) |
| burst_noise noted | 0 of 10 (0%) | 0 of 10 (0%) | 47 of 2146 (2%) |
| electrodes_problems noted | 0 of 10 (0%) | 0 of 10 (0%) | 4 of 2146 (0%) |
| extra_beats noted | 0 of 10 (0%) | 1 of 10 (10%) | 187 of 2146 (9%) |
| pacemaker noted | 0 of 10 (0%) | 0 of 10 (0%) | 2 of 2146 (0%) |
| any quality note | 1 of 10 (10%) | 3 of 10 (30%) | 667 of 2146 (31%) |
| validated_by_human | 10 of 10 (100%) | 10 of 10 (100%) | 2146 of 2146 (100%) |
| second_opinion | 0 of 10 (0%) | 1 of 10 (10%) | 68 of 2146 (3%) |
| initial_autogenerated_report | 0 of 10 (0%) | 1 of 10 (10%) | 197 of 2146 (9%) |
| mean superclass labels per record | 1.40 | 1.20 | 1.30 |

## Whether the patterns hold over the whole split

Ten examples per kind are anecdotes, so each pattern was checked on every validation record at the same
operating point.

Missed positives by the class's SCP statement (a record with several statements counts under each):

| Class | Statement: missed/positives |
|---|---|
| MI | IMI 25/266 (9%), ASMI 8/236 (3%), ILMI 2/47 (4%), AMI 5/36 (14%), ALMI 0/29 (0%), INJAS 2/21 (10%), LMI 12/20 (60%), INJAL 1/15 (7%), IPLMI 1/5 (20%), IPMI 0/4 (0%), INJIL 0/2 (0%), INJLA 0/2 (0%), INJIN 1/2 (50%), PMI 0/2 (0%) |
| STTC | NDT 17/182 (9%), ISC_ 3/125 (2%), NST_ 7/75 (9%), ISCAL 6/66 (9%), ISCIN 4/21 (19%), DIG 1/18 (6%), ISCIL 1/18 (6%), ISCAS 1/17 (6%), ISCLA 5/14 (36%), LNGQT 1/12 (8%), ANEUR 7/10 (70%), EL 0/10 (0%), ISCAN 0/5 (0%) |
| CD | LAFB 4/163 (2%), IRBBB 8/112 (7%), 1AVB 3/80 (4%), IVCD 27/78 (35%), CRBBB 0/55 (0%), CLBBB 1/54 (2%), LPFB 3/18 (17%), ILBBB 1/7 (14%), WPW 2/7 (29%), 3AVB 1/2 (50%), 2AVB 1/1 (100%) |
| HYP | LVH 11/210 (5%), LAO/LAE 9/43 (21%), RVH 2/12 (17%), RAO/RAE 4/10 (40%), SEHYP 1/3 (33%) |

Missed positives by the highest likelihood of the class's statements in the record:

| Class | Likelihood: missed/positives |
|---|---|
| MI | 15: 27/114 (24%), 35: 1/14 (7%), 50: 9/106 (8%), 80: 3/23 (13%), 100: 14/283 (5%) |
| STTC | 0: 4/13 (31%), 15: 1/1 (100%), 35: 0/4 (0%), 50: 9/48 (19%), 80: 0/7 (0%), 100: 38/455 (8%) |
| CD | 50: 2/7 (29%), 80: 1/3 (33%), 100: 46/485 (9%) |
| HYP | 15: 3/32 (9%), 35: 3/20 (15%), 50: 8/74 (11%), 80: 1/8 (12%), 100: 11/134 (8%) |

Flagged negatives (false alarms), by another label the record carries:

| Class | with NORM | with MI | with STTC | with CD | with HYP | all negatives |
|---|---|---|---|---|---|---|
| MI | 104/955 (11%) | - | 139/383 (36%) | 108/310 (35%) | 45/179 (25%) | 319/1606 (20%) |
| STTC | 104/950 (11%) | 152/395 (38%) | - | 79/386 (20%) | 42/136 (31%) | 312/1618 (19%) |
| CD | 111/919 (12%) | 149/355 (42%) | 148/419 (35%) | - | 54/181 (30%) | 363/1651 (22%) |
| HYP | 166/954 (17%) | 140/451 (31%) | 190/396 (48%) | 176/408 (43%) | - | 524/1878 (28%) |

## Caveats

- The thresholds were chosen on these same records, so the error counts are those of the fitted
  operating point; the unbiased counts come from the one-time test evaluation.
- The likelihood-15 effect is clear for MI only; for HYP the miss rate barely changes with likelihood
  (8% to 15%), so there the sub-type matters more than the certainty.
- Several sub-types have few positives (RAO/RAE 10, RVH 12, WPW 7), so their miss rates are rough.
- "Real but differently labelled abnormality" is a reading of PTB-XL's own statements, not a new
  clinical review of the tracings.
