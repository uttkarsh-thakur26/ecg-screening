# Demo script

About five minutes: three held-out test records (a clear normal, a correct abnormal, and one the model
gets wrong), then one upload, then the performance screen. Every record below is from the test set
(fold 10), which the model never saw during training or tuning. The probabilities quoted are what the
release model returns for these records; they are the same on the GPU and on the CPU to three decimals.

## Before the demo

Run these at least 10 minutes before you present, then switch the network off (airplane mode): the
demo needs no internet.

PowerShell window 1, in the repository folder:

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.api.main
```

Wait for `Application startup complete`. PowerShell window 2:

```powershell
cd frontend
npm run dev
```

Open **http://localhost:5173** (exactly this address: the API accepts only this origin). Click
through once: Screen an ECG, record 9, Model performance. If any step fails, see
[If something fails](#if-something-fails).

## 1. Record 9: a normal ECG, nothing flagged

**Click:** Screen an ECG → type `9` in *Search by record ID* → click record **9** → **Screen record 9**.
Point at the line under the ECG, then click **Reveal ground truth**.

**Say:** "This is a held-out test record the model has never seen. It scores NORM at 0.949 and every
abnormal category far below its threshold, so nothing is flagged, and because nothing is flagged we
deliberately show no attention band." After the reveal: "The dataset labels it normal, so all four
flags agree with the label."

## 2. Record 63: myocardial infarction, correctly flagged

**Click:** Screen another ECG → search `63` → record **63** → **Screen record 63**. Point at the shaded
band, then **Reveal ground truth**.

**Say:** "MI comes out at 0.566 against a threshold of 0.181, so it is flagged; the threshold is low on
purpose, because we chose it on the validation fold to catch about nine in ten MIs. The band shows
where in these ten seconds the model's evidence for MI was strongest; that is model attention, not a
clinical reason, and the label confirms MI."

## 3. Record 38: a false alarm, and why we accept it

**Click:** Screen another ECG → search `38` → record **38** → **Screen record 38** →
**Reveal ground truth**. Point at *flagged, not in labels* in the CD row.

**Say:** "Here the model flags a conduction disturbance (CD at 0.399 against a threshold of 0.132) on an
ECG the dataset labels normal: a false alarm. That is the price of the high-sensitivity operating point;
in a screening setting a false alarm costs a second look, while a missed abnormality can cost much
more."

The test-set numbers behind that trade-off (from `reports/metrics.json`; this paragraph is written by
`python -m src.evaluation.doc_tables` and checked by a test):

<!-- metrics:cd-specificity:start -->
On the test set, CD specificity at the high-sensitivity point is 0.792 [0.772, 0.812]: about 2 in 10 records without CD still get a CD flag (346 of 1,662). In exchange, CD sensitivity is 0.889 [0.861, 0.916] (441 of 496 CD records flagged).
<!-- metrics:cd-specificity:end -->

## 4. An upload: a raw WFDB record with two abnormalities

**Click:** Screen another ECG → drag the two files of test record 516 onto the drop zone (or
*browse your files*):

```text
data\raw\records100\00000\00516_lr.hea
data\raw\records100\00000\00516_lr.dat
```

→ **Screen the uploaded record**.

**Say:** "This is the same path a hospital file would take: the API reads the raw WFDB header and
samples, checks them and runs the model; it gives exactly the probabilities that selecting record 516
gives. It flags ST/T change (0.996) and hypertrophy (0.998) at once, which is why every category has
its own sigmoid output instead of one softmax: an ECG can have several abnormalities, and PTB-XL labels
this one with both."

Uploads have no *Reveal ground truth* button: the API keeps nothing about an uploaded file.

## 5. Model performance (optional, 30 seconds)

**Click:** Model performance.

**Say:** "These numbers are the one and only evaluation on the held-out test fold, run after a
pre-registered plan, and the screen reads every one of them straight from the evaluation's output
file. The strip at the top gives the three that matter most: how well the model ranks records, how many
abnormal cases the screening point catches, and how trustworthy the probabilities are; the
limitations are at the bottom."

Read the three values off the headline strip as you say this rather than from memory.

## If something fails

| Problem | What to do |
|---|---|
| The API stops with `api.device is 'cuda' but torch cannot see a CUDA GPU` | Set `api.device: cpu` in `config/default.yaml` and start the API again. The CPU gives the same flags for every record above; a prediction takes a few tenths of a second instead of tens of milliseconds |
| The page says *The API is not responding* | Check window 1. If the API crashed, start it again with `python -m src.api.main`, wait for `Application startup complete`, then click **Try again** |
| The API says port 8000 is in use | An old API is still running: close its window (or end the `python` process), then start it again |
| `npm run dev` says port 5173 is in use | Close the other dev server; the app must run on port 5173, the only origin the API accepts |
| The page loads but every request fails | Open exactly `http://localhost:5173`, not `127.0.0.1:5173` |
| Nothing works in time | Play the recorded 90-second screen capture of this script (record it beforehand), then show `docs/images/result-screen.png` and the figures in `reports/figures/` (`*_test.png`) with `reports/TEST_RUN.md` for the numbers |
