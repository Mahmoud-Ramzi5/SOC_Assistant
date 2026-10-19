# SOC Assistant

`soc_assistant.py` is a Streamlit web app for SOC analysts. It classifies network-flow logs with the team's trained model and explains each result.

For every flow in a log, it reports:

1. **Benign or attack.** Whether the flow is BENIGN or an ATTACK, based on the model's attack probability P(ATTACK) and an alert threshold you set.
2. **For attacks:**
   - **Attack category**, such as DDoS, DoS, Reconnaissance, Brute Force, Web Attack, Botnet or Exploit, with its confidence.
   - **Severity**: LOW, MEDIUM, HIGH or CRITICAL, from a 0–100 score.
   - **Evidence**: which feature values are unusual compared with benign traffic.
   - **MITRE ATT&CK technique** for the attack category, with a link.
   - **Response playbook** that best matches the alert.

## Requirements

- Python 3.10 or newer
- The packages below. **scikit-learn must be version 1.6.1**, the version the model was trained with. With any other version, the app shows a warning, and loading may fail or predictions may differ.

```bash
pip install streamlit "scikit-learn==1.6.1" pandas numpy joblib
```

Using a virtual environment keeps this scikit-learn version separate from your other projects:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install streamlit "scikit-learn==1.6.1" pandas numpy joblib
```

## Model files

The app looks for models in the `models/` folder next to `soc_assistant.py`:

| File | Required | Purpose |
|---|---|---|
| `models/soc_model.joblib` | Yes | Main model (benign vs attack, or both stages in one file) |
| `models/attack_type_model.joblib` | No | Separate attack-category model |

To use model files stored somewhere else, either set environment variables or enter the path or upload the file in the app's sidebar.

```powershell
$env:SOC_MODEL_PATH = "D:\path\to\soc_model.joblib"
$env:SOC_FAMILY_MODEL_PATH = "D:\path\to\attack_type_model.joblib"
```

The app accepts these kinds of model file:

- A fitted scikit-learn estimator or Pipeline that uses the 7 features below. It can be binary (BENIGN vs ATTACK) or multi-class (BENIGN plus one class per attack category).
- A dict holding both stages, for example `{"binary_model": ..., "family_model": ...}`.
- A bundle dict: `{"model": ..., "features": [...], "threshold": 0.5, "calibrator": ..., "class_names": ...}`. Per-dataset detectors such as `improvement2_detectors.joblib` from the notebook also work.

> **Security:** loading a `.joblib` file runs code. Only load model files from people you trust.

## Running the app

From this folder:

```bash
streamlit run soc_assistant.py
```

Streamlit opens the app in your browser at http://localhost:8501.

## Using the app

### Sidebar

1. **Model.** The default model loads automatically. You can upload a different one or enter a path. You can also load an attack-category model, and a *benign reference*: known-benign flows used to decide which values count as unusual. Without a benign reference, the app uses the flows that the model classified as BENIGN in your log, provided there are at least 30 of them.
2. **Settings.** The **alert threshold** slider sets the P(ATTACK) value above which a flow becomes an alert.

### Tabs

- **📄 Analyze logs.** Upload a log file, enter its path, or paste logs. The app shows:
  - row and alert counts for each severity level
  - a results table (attacks only, or all rows)
  - a **Download results (CSV)** button
  - a details view for any row: verdict, evidence, ATT&CK technique and playbook steps
- **✍️ Single log entry.** Enter the 7 feature values for one flow and classify it.
- **📘 Playbooks.** Lists all response playbooks and their steps.

## Log formats

The app accepts logs in these formats:

- **CSV**
- **JSON**: a list of objects, a single object, an object that holds a list, nested fields, or JSON Lines
- **Plain text**, one flow per line, either as `key=value` / `key: value` pairs or as the 7 values in feature order

Logs can use the 7 harmonized features directly. Raw **CIC-IDS2017** or **UNSW-NB15** columns are converted automatically. Logs can have up to 200,000 rows.

The 7 harmonized features, in order:

| # | Feature | Meaning |
|---|---|---|
| 1 | `flow_duration_seconds` | Flow duration in seconds |
| 2 | `src_packets` | Packets sent by the source |
| 3 | `dst_packets` | Packets sent by the destination |
| 4 | `src_bytes` | Bytes sent by the source |
| 5 | `dst_bytes` | Bytes sent by the destination |
| 6 | `src_mean_packet_bytes` | Average source packet size |
| 7 | `dst_mean_packet_bytes` | Average destination packet size |

Example lines for the paste box:

```text
flow_duration_seconds=0.52 src_packets=12 dst_packets=10 src_bytes=1500 dst_bytes=9000 src_mean_packet_bytes=125 dst_mean_packet_bytes=900
0.01, 2, 0, 80, 0, 40, 0
```

Label columns in a log, such as `Label`, `attack_cat` or `is_attack`, are never given to the model. Logs that contain the true labels can therefore be used to check the model's results.

## How severity is calculated

```
severity = 100 × (0.5 × confidence + 0.3 × impact + 0.2 × evidence)
```

- **confidence**: the model's probability for the alert
- **impact**: a fixed weight for the attack category, from 0.30 for Reconnaissance to 1.00 for Worm
- **evidence**: how unusual the flow's values are compared with benign traffic

| Score | Level |
|---|---|
| 80 and above | 🔴 CRITICAL |
| 60–79 | 🟠 HIGH |
| 40–59 | 🟡 MEDIUM |
| below 40 | 🟢 LOW |

A model bundle can include its own weights and thresholds, which replace these defaults.

## Testing with the test-case suite

`test_cases.py` builds 10 benchmark flows: 6 synthetic and 4 taken from CIC-IDS2017 and UNSW-NB15. It saves them to `artifacts/test_cases/test_cases.csv`, `test_cases.json` and `test_cases.txt`.

```bash
python test_cases.py
```

The script downloads both datasets from Hugging Face, so it also needs `datasets`, `huggingface_hub` and `pyarrow`. Once it has run, upload `artifacts/test_cases/test_cases.csv` in the **Analyze logs** tab and compare the app's verdicts with the `is_attack` column.

## Troubleshooting

| Problem | Fix |
|---|---|
| Warning that the scikit-learn version differs from 1.6.1 | `pip install "scikit-learn==1.6.1"` |
| "Start here: put the team's model file…" or "File not found" | Put the model at `models/soc_model.joblib`, set `SOC_MODEL_PATH`, or upload the model in the sidebar |
| Rows marked **UNSCORED** | The model has a separate detector for each dataset, and these rows have no `dataset_source` column (or no raw CIC/UNSW columns) to show which dataset they belong to |
| `streamlit` not recognized as a command | Run `python -m streamlit run soc_assistant.py` |
