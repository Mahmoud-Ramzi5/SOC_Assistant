"""SOC Assistant: classify network-flow logs with your team's joblib model and explain every result.

Stage 1: BENIGN or ATTACK (custom threshold slider on P(ATTACK)).
Stage 2: for attacks, the attack category (family) with its confidence, a severity level
(LOW / MEDIUM / HIGH / CRITICAL), evidence, the MITRE ATT&CK technique and the most similar playbook.

Setup (scikit-learn must match the training version, 1.6.1):

    pip install streamlit "scikit-learn==1.6.1" pandas numpy joblib
    mkdir models                      # next to this file
    copy the team's file to models/soc_model.joblib
    streamlit run soc_assistant.py

The app loads models/soc_model.joblib automatically (or the path in the SOC_MODEL_PATH environment
variable); you can also upload a file or type a path in the sidebar.

What the model file can be (the app detects which):
  1. A fitted scikit-learn estimator or Pipeline on the 7 FEATURES: binary (BENIGN vs ATTACK) or
     multi-class (BENIGN plus one class per attack category).
  2. Both stages in one dict, e.g. {"binary_model": ..., "family_model": ...}
     (also accepted: "detector"/"detection_model" and "category_model"/"attack_type_model"/"classifier").
  3. A dict bundle: {"model": estimator, "features": [...], "threshold": 0.5, "calibrator": optional,
     "class_names": optional}, or per-dataset detectors like the notebook's `improvement2_detectors.joblib`.
A separate attack-category model can also be loaded in the sidebar (default: models/attack_type_model.joblib).

Logs: CSV, JSON (list, single object, object holding a list, nested fields, JSON lines) or plain text
(one flow per line, as key=value / key: value pairs, or the 7 feature values in FEATURES order).
Raw CIC-IDS2017 or UNSW-NB15 columns are converted to the 7 features automatically; label columns in
the logs are never given to the model.

Security: loading a joblib file runs code. Only load model files from people you trust.
"""
from __future__ import annotations

import io
import json
import os
import re
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# =====================================================================================================
# Knowledge shared with the notebook (Day 3): attack families, impact, MITRE ATT&CK, playbooks, severity
# =====================================================================================================
HARMONIZED_FEATURES = [
    "flow_duration_seconds", "src_packets", "dst_packets", "src_bytes", "dst_bytes",
    "src_mean_packet_bytes", "dst_mean_packet_bytes",
]

FAMILY_RULES = [  # (normalized substring, family), checked in order: the first match wins
    ("benign", "BENIGN"), ("normal", "BENIGN"),
    ("ddos", "DDoS"), ("dos", "DoS"),
    ("portscan", "Reconnaissance"), ("reconnaissance", "Reconnaissance"),
    ("patator", "Brute Force"), ("bruteforce", "Brute Force"),
    ("sqlinjection", "Web Attack"), ("xss", "Web Attack"), ("webattack", "Web Attack"),
    ("bot", "Botnet"), ("infiltration", "Infiltration"),
    ("heartbleed", "Exploit"), ("exploit", "Exploit"),
    ("generic", "Generic"), ("fuzzer", "Fuzzing"), ("analysis", "Analysis"),
    ("backdoor", "Backdoor"), ("shellcode", "Shellcode"), ("worm", "Worm"),
]
FAMILY_IMPACT = {
    "Worm": 1.00, "Backdoor": 0.95, "Exploit": 0.90, "Shellcode": 0.90, "Infiltration": 0.90,
    "Botnet": 0.85, "DDoS": 0.80, "Web Attack": 0.80, "DoS": 0.70, "Brute Force": 0.70,
    "Generic": 0.50, "Unknown": 0.50, "Fuzzing": 0.40, "Analysis": 0.40, "Reconnaissance": 0.30,
}
ATTACK_FAMILIES = sorted(FAMILY_IMPACT)

TACTICS = {
    "TA0043": "Reconnaissance", "TA0001": "Initial Access", "TA0002": "Execution",
    "TA0006": "Credential Access", "TA0007": "Discovery", "TA0008": "Lateral Movement",
    "TA0011": "Command and Control", "TA0040": "Impact",
}
ATTACK_MAPPING = {  # family -> (technique id, technique name, tactic id, confidence)
    "DoS": ("T1499", "Endpoint Denial of Service", "TA0040", "high"),
    "DDoS": ("T1498", "Network Denial of Service", "TA0040", "high"),
    "Reconnaissance": ("T1046", "Network Service Discovery", "TA0007", "high"),
    "Brute Force": ("T1110.001", "Brute Force: Password Guessing", "TA0006", "high"),
    "Web Attack": ("T1190", "Exploit Public-Facing Application", "TA0001", "medium"),
    "Botnet": ("T1071.001", "Application Layer Protocol: Web Protocols", "TA0011", "medium"),
    "Worm": ("T1210", "Exploitation of Remote Services", "TA0008", "medium"),
    "Exploit": ("T1190", "Exploit Public-Facing Application", "TA0001", "medium"),
    "Infiltration": ("T1105", "Ingress Tool Transfer", "TA0011", "low"),
    "Fuzzing": ("T1595.002", "Active Scanning: Vulnerability Scanning", "TA0043", "low"),
    "Analysis": ("T1595", "Active Scanning", "TA0043", "low"),
    "Backdoor": ("T1071", "Application Layer Protocol", "TA0011", "low"),
    "Shellcode": ("T1203", "Exploitation for Client Execution", "TA0002", "low"),
}

PLAYBOOKS = [
    {"id": "PB-DOS-01", "title": "Denial-of-service response",
     "families": ["DoS", "DDoS"], "technique_ids": ["T1498", "T1499"], "tactic_ids": ["TA0040"],
     "steps": ["Confirm the service impact: latency, error rate and availability of the targeted host or service.",
               "Identify the top source addresses or networks and the targeted service and port.",
               "Apply rate limiting, WAF or load-balancer rules, or upstream scrubbing; contact the ISP for volumetric floods.",
               "Preserve flow logs and a packet sample for the incident record.",
               "After recovery, review capacity and auto-scaling limits for the service."]},
    {"id": "PB-SCAN-01", "title": "Scanning and service discovery",
     "families": ["Reconnaissance", "Analysis", "Fuzzing"], "technique_ids": ["T1046", "T1595", "T1595.002"],
     "tactic_ids": ["TA0007", "TA0043"],
     "steps": ["Determine whether the scanning source is internal or external.",
               "Internal source: treat the host as possibly compromised and open a host investigation.",
               "Check whether the scan was followed by successful connections or exploitation attempts on the same targets.",
               "External source: block or rate-limit it at the perimeter if policy allows.",
               "Review which services the scan could reach and close those that should not be exposed."]},
    {"id": "PB-CRED-01", "title": "Brute force and password guessing",
     "families": ["Brute Force"], "technique_ids": ["T1110", "T1110.001"], "tactic_ids": ["TA0006"],
     "steps": ["Identify the targeted service (SSH, FTP, web login) and the targeted accounts.",
               "Search authentication logs for a successful login after the failed attempts.",
               "If a login succeeded: disable or reset the account, revoke its sessions, and investigate the host.",
               "Block the source and enforce MFA and account lockout on the targeted service.",
               "Review the password policy and whether the service needs to be exposed."]},
    {"id": "PB-WEB-01", "title": "Web application attack",
     "families": ["Web Attack", "Exploit"], "technique_ids": ["T1190"], "tactic_ids": ["TA0001"],
     "steps": ["Pull web-server and WAF logs for the flow and look for attack payloads (SQL injection, XSS, exploit strings).",
               "Check application and database logs for signs of success (errors, unusual queries, new admin users).",
               "Apply a WAF rule or virtual patch, then patch the vulnerable component.",
               "Look for web shells or unexpected files on the server and rotate secrets the application could expose.",
               "Notify the application owner and record the vulnerability."]},
    {"id": "PB-C2-01", "title": "Command-and-control and backdoor activity",
     "families": ["Botnet", "Backdoor", "Infiltration"], "technique_ids": ["T1071", "T1071.001", "T1105"],
     "tactic_ids": ["TA0011"],
     "steps": ["Isolate the internal host with EDR or network containment.",
               "Identify the remote destination (domain or IP) and block it at the proxy, DNS and firewall.",
               "Collect a memory image and a disk image before cleaning.",
               "Hunt for the same destination and behaviour on other hosts.",
               "Re-image or clean the host, reset its credentials, and watch for re-infection."]},
    {"id": "PB-MAL-01", "title": "Exploitation and malware spread",
     "families": ["Exploit", "Shellcode", "Worm"], "technique_ids": ["T1190", "T1203", "T1210"],
     "tactic_ids": ["TA0001", "TA0002", "TA0008"],
     "steps": ["Isolate affected hosts and segment the network to stop the spread.",
               "Identify the exploited service or vulnerability (CVE) and patch or disable it.",
               "Hunt for lateral movement: new connections from affected hosts to other internal services.",
               "Collect malware samples and indicators and share them with threat intelligence.",
               "Restore affected hosts from known-good images and verify the patch."]},
    {"id": "PB-GEN-01", "title": "Generic suspicious network flow",
     "families": ["Generic", "Unknown"], "technique_ids": [], "tactic_ids": [],
     "steps": ["Validate the alert against raw logs or a packet capture of the flow.",
               "Enrich the source and destination: asset owner, criticality, reputation and recent alerts.",
               "Decide whether to escalate, monitor, or close as a false positive.",
               "Record the verdict so the model, mapping and playbooks can be improved."]},
]
PLAYBOOK_BY_ID = {playbook["id"]: playbook for playbook in PLAYBOOKS}
FALLBACK_PLAYBOOK = "PB-GEN-01"
PLAYBOOK_MATCH_WEIGHTS = {"family": 0.5, "technique": 0.3, "tactic": 0.2}
STRUCTURED_SHARE = 0.7  # similarity = 0.7 x structured match + 0.3 x text similarity
MIN_PLAYBOOK_SIMILARITY = 0.2

SEVERITY_WEIGHTS = {"confidence": 0.5, "impact": 0.3, "evidence": 0.2}
SEVERITY_LEVELS = [(80.0, "CRITICAL"), (60.0, "HIGH"), (40.0, "MEDIUM"), (0.0, "LOW")]
SEVERITY_BADGE = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢", "-": "⚪"}
EVIDENCE_SATURATION = 3.0  # three strong anomalies give evidence strength 1
MIN_BATCH_REFERENCE_ROWS = 30
MAX_LOG_ROWS = 200_000
TRAINING_SKLEARN_VERSION = "1.6.1"
DEFAULT_THRESHOLD = 0.5
MODELS_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL_FILE = MODELS_DIR / "soc_model.joblib"
DEFAULT_FAMILY_MODEL_FILE = MODELS_DIR / "attack_type_model.joblib"


def col_key(name: Any) -> str:
    """Column name reduced to lower-case letters and digits, so ' Flow Duration' matches 'flow_duration'."""
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


LABEL_COLUMN_KEYS = {"label", "attackcat", "attackcategory", "isattack", "attacklabel", "class", "originallabel",
                     "truelabel", "binarylabel"}


def to_attack_family(label: Any) -> str:
    """Map a CIC/UNSW label or a family name to the shared attack family ('Unknown' when nothing matches)."""
    key = col_key(label)
    if key in {"0", "false"}:
        return "BENIGN"
    for pattern, family in FAMILY_RULES:
        if pattern in key:
            return family
    return "Unknown"


def attack_url(technique_id: str) -> str:
    return "https://attack.mitre.org/techniques/" + technique_id.replace(".", "/") + "/"


def map_to_attack(family: str) -> dict:
    """ATT&CK record for a family; families without a defensible mapping return technique_id=None."""
    technique_id, technique_name, tactic_id, confidence = ATTACK_MAPPING.get(family, (None, None, None, None))
    return {"family": family, "technique_id": technique_id, "technique_name": technique_name,
            "tactic_id": tactic_id, "tactic": TACTICS.get(tactic_id), "confidence": confidence,
            "url": attack_url(technique_id) if technique_id else None}


def _unit(name: str, value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite value between 0 and 1.")
    return value


def severity_score(confidence: float, impact: float, evidence: float, weights: dict | None = None) -> float:
    """100 x (w_confidence x confidence + w_impact x impact + w_evidence x evidence), each part in [0, 1].
    Default weights 0.5 / 0.3 / 0.2 (the notebook's); a model bundle can bring its own."""
    weights = weights or SEVERITY_WEIGHTS
    parts = {"confidence": confidence, "impact": impact, "evidence": evidence}
    return round(100 * sum(weights[name] * _unit(name, value) for name, value in parts.items()), 1)


def severity_level(score: float, levels: list | None = None) -> str:
    """Level of the highest threshold the score reaches (levels sorted from highest threshold to lowest)."""
    levels = levels or SEVERITY_LEVELS
    return next((level for threshold, level in levels if score >= threshold), levels[-1][1])


def severity_badge(level: str) -> str:
    return SEVERITY_BADGE.get(str(level).upper(), "⚪")


# =====================================================================================================
# Reading logs and turning raw CIC / UNSW columns into the 7 harmonized features
# =====================================================================================================
RAW_FEATURE_MAPS = {  # dataset -> harmonized feature -> (accepted raw column names, multiplier)
    "CIC-IDS2017": {
        "flow_duration_seconds": (["Flow Duration"], 1e-6),  # microseconds -> seconds
        "src_packets": (["Total Fwd Packets"], 1.0),
        "dst_packets": (["Total Backward Packets"], 1.0),
        "src_bytes": (["Total Length of Fwd Packets"], 1.0),
        "dst_bytes": (["Total Length of Bwd Packets"], 1.0),
        "src_mean_packet_bytes": (["Fwd Packet Length Mean"], 1.0),
        "dst_mean_packet_bytes": (["Bwd Packet Length Mean"], 1.0),
    },
    "UNSW-NB15": {
        "flow_duration_seconds": (["dur"], 1.0),
        "src_packets": (["spkts"], 1.0),
        "dst_packets": (["dpkts"], 1.0),
        "src_bytes": (["sbytes"], 1.0),
        "dst_bytes": (["dbytes"], 1.0),
        "src_mean_packet_bytes": (["smean", "smeansz"], 1.0),
        "dst_mean_packet_bytes": (["dmean", "dmeansz"], 1.0),
    },
}


_NUMBER = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$")
_JSON_LIST_KEYS = ("flows", "records", "logs", "data", "events", "items")


def _json_records(text: str) -> list[dict]:
    """Records from JSON: a list, one object, an object holding a list (e.g. {"flows": [...]}), or JSON lines."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        lines = [line for line in text.splitlines() if line.strip()]
        try:
            parsed = [json.loads(line) for line in lines]
        except json.JSONDecodeError as exc:
            raise ValueError(f"Not valid JSON or JSON lines: {exc}") from exc
    if isinstance(parsed, dict):
        nested = next((parsed[k] for k in _JSON_LIST_KEYS if isinstance(parsed.get(k), list)), None)
        parsed = nested if nested is not None else [parsed]
    if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
        raise ValueError("JSON logs must be objects (one per flow), a list of objects, or JSON lines.")
    return parsed


def _text_line_record(line: str) -> dict | None:
    """One plain-text log line -> record. Accepts key=value / key: value pairs, or the 7 feature values in order."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if "=" in line or re.search(r"[A-Za-z_]\s*:", line):
        # Pairs separated by commas, semicolons, pipes or tabs; otherwise by spaces (then keys cannot contain spaces).
        parts = re.split(r"[,;|\t]", line) if re.search(r"[,;|\t]", line) else line.split()
        record = {}
        for part in parts:
            match = re.match(r"\s*([^=:]+?)\s*[=:]\s*(.*?)\s*$", part)
            if match:
                record[match.group(1).strip().strip("\"'")] = match.group(2).strip().strip("\"'")
        return record or None
    values = [v for v in re.split(r"[,;\s|]+", line) if v]
    if len(values) == len(HARMONIZED_FEATURES) and all(_NUMBER.match(v) for v in values):
        return dict(zip(HARMONIZED_FEATURES, values))
    raise ValueError(f"Cannot read this text line: {line[:120]!r}. Use key=value pairs or the "
                     f"{len(HARMONIZED_FEATURES)} feature values in this order: {', '.join(HARMONIZED_FEATURES)}.")


def _looks_like_csv(text: str) -> bool:
    first = next(line for line in text.splitlines() if line.strip())
    if "=" in first or re.search(r"[A-Za-z_]\s*:", first) or "," not in first:
        return False
    return not all(_NUMBER.match(v.strip()) for v in first.split(",") if v.strip())  # a header row, not numbers


def _looks_like_text_lines(text: str) -> bool:
    """True when the first line is key=value / key: value pairs or just the feature values (not a CSV header)."""
    first = next(line for line in text.splitlines() if line.strip())
    values = [v for v in re.split(r"[,;\s|]+", first.strip()) if v]
    return "=" in first or bool(re.search(r"[A-Za-z_]\s*:", first)) or all(_NUMBER.match(v) for v in values)


def read_logs(name: str, data: bytes) -> pd.DataFrame:
    """Parse logs: CSV, JSON (several shapes) or plain text (key=value pairs or feature values, one flow per line)."""
    text = data.decode("utf-8-sig", errors="replace").strip()
    if not text:
        raise ValueError("The log is empty.")
    suffix = Path(name).suffix.lower()
    try:
        if suffix in {".json", ".jsonl", ".ndjson"} or text[0] in "[{":
            frame = pd.json_normalize(_json_records(text))
            # Nested fields become "flow.src_bytes"; keep the last part when that stays unique.
            short = [str(c).split(".")[-1] for c in frame.columns]
            if len(set(short)) == len(short):
                frame.columns = short
        elif _looks_like_csv(text) or (suffix == ".csv" and not _looks_like_text_lines(text)):
            frame = pd.read_csv(io.StringIO(text), low_memory=False)
        else:
            records = [r for r in (_text_line_record(line) for line in text.splitlines()) if r]
            if not records:
                raise ValueError("No flow found in the text.")
            frame = pd.DataFrame(records)
    except pd.errors.ParserError as exc:
        raise ValueError(f"Could not read the log: {exc}") from exc
    if frame.empty:
        raise ValueError("The log has no rows.")
    if len(frame) > MAX_LOG_ROWS:
        raise ValueError(f"The log has {len(frame):,} rows; the limit is {MAX_LOG_ROWS:,}. Split it into smaller files.")
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame.reset_index(drop=True)


def harmonize_logs(logs: pd.DataFrame) -> pd.DataFrame:
    """Add the 7 harmonized features (and dataset_source) when the log has raw CIC-IDS2017 or UNSW-NB15 columns."""
    out = logs.copy()
    keys = {col_key(c): c for c in out.columns}
    if all(col_key(f) in keys for f in HARMONIZED_FEATURES):
        for feature in HARMONIZED_FEATURES:
            out[feature] = pd.to_numeric(out[keys[col_key(feature)]], errors="coerce")
        return out
    for source, mapping in RAW_FEATURE_MAPS.items():
        converted = {}
        for feature, (aliases, multiplier) in mapping.items():
            column = next((keys[col_key(a)] for a in aliases if col_key(a) in keys), None)
            if column is None:
                break
            converted[feature] = pd.to_numeric(out[column], errors="coerce") * multiplier
        else:
            for feature, values in converted.items():
                out[feature] = values
            if "dataset_source" not in out.columns:
                out["dataset_source"] = source
            return out
    return out


# =====================================================================================================
# The model adapter: one interface for every supported joblib layout
# =====================================================================================================
def _is_benign_class(value: Any) -> bool:
    return col_key(value) in {"0", "benign", "normal", "false", "0.0"}


def required_columns(estimator: Any) -> list[str] | None:
    """Input columns the estimator really uses (labels the pipeline drops are not required)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    first = estimator.steps[0][1] if isinstance(estimator, Pipeline) else estimator
    if isinstance(first, ColumnTransformer) and hasattr(first, "transformers_"):
        remainder_used = not (isinstance(first.remainder, str) and first.remainder == "drop")
        columns = []
        for _, transformer, cols in first.transformers_:
            if isinstance(transformer, str) and transformer == "drop":
                continue
            if isinstance(cols, str):
                cols = [cols]
            if not all(isinstance(c, str) for c in np.atleast_1d(cols)):
                remainder_used = True  # positional columns: fall back to every fitted column
                break
            columns += [c for c in cols if c not in columns]
        if not remainder_used:
            return columns
    names = getattr(estimator, "feature_names_in_", None)
    return [str(n) for n in names] if names is not None else None


@dataclass
class Scorer:
    """One fitted estimator with its threshold and optional calibrator."""
    name: str
    estimator: Any
    threshold: float = 0.5
    calibrator: Any = None
    class_names: dict | None = None
    columns: list[str] | None = None
    abstain_below: float = 0.0  # attack-type models: below this probability the type is reported as Unknown

    def __post_init__(self):
        if self.columns is None:
            self.columns = required_columns(self.estimator)
        self.classes = np.asarray(self.estimator.classes_) if hasattr(self.estimator, "classes_") else None
        self.has_proba = hasattr(self.estimator, "predict_proba") and self.classes is not None

    def label(self, value: Any) -> str:
        if self.class_names:
            value = self.class_names.get(value, self.class_names.get(str(value), value))
        return str(value)

    @property
    def is_binary(self) -> bool:
        return self.classes is not None and len(self.classes) == 2

    def model_input(self, logs: pd.DataFrame) -> pd.DataFrame:
        """Columns the estimator needs, matched by name (case, spaces and an 'unsw__' prefix are ignored)."""
        keys = {col_key(c): c for c in logs.columns if col_key(c) not in LABEL_COLUMN_KEYS}
        if self.columns is None:  # estimator fitted without column names: use the harmonized features
            if not all(f in logs.columns for f in HARMONIZED_FEATURES):
                raise ValueError("The model does not record its input columns and the log has no harmonized features.")
            n_expected = getattr(self.estimator, "n_features_in_", len(HARMONIZED_FEATURES))
            if n_expected != len(HARMONIZED_FEATURES):
                raise ValueError(f"The model expects {n_expected} unnamed inputs; save it with column names instead.")
            return logs[HARMONIZED_FEATURES].to_numpy(dtype=float)
        data, missing = {}, []
        for name in self.columns:
            candidates = [name, re.sub(r"^unsw__", "", name)]
            column = next((keys[col_key(c)] for c in candidates if col_key(c) in keys), None)
            if column is None:
                if col_key(name) in LABEL_COLUMN_KEYS:
                    data[name] = np.nan  # a label column the pipeline drops anyway: never filled from the log
                    continue
                missing.append(name)
                continue
            values = logs[column]
            numeric = pd.to_numeric(values, errors="coerce")
            data[name] = numeric if numeric.notna().sum() >= 0.99 * values.notna().sum() else values.astype(object)
        if missing:
            raise ValueError(f"The log is missing {len(missing)} column(s) the model needs: {missing[:12]}"
                             f"{' ...' if len(missing) > 12 else ''}")
        return pd.DataFrame(data, index=logs.index)

    def score(self, logs: pd.DataFrame) -> pd.DataFrame:
        """Per row: attack probability, predicted label, and the probability of that label."""
        x = self.model_input(logs)
        if not self.has_proba:
            predicted = np.asarray(self.estimator.predict(x))
            attack = np.array([not _is_benign_class(p) for p in predicted], dtype=float)
            labels = [self.label(p) if a else "BENIGN" for p, a in zip(predicted, attack)]
            return pd.DataFrame({"attack_probability": attack, "predicted_label": labels,
                                 "label_probability": np.nan, "probability_known": False}, index=logs.index)
        proba = np.asarray(self.estimator.predict_proba(x), dtype=float)
        benign = np.array([_is_benign_class(c) for c in self.classes])
        if not benign.any():
            raise ValueError(f"The model has no BENIGN class ({list(self.classes)[:6]}...). "
                             "Load it as the attack-type model instead.")
        raw_attack_probability = proba[:, ~benign].sum(axis=1)
        attack_probability = raw_attack_probability
        if self.calibrator is not None:
            attack_probability = np.asarray(self.calibrator.predict(attack_probability), dtype=float)
        attack_probability = np.clip(attack_probability, 0.0, 1.0)
        attack_proba = np.where(benign, -1.0, proba)  # best attack class, used when the row is an attack
        best_attack = attack_proba.argmax(axis=1)
        is_attack = attack_probability >= self.threshold
        labels = [self.label(self.classes[j]) if a else "BENIGN" for j, a in zip(best_attack, is_attack)]
        # Category confidence = P(category | attack): the best attack class's share of all attack probability.
        category_confidence = proba[np.arange(len(proba)), best_attack] / np.maximum(raw_attack_probability, 1e-12)
        label_probability = np.where(is_attack, category_confidence, proba[:, benign].sum(axis=1))
        return pd.DataFrame({"attack_probability": attack_probability, "predicted_label": labels,
                             "label_probability": label_probability, "probability_known": True}, index=logs.index)


@dataclass
class Policy:
    """Severity and evidence settings. Defaults are the notebook's; a model bundle can replace any of them."""
    family_impact: dict = field(default_factory=lambda: dict(FAMILY_IMPACT))
    severity_weights: dict = field(default_factory=lambda: dict(SEVERITY_WEIGHTS))
    severity_levels: list = field(default_factory=lambda: list(SEVERITY_LEVELS))
    benign_reference: dict | None = None   # dataset_source -> DataFrame (index = quantiles, columns = features)
    default_source: str | None = None      # dataset assumed for logs without a dataset_source
    alert_threshold: float | None = None   # starting value of the threshold slider
    test_metrics: Any = None               # shown in the sidebar
    sources: dict = field(default_factory=dict)  # setting -> "model bundle" or "app default"

    def impact(self, family: str) -> float:
        return float(self.family_impact.get(family, self.family_impact.get("Unknown", FAMILY_IMPACT["Unknown"])))


@dataclass
class DetectionModel:
    """The loaded joblib: one scorer for every row, or one scorer per dataset_source."""
    scorers: dict[str, Scorer]
    routed: bool
    description: str
    load_warnings: list[str] = field(default_factory=list)
    policy: Policy = field(default_factory=Policy)

    def score(self, logs: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
        """Scores per row. threshold=None uses each scorer's own (saved) threshold."""
        frames = []
        if not self.routed:
            scorer = next(iter(self.scorers.values()))
            frames.append(self._with_threshold(scorer, logs, threshold).assign(scored_by=scorer.name))
        else:
            if "dataset_source" not in logs.columns:
                raise ValueError(f"This model has one detector per dataset {list(self.scorers)}; "
                                 "the log needs a dataset_source column (or raw CIC/UNSW columns).")
            for source, part in logs.groupby(logs["dataset_source"].astype(str), sort=False):
                if source not in self.scorers:
                    frames.append(pd.DataFrame({"attack_probability": np.nan, "predicted_label": "UNSCORED",
                                                "label_probability": np.nan, "probability_known": False,
                                                "scored_by": f"no detector for {source}"}, index=part.index))
                    continue
                frames.append(self._with_threshold(self.scorers[source], part, threshold).assign(scored_by=self.scorers[source].name))
        return pd.concat(frames).loc[logs.index]

    @staticmethod
    def _with_threshold(scorer: Scorer, logs: pd.DataFrame, threshold: float | None) -> pd.DataFrame:
        if threshold is None:
            return scorer.score(logs)
        original, scorer.threshold = scorer.threshold, threshold
        try:
            return scorer.score(logs)
        finally:
            scorer.threshold = original

    def scorer_for(self, row: pd.Series) -> Scorer | None:
        if not self.routed:
            return next(iter(self.scorers.values()))
        return self.scorers.get(str(row.get("dataset_source")))

    def scorer_groups(self, rows: pd.DataFrame) -> list[tuple[Scorer, pd.DataFrame]]:
        """(scorer, rows) pairs: every row with the scorer of its dataset (rows without one are left out)."""
        if not self.routed:
            return [(next(iter(self.scorers.values())), rows)]
        if "dataset_source" not in rows.columns:
            return []
        return [(self.scorers[source], part) for source, part in rows.groupby(rows["dataset_source"].astype(str))
                if source in self.scorers]


def wrap_model(obj: Any, name: str = "model") -> DetectionModel:
    """Turn whatever the joblib file contains into a DetectionModel (see the module docstring)."""
    if isinstance(obj, dict) and "model" in obj:
        scorer = Scorer(name, obj["model"], float(obj.get("threshold", 0.5)), obj.get("calibrator"),
                        obj.get("class_names"), obj.get("features"), float(obj.get("abstain_below", 0.0)))
        return DetectionModel({name: scorer}, False, f"bundle with {type(obj['model']).__name__}")
    if isinstance(obj, dict) and obj and all(isinstance(v, dict) and "model" in v for v in obj.values()):
        scorers = {str(source): Scorer(str(v.get("model_name", source)), v["model"], float(v.get("threshold", 0.5)),
                                       v.get("calibrator"), v.get("class_names"), v.get("features"),
                                       float(v.get("abstain_below", 0.0)))
                   for source, v in obj.items()}
        return DetectionModel(scorers, True, f"one detector per dataset: {', '.join(scorers)}")
    if hasattr(obj, "predict"):
        return DetectionModel({name: Scorer(name, obj)}, False, type(obj).__name__)
    raise ValueError(f"Unsupported model file content: {type(obj).__name__}. Expected a fitted estimator, a dict with "
                     "'model', or a dict of per-dataset detectors.")


DETECTION_KEYS = ("attack_model", "binary_model", "detection_model", "detector", "binary", "stage1")
FAMILY_KEYS = ("family_model", "attack_type_model", "category_model", "multiclass_model", "classifier", "family",
               "stage2")
WEIGHT_ALIASES = {"confidence": ("confidence", "prob_attack", "p_attack", "probability", "attack_probability"),
                  "impact": ("impact", "family_impact"),
                  "evidence": ("evidence", "evidence_strength")}


def _as_quantile(label: Any) -> float | None:
    """0.99, '0.99', 99, '99%', 'p99' or 'q99' -> 0.99; anything else -> None."""
    text = str(label).strip().lower().lstrip("pq").rstrip("%")
    try:
        value = float(text)
    except ValueError:
        return None
    value = value / 100 if 1 < value <= 100 else value
    return value if 0 <= value <= 1 else None


def _quantile_frame(value: Any) -> pd.DataFrame:
    """One dataset's reference as a DataFrame with quantiles as the index and features as columns."""
    frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    for candidate in (frame, frame.T):
        quantiles = [_as_quantile(q) for q in candidate.index]
        if quantiles and all(q is not None for q in quantiles):
            out = candidate.copy()
            out.index = quantiles
            return out.apply(pd.to_numeric, errors="coerce").sort_index()
    raise ValueError("no quantile labels (like 0.01 ... 0.99) on either axis")


def normalize_benign_reference(raw: Any) -> dict[str, pd.DataFrame]:
    """Accept {dataset: DataFrame or dict}, one DataFrame/dict for all data, or a (dataset, quantile) MultiIndex frame."""
    if isinstance(raw, pd.DataFrame) and isinstance(raw.index, pd.MultiIndex):
        return {str(source): _quantile_frame(part.droplevel(0)) for source, part in raw.groupby(level=0)}
    if isinstance(raw, dict) and raw and all(isinstance(v, (dict, pd.DataFrame)) for v in raw.values()):
        try:  # {dataset: per-dataset reference}
            return {str(source): _quantile_frame(value) for source, value in raw.items()}
        except ValueError:
            pass  # maybe {feature: {quantile: value}} for all data
    return {"ALL": _quantile_frame(raw)}


def policy_from_bundle(bundle: dict) -> tuple[Policy, list[str]]:
    """Read the bundle's severity / evidence settings; anything unreadable keeps the app default, with a warning."""
    policy, notes = Policy(), []

    def use(setting, build):
        if setting not in bundle or bundle[setting] is None:
            policy.sources[setting] = "app default"
            return
        try:
            build(bundle[setting])
            policy.sources[setting] = "model bundle"
        except (TypeError, ValueError, KeyError, AttributeError) as exc:
            policy.sources[setting] = "app default (bundle value not readable)"
            notes.append(f"Bundle setting '{setting}' could not be read ({exc}); using the app default.")

    def set_impact(value):
        impact = {str(k): float(v) for k, v in dict(value).items()}
        if not all(0.0 <= v <= 1.0 for v in impact.values()):
            raise ValueError("impact values must be between 0 and 1")
        policy.family_impact = {**FAMILY_IMPACT, **impact}

    def set_weights(value):
        value = {str(k).lower(): float(v) for k, v in dict(value).items()}
        weights = {}
        for part, aliases in WEIGHT_ALIASES.items():
            key = next((a for a in aliases if a in value), None)
            if key is None:
                raise KeyError(f"no weight for {part} (expected one of {aliases})")
            weights[part] = value[key]
        policy.severity_weights = weights

    def set_levels(value):
        pairs = value.items() if isinstance(value, dict) else value
        levels = []
        for pair in pairs:
            a, b = list(pair)
            threshold, level = (a, b) if isinstance(b, str) else (b, a)  # (threshold, name) or {name: threshold}
            levels.append((float(threshold), str(level).upper()))
        if not levels:
            raise ValueError("empty list")
        policy.severity_levels = sorted(levels, reverse=True)

    def set_reference(value):
        policy.benign_reference = normalize_benign_reference(value)

    def set_threshold(value):
        value = float(value)
        if not 0.0 < value < 1.0:
            raise ValueError("must be between 0 and 1")
        policy.alert_threshold = value

    use("family_impact", set_impact)
    use("severity_weights", set_weights)
    use("severity_levels", set_levels)
    use("benign_reference", set_reference)
    use("alert_threshold", set_threshold)
    use("default_source", lambda value: setattr(policy, "default_source", str(value)))
    use("model_test_metrics", lambda value: setattr(policy, "test_metrics", value))
    return policy, notes


def wrap_model_stages(obj: Any, name: str = "model") -> tuple[DetectionModel, DetectionModel | None]:
    """(stage-1 detection model, stage-2 attack-category model or None) from the joblib content."""
    if isinstance(obj, dict):
        detection_key = next((k for k in DETECTION_KEYS if k in obj), None)
        family_key = next((k for k in FAMILY_KEYS if k in obj), None)
        features = list(obj["features"]) if isinstance(obj.get("features"), (list, tuple)) else None

        def stage(key):  # the bundle's feature list tells each stage which columns to use
            value = obj[key]
            return {"model": value, "features": features} if features and not isinstance(value, dict) else value

        if detection_key:
            detection = wrap_model(stage(detection_key), f"{name}:{detection_key}")
            family = wrap_model(stage(family_key), f"{name}:{family_key}") if family_key else None
            detection.policy, notes = policy_from_bundle(obj)
            detection.load_warnings += notes
            return detection, family
        if family_key and "model" in obj:
            return wrap_model({k: v for k, v in obj.items() if k != family_key}, name), \
                wrap_model(stage(family_key), f"{name}:{family_key}")
    return wrap_model(obj, name), None


def load_joblib(data: bytes) -> tuple[Any, list[str]]:
    """Load a joblib file from bytes and collect warnings (for example a scikit-learn version mismatch)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        obj = joblib.load(io.BytesIO(data))
    return obj, sorted({str(w.message).splitlines()[0][:300] for w in caught})


# =====================================================================================================
# Evidence: benign reference ranges and the features that drove the decision
# =====================================================================================================
QUANTILES = (0.01, 0.05, 0.5, 0.95, 0.99)


def benign_reference(frame: pd.DataFrame, columns: list[str]) -> dict[str, pd.DataFrame]:
    """Per dataset_source (or 'ALL'): 1st, 5th, 50th, 95th and 99th percentiles of numeric columns over benign flows."""
    numeric = [c for c in columns if c in frame.columns and pd.api.types.is_numeric_dtype(frame[c])]
    if not numeric:
        return {}
    reference = {"ALL": frame[numeric].quantile(list(QUANTILES))}
    if "dataset_source" in frame.columns:
        for source, part in frame.groupby("dataset_source"):
            if len(part) >= MIN_BATCH_REFERENCE_ROWS:
                reference[str(source)] = part[numeric].quantile(list(QUANTILES))
    return reference


def pick_reference(reference: dict[str, pd.DataFrame], source: Any) -> tuple[pd.DataFrame | None, str]:
    """The reference for a dataset: its own, else 'ALL', else the only one there is."""
    if not reference:
        return None, ""
    source = str(source) if source is not None and not pd.isna(source) else ""
    if source in reference:
        return reference[source], source
    if "ALL" in reference:
        return reference["ALL"], "all"
    first = next(iter(reference))
    return reference[first], first


def _quantile_row(ref: pd.DataFrame, quantile: float) -> pd.Series | None:
    match = [q for q in ref.index if np.isclose(float(q), quantile)]
    return ref.loc[match[0]] if match else None


def unusual_values(row: dict, reference: dict[str, pd.DataFrame]) -> tuple[list[dict], float]:
    """Features outside the benign 5th-95th (moderate) or 1st-99th (strong) percentile, and evidence strength.
    Checks whose percentile is not in the reference are skipped."""
    ref, label = pick_reference(reference, row.get("dataset_source"))
    if ref is None:
        return [], 0.0
    p = {q: _quantile_row(ref, q) for q in (0.01, 0.05, 0.95, 0.99)}
    items, weight = [], 0.0
    for feature in ref.columns:
        value = pd.to_numeric(row.get(feature), errors="coerce")
        if pd.isna(value):
            continue

        def bound(q):
            return None if p[q] is None or pd.isna(p[q].get(feature)) else float(p[q][feature])

        p01, p05, p95, p99 = bound(0.01), bound(0.05), bound(0.95), bound(0.99)
        checks = [
            (p99 is not None and value > p99, "above", "strong", "99th", p99),
            (p95 is not None and value > p95 and (p99 is None or value <= p99), "above", "moderate", "95th", p95),
            (p01 is not None and value < p01, "below", "strong", "1st", p01),
            (p05 is not None and value < p05 and (p01 is None or value >= p01), "below", "moderate", "5th", p05),
        ]
        for test, direction, strength, pct, threshold in checks:
            if test:
                weight += 1.0 if strength == "strong" else 0.5
                items.append({"feature": feature, "value": float(value), "direction": direction, "strength": strength,
                              "text": f"{feature} = {value:,.6g} is {direction} the {pct} percentile of benign "
                                      f"{label} traffic ({threshold:,.6g})"})
    items.sort(key=lambda item: item["strength"] != "strong")
    return items, min(1.0, weight / EVIDENCE_SATURATION)


def typical_benign_values(ref: pd.DataFrame) -> pd.Series | None:
    """Median per feature; if the reference has no median, the midpoint of the 5th-95th range on a log scale."""
    median = _quantile_row(ref, 0.5)
    if median is not None:
        return median
    low, high = _quantile_row(ref, 0.05), _quantile_row(ref, 0.95)
    if low is None or high is None:
        return None
    return np.expm1((np.log1p(low.clip(lower=0)) + np.log1p(high.clip(lower=0))) / 2)


def decision_drivers(scorer: Scorer, row: pd.DataFrame, reference: dict[str, pd.DataFrame], top: int = 5) -> pd.DataFrame:
    """How much P(ATTACK) drops when each feature is set to its typical benign value (bigger drop = stronger driver)."""
    empty = pd.DataFrame(columns=["feature", "value", "typical_benign", "probability_change"])
    if not reference or not scorer.has_proba:
        return empty
    ref, _ = pick_reference(reference, row["dataset_source"].iloc[0] if "dataset_source" in row.columns else None)
    medians = typical_benign_values(ref) if ref is not None else None
    if medians is None:
        return empty
    base = float(scorer.score(row)["attack_probability"].iloc[0])
    rows = []
    for feature, typical in medians.items():
        if feature not in row.columns or pd.isna(typical):
            continue
        changed = row.copy()
        changed[feature] = typical
        rows.append({"feature": feature, "value": float(pd.to_numeric(row[feature].iloc[0], errors="coerce")),
                     "typical_benign": float(typical),
                     "probability_change": base - float(scorer.score(changed)["attack_probability"].iloc[0])})
    drivers = pd.DataFrame(rows, columns=["feature", "value", "typical_benign", "probability_change"])
    return drivers.reindex(drivers["probability_change"].abs().sort_values(ascending=False).index).head(top)


# =====================================================================================================
# Playbook similarity
# =====================================================================================================
def _playbook_text(playbook: dict) -> str:
    techniques = [ATTACK_MAPPING[f][1] for f in playbook["families"] if f in ATTACK_MAPPING]
    return " ".join([playbook["title"], *playbook["families"], *techniques,
                     *[TACTICS[t] for t in playbook["tactic_ids"]], *playbook["steps"]])


@lru_cache(maxsize=1)
def _playbook_vectorizer():
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform([_playbook_text(pb) for pb in PLAYBOOKS])
    return vectorizer, matrix


def playbook_similarity(family: str, evidence_features: tuple[str, ...] = ()) -> pd.DataFrame:
    """Every playbook ranked by similarity to the alert (0-1), with the structured and text parts and the reasons."""
    return pd.DataFrame(_ranked_playbooks(family, tuple(evidence_features)))


@lru_cache(maxsize=4096)  # alerts repeat the same (type, evidence) combinations, so large logs stay fast
def _ranked_playbooks(family: str, evidence_features: tuple[str, ...]) -> tuple[dict, ...]:
    attack = map_to_attack(family)
    alert_text = " ".join(filter(None, [family, attack["technique_name"], attack["tactic"],
                                        *[f.replace("_", " ") for f in evidence_features]]))
    vectorizer, matrix = _playbook_vectorizer()
    text_similarity = (matrix @ vectorizer.transform([alert_text]).T).toarray().ravel()
    rows = []
    for playbook, text_score in zip(PLAYBOOKS, text_similarity):
        structured, reasons = 0.0, []
        if family in playbook["families"]:
            structured += PLAYBOOK_MATCH_WEIGHTS["family"]
            reasons.append(f"covers {family}")
        if attack["technique_id"] and attack["technique_id"] in playbook["technique_ids"]:
            structured += PLAYBOOK_MATCH_WEIGHTS["technique"]
            reasons.append(f"covers ATT&CK {attack['technique_id']}")
        if attack["tactic_id"] and attack["tactic_id"] in playbook["tactic_ids"]:
            structured += PLAYBOOK_MATCH_WEIGHTS["tactic"]
            reasons.append(f"covers tactic {attack['tactic']}")
        similarity = STRUCTURED_SHARE * structured + (1 - STRUCTURED_SHARE) * float(text_score)
        rows.append({"playbook": playbook["id"], "title": playbook["title"], "similarity": round(similarity, 4),
                     "structured_match": round(structured, 4), "text_similarity": round(float(text_score), 4),
                     "reasons": "; ".join(reasons) or "text similarity only"})
    ranked = pd.DataFrame(rows).sort_values(["similarity", "playbook"], ascending=[False, True], kind="stable")
    if ranked["similarity"].iloc[0] < MIN_PLAYBOOK_SIMILARITY:  # nothing fits: the generic playbook leads
        fallback = ranked["playbook"].eq(FALLBACK_PLAYBOOK)
        ranked = pd.concat([ranked[fallback], ranked[~fallback]])
    return tuple(ranked.to_dict("records"))


# =====================================================================================================
# The full analysis: classification, attack type, evidence, severity, ATT&CK and playbook
# =====================================================================================================
@dataclass
class Analysis:
    logs: pd.DataFrame          # harmonized logs (model inputs come from here)
    results: pd.DataFrame       # one row per log row
    reference: dict             # benign reference used for evidence
    reference_note: str


def analyze_logs(logs: pd.DataFrame, model: DetectionModel, threshold: float | None = None,
                 family_model: DetectionModel | None = None, reference_logs: pd.DataFrame | None = None) -> Analysis:
    """Classify every log row and add attack type, evidence, severity, ATT&CK technique and the best playbook."""
    policy = model.policy
    logs = harmonize_logs(logs)
    if policy.default_source:  # logs that do not say their dataset use the bundle's default (evidence reference)
        if "dataset_source" not in logs.columns:
            logs["dataset_source"] = policy.default_source
        else:
            logs["dataset_source"] = logs["dataset_source"].where(logs["dataset_source"].notna(), policy.default_source)
    scores = model.score(logs, threshold)
    evidence_columns = sorted({c for s in model.scorers.values() for c in (s.columns or HARMONIZED_FEATURES)
                               if col_key(c) not in LABEL_COLUMN_KEYS} | set(HARMONIZED_FEATURES))
    evidence_columns = [c for c in evidence_columns if c in logs.columns]

    # Benign reference for the evidence: uploaded file > the model bundle's > benign flows in this log.
    if reference_logs is not None:
        reference = benign_reference(harmonize_logs(reference_logs), evidence_columns)
        reference_note = f"uploaded benign reference ({len(reference_logs):,} flows)"
    elif policy.benign_reference:
        reference = policy.benign_reference
        reference_note = f"the benign reference in the model bundle ({', '.join(reference)})"
    else:
        benign_rows = logs[scores["predicted_label"].eq("BENIGN")]
        if len(benign_rows) >= MIN_BATCH_REFERENCE_ROWS:
            reference = benign_reference(benign_rows, evidence_columns)
            reference_note = f"{len(benign_rows):,} flows in this log that the model classified as BENIGN"
        else:
            reference, reference_note = {}, ("no benign reference: upload known-benign flows in the sidebar, or analyze "
                                             f"a log with at least {MIN_BATCH_REFERENCE_ROWS} benign flows")

    is_attack = ~scores["predicted_label"].isin(["BENIGN", "UNSCORED"])
    family = pd.Series(np.where(is_attack, scores["predicted_label"].map(to_attack_family), "-"), index=logs.index)
    family_probability = pd.Series(np.where(is_attack, scores["label_probability"], np.nan), index=logs.index)
    impact = family.map(policy.impact)
    family_source = pd.Series(np.where(is_attack, "detection model", "-"), index=logs.index)
    if family_model is not None and is_attack.any():
        for family_scorer, part in family_model.scorer_groups(logs[is_attack]):
            x = family_scorer.model_input(part)
            if family_scorer.has_proba:
                proba = np.asarray(family_scorer.estimator.predict_proba(x), dtype=float)
                classes = np.asarray([to_attack_family(family_scorer.label(c)) for c in family_scorer.classes])
                impacts = np.array([policy.impact(c) for c in classes])
                best = proba.max(axis=1)
                # Below the model's own abstention cut-off the attack type is reported as Unknown.
                family.loc[part.index] = np.where(best >= family_scorer.abstain_below, classes[proba.argmax(axis=1)], "Unknown")
                family_probability.loc[part.index] = best
                impact.loc[part.index] = np.clip(proba @ impacts, 0.0, 1.0)
            else:
                family.loc[part.index] = [to_attack_family(family_scorer.label(p)) for p in family_scorer.estimator.predict(x)]
            family_source.loc[part.index] = "attack-type model"
    family = family.where(~(is_attack & family.eq("BENIGN")), "Unknown")  # an attack never gets a BENIGN family

    rows = []
    for index, log_row in zip(logs.index, logs.to_dict("records")):
        score, attack = scores.loc[index], bool(is_attack.loc[index])
        confidence = score["attack_probability"] if attack else 1.0 - score["attack_probability"]
        if not score["probability_known"]:
            confidence = np.nan  # a model without predict_proba gives labels only
        items, strength = unusual_values(log_row, reference) if attack else ([], 0.0)
        record = {"row": index, "dataset_source": log_row.get("dataset_source", "-"),
                  "verdict": "ATTACK" if attack else score["predicted_label"],
                  "attack_probability": score["attack_probability"], "confidence": confidence,
                  "attack_type": family.loc[index], "attack_type_probability": family_probability.loc[index],
                  "attack_type_from": family_source.loc[index], "scored_by": score["scored_by"],
                  "evidence_strength": strength, "evidence": items}
        if attack:
            usable_confidence = score["attack_probability"] if score["probability_known"] else 1.0
            record["severity_score"] = severity_score(usable_confidence, impact.loc[index], strength,
                                                      policy.severity_weights)
            record["severity_level"] = severity_level(record["severity_score"], policy.severity_levels)
            attack_record = map_to_attack(record["attack_type"])
            best_playbook = _ranked_playbooks(record["attack_type"], tuple(i["feature"] for i in items[:3]))[0]
            record.update({"attack_technique": attack_record["technique_id"] or "unmapped",
                           "attack_tactic": attack_record["tactic"] or "-",
                           "playbook": best_playbook["playbook"], "playbook_similarity": best_playbook["similarity"]})
        else:
            record.update({"severity_score": np.nan, "severity_level": "-", "attack_technique": "-",
                           "attack_tactic": "-", "playbook": "-", "playbook_similarity": np.nan})
        rows.append(record)
    results = pd.DataFrame(rows).set_index("row", drop=False).rename_axis(None)
    for key, column in {col_key(c): c for c in logs.columns}.items():  # show labels from the log, never used
        if key in {"label", "attackcat"} and column in logs.columns:
            results[f"log_{key}"] = logs[column].astype(str).to_numpy()
    return Analysis(logs, results, reference, reference_note)


def results_for_download(analysis: Analysis) -> pd.DataFrame:
    out = analysis.results.drop(columns="evidence").copy()
    out["evidence_text"] = [" | ".join(i["text"] for i in items) for items in analysis.results["evidence"]]
    return out


# =====================================================================================================
# Streamlit user interface
# =====================================================================================================
def _bytes_from(uploaded, path_text: str) -> tuple[bytes, str] | tuple[None, None]:
    if uploaded is not None:
        return uploaded.getvalue(), uploaded.name
    path_text = (path_text or "").strip().strip('"')
    if path_text:
        path = Path(path_text).expanduser()
        if not path.is_file():
            raise ValueError(f"File not found: {path}")
        return path.read_bytes(), path.name
    return None, None


def main() -> None:
    import hashlib

    import streamlit as st

    st.set_page_config(page_title="SOC Assistant", page_icon="🛡️", layout="wide")

    @st.cache_resource(show_spinner="Loading model…")
    def load_model_cached(digest: str, _data: bytes, name: str):
        obj, load_warnings = load_joblib(_data)
        model, embedded_family = wrap_model_stages(obj, Path(name).stem)
        model.load_warnings = load_warnings + model.load_warnings  # keep the bundle-setting notes
        return model, embedded_family

    @st.cache_data(show_spinner="Reading logs…")
    def read_logs_cached(digest: str, _data: bytes, name: str) -> pd.DataFrame:
        return read_logs(name, _data)

    @st.cache_data(show_spinner="Classifying logs…", max_entries=8)
    def analyze_cached(key: tuple, _logs, _model, threshold, _family_model, _reference):
        return analyze_logs(_logs, _model, threshold, _family_model, _reference)

    def digest(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    st.title("🛡️ SOC Assistant")
    st.caption("Step 1: BENIGN or ATTACK. Step 2: for attacks, the attack category with its confidence, the severity "
               "(LOW / MEDIUM / HIGH / CRITICAL), the evidence, the MITRE ATT&CK technique and the most similar "
               "playbook. An alert is a lead for investigation, not proof of an attack.")

    def default_path(env_name: str, default_file: Path) -> str:
        return os.environ.get(env_name) or (str(default_file) if default_file.is_file() else "")

    # ---------------- Sidebar: model, optional helpers, settings ----------------
    with st.sidebar:
        st.header("1. Model")
        st.caption("⚠️ Loading a joblib file runs code. Only load model files from people you trust.")
        model_upload = st.file_uploader("Model file (.joblib)", type=["joblib", "pkl"], key="model_upload")
        model_path = st.text_input("…or path to the model file", value=default_path("SOC_MODEL_PATH", DEFAULT_MODEL_FILE),
                                   key="model_path", help=f"Default: {DEFAULT_MODEL_FILE}")
        with st.expander("Optional: separate attack-category model"):
            st.caption("Only needed when the main model says BENIGN/ATTACK only and the category model is a "
                       "separate file.")
            family_upload = st.file_uploader("Attack-category model (.joblib)", type=["joblib", "pkl"], key="family_upload")
            family_path = st.text_input("…or path", value=default_path("SOC_FAMILY_MODEL_PATH", DEFAULT_FAMILY_MODEL_FILE),
                                        key="family_path", help=f"Default: {DEFAULT_FAMILY_MODEL_FILE}")
        with st.expander("Optional: benign reference"):
            st.caption("Known-benign flows in the same format as your logs. Used to say which values are unusual. "
                       "Without it, the flows the model classifies as BENIGN in your log are used.")
            reference_upload = st.file_uploader("Benign flows (CSV/JSON/text)", type=["csv", "json", "jsonl", "txt", "log"],
                                                key="reference_upload")
            reference_path = st.text_input("…or path", value="", key="reference_path")

    try:
        model_bytes, model_name = _bytes_from(model_upload, model_path)
        family_bytes, family_name = _bytes_from(family_upload, family_path)
        reference_bytes, reference_name = _bytes_from(reference_upload, reference_path)
    except ValueError as exc:
        st.error(str(exc))
        return

    import sklearn
    if sklearn.__version__ != TRAINING_SKLEARN_VERSION:
        st.warning(f"scikit-learn {sklearn.__version__} is installed but the model was trained with "
                   f"{TRAINING_SKLEARN_VERSION}. Install the same version (`pip install scikit-learn=="
                   f"{TRAINING_SKLEARN_VERSION}`) to avoid loading errors or different predictions.")

    if model_bytes is None:
        st.info(f"**Start here:** put the team's model file at `{DEFAULT_MODEL_FILE}` and refresh the page, "
                "or upload it / type its path in the sidebar.")
        with st.expander("Which model files work?"):
            st.markdown(
                "- A fitted scikit-learn model or Pipeline on the 7 features: BENIGN/ATTACK, or BENIGN plus one class "
                "per attack category.\n"
                "- Both stages in one dict: `{'binary_model': ..., 'family_model': ...}`.\n"
                "- A dict `{'model': ..., 'features': [...], 'class_names': {...}}`.\n\n"
                f"Features: {', '.join(f'`{f}`' for f in HARMONIZED_FEATURES)}.")
        return

    try:
        model, embedded_family = load_model_cached(digest(model_bytes), model_bytes, model_name)
        family_model = load_model_cached(digest(family_bytes), family_bytes, family_name)[0] if family_bytes \
            else embedded_family
    except Exception as exc:  # unpickling can fail in many ways; show it instead of crashing
        st.error(f"Could not load the model: {type(exc).__name__}: {exc}")
        st.caption("Common causes: a different scikit-learn version than the one used for training (1.6.1), or a "
                   "model that uses functions defined only in the training notebook.")
        return

    policy = model.policy
    with st.sidebar:
        st.header("2. Settings")
        start = float(np.clip(round(policy.alert_threshold or DEFAULT_THRESHOLD, 2), 0.05, 0.95))
        threshold = st.slider("Alert threshold on P(ATTACK)", 0.05, 0.95, start, 0.01,
                              key=f"threshold_{digest(model_bytes)[:12]}",
                              help="Stage 1: a flow is an ATTACK when the model's attack probability is at least this "
                                   "value. Lower = fewer missed attacks but more false alarms."
                                   + (f" Starts at the bundle's alert_threshold ({policy.alert_threshold:.2f})."
                                      if policy.alert_threshold else ""))
        st.header("Loaded")
        st.write(f"**Model:** `{model_name}` ({model.description})")

        def describe(scorers):
            for source, scorer in scorers.items():
                classes = [str(c) for c in scorer.classes] if scorer.classes is not None else ["?"]
                st.caption(f"{source}: {len(scorer.columns or [])} input columns · classes "
                           f"{', '.join(classes[:10])}{' …' if len(classes) > 10 else ''}")

        describe(model.scorers)
        if family_model:
            st.write(f"**Attack-category model:** `{family_name or 'inside the main model file'}`")
            describe(family_model.scorers)
        elif all(s.is_binary for s in model.scorers.values()):
            st.info("This model only says BENIGN/ATTACK, so attacks get the category 'Unknown'. Load the "
                    "attack-category model too, if the team has one.")
        if policy.sources:
            with st.expander("Severity & evidence settings"):
                st.markdown("\n".join(f"- `{name}`: {origin}" for name, origin in policy.sources.items()))
                st.caption("Levels: " + " · ".join(f"{severity_badge(l)} {l} ≥ {t:g}" for t, l in policy.severity_levels)
                           + " | Weights: " + ", ".join(f"{k} {v:g}" for k, v in policy.severity_weights.items()))
        if policy.test_metrics is not None:
            with st.expander("Model test metrics (from the team)"):
                st.json(json.loads(json.dumps(policy.test_metrics, default=str)))
        for message in model.load_warnings + (family_model.load_warnings if family_model else []):
            st.warning(message)

    # ---------------- Logs input ----------------
    tab_logs, tab_single, tab_playbooks = st.tabs(["📄 Analyze logs", "✍️ Single log entry", "📘 Playbooks"])
    analysis = None
    with tab_logs:
        left, right = st.columns([2, 1])
        with left:
            logs_upload = st.file_uploader("Flow logs (CSV, JSON, JSON lines or text)",
                                           type=["csv", "json", "jsonl", "txt", "log"], key="logs_upload")
        with right:
            logs_path = st.text_input("…or path to a log file", value="", key="logs_path")
        logs_text = st.text_area(
            "…or paste logs (CSV, JSON or plain text: one flow per line)", height=140, key="logs_text",
            placeholder="flow_duration_seconds=0.52 src_packets=12 dst_packets=10 src_bytes=1500 dst_bytes=9000 "
                        "src_mean_packet_bytes=125 dst_mean_packet_bytes=900\n"
                        "0.01, 2, 0, 80, 0, 40, 0      ← or just the 7 values in feature order")
        try:
            if logs_text.strip() and logs_upload is None and not logs_path.strip():
                logs_bytes, logs_name = logs_text.encode("utf-8"), "pasted.txt"  # format detected from the content
            else:
                logs_bytes, logs_name = _bytes_from(logs_upload, logs_path)
            if logs_bytes is not None:
                logs = read_logs_cached(digest(logs_bytes), logs_bytes, logs_name)
                reference_logs = read_logs_cached(digest(reference_bytes), reference_bytes, reference_name) \
                    if reference_bytes else None
                key = (digest(model_bytes), digest(logs_bytes), digest(family_bytes or b""),
                       digest(reference_bytes or b""))
                analysis = analyze_cached(key, logs, model, threshold, family_model, reference_logs)
        except ValueError as exc:
            st.error(str(exc))
        if analysis is None:
            st.info("Upload, point to, or paste a log to classify it. Raw CIC-IDS2017 or UNSW-NB15 columns are "
                    "converted automatically.")
        else:
            _render_analysis(st, analysis, model, threshold)

    with tab_single:
        _render_single_entry(st, model, threshold, family_model, analysis)

    with tab_playbooks:
        for playbook in PLAYBOOKS:
            with st.expander(f"{playbook['id']}: {playbook['title']}"):
                st.markdown(f"**Families:** {', '.join(playbook['families'])} · "
                            f"**ATT&CK:** {', '.join(playbook['technique_ids']) or '-'}")
                st.markdown("\n".join(f"{n}. {s}" for n, s in enumerate(playbook["steps"], start=1)))


def _render_analysis(st, analysis: Analysis, model: DetectionModel, threshold: float) -> None:
    results = analysis.results
    alerts = results[results["verdict"].eq("ATTACK")].sort_values(["severity_score", "attack_probability"],
                                                                     ascending=False)
    counts = alerts["severity_level"].value_counts()
    cols = st.columns(2 + len(model.policy.severity_levels))
    cols[0].metric("Log rows", f"{len(results):,}")
    cols[1].metric("Alerts", f"{len(alerts):,}")
    for col, (_, level) in zip(cols[2:], model.policy.severity_levels):
        col.metric(f"{severity_badge(level)} {level}", f"{int(counts.get(level, 0)):,}")
    if results["verdict"].eq("UNSCORED").any():
        st.warning(f"{int(results['verdict'].eq('UNSCORED').sum()):,} rows were not scored: the model has no detector "
                   "for their dataset_source.")
    st.caption(f"Evidence compares values with {analysis.reference_note}.")

    table_columns = ["row", "verdict", "confidence", "attack_type", "attack_type_probability", "severity_level",
                     "severity_score", "attack_technique", "playbook", "playbook_similarity", "dataset_source"]
    table_columns += [c for c in results.columns if c.startswith("log_")]
    only_alerts = st.toggle("Show attacks only", value=True, key="only_alerts")
    shown = (alerts if only_alerts else results.sort_values("attack_probability", ascending=False))[table_columns]
    st.dataframe(shown.head(2_000), hide_index=True, column_config={
        "verdict": st.column_config.TextColumn("benign / attack"),
        "confidence": st.column_config.ProgressColumn("confidence", min_value=0.0, max_value=1.0, format="%.3f"),
        "attack_type": st.column_config.TextColumn("attack category"),
        "attack_type_probability": st.column_config.ProgressColumn("category confidence", min_value=0.0,
                                                                   max_value=1.0, format="%.3f"),
        "playbook_similarity": st.column_config.ProgressColumn("playbook similarity", min_value=0.0, max_value=1.0,
                                                               format="%.2f"),
        "severity_score": st.column_config.NumberColumn("severity", format="%.1f"),
    })
    st.download_button("⬇️ Download results (CSV)", results_for_download(analysis).to_csv(index=False).encode("utf-8"),
                       file_name="soc_assistant_results.csv", mime="text/csv")

    st.subheader("Inspect a log row")
    options = list(alerts.index[:500]) or list(results.index[:500])
    left, right = st.columns([3, 1])
    picked = left.selectbox("Most severe alerts" if len(alerts) else "Rows", options, key="picked_row",
                            format_func=lambda i: (f"row {i} · {results.loc[i, 'verdict']} · {results.loc[i, 'attack_type']} · "
                                                   f"{results.loc[i, 'severity_level']} · "
                                                   f"P(ATTACK)={results.loc[i, 'attack_probability']:.3f}"))
    typed = right.text_input("…or type any row number", value="", key="typed_row").strip()
    if typed:
        if not typed.isdigit() or int(typed) not in results.index:
            st.error(f"Row must be a number between {results.index.min()} and {results.index.max()}.")
            return
        picked = int(typed)
    _render_row(st, analysis, int(picked), model)


def _render_row(st, analysis: Analysis, index: int, model: DetectionModel) -> None:
    result = analysis.results.loc[index]
    verdict = result["verdict"]
    level = result["severity_level"]
    is_attack = verdict == "ATTACK"
    cols = st.columns(5)
    cols[0].metric("1 · Benign or attack", verdict)
    cols[1].metric("Confidence", f"{result['confidence']:.1%}" if pd.notna(result["confidence"]) else "n/a",
                   help="The model's probability for the stage-1 answer.")
    cols[2].metric("2 · Attack category", result["attack_type"] if is_attack else "-")
    cols[3].metric("Category confidence",
                   f"{result['attack_type_probability']:.1%}" if is_attack and pd.notna(result["attack_type_probability"])
                   else "n/a", help=f"Probability of this category, given that the flow is an attack "
                                    f"(from the {result['attack_type_from']}).")
    policy = model.policy
    weights = policy.severity_weights
    cols[4].metric("Severity", f"{severity_badge(level) if level != '-' else ''} {level}" +
                   (f" ({result['severity_score']:.1f})" if pd.notna(result["severity_score"]) else ""),
                   help=" · ".join(f"{l} ≥ {t:g}" for t, l in policy.severity_levels) +
                        f". Score = 100 × ({weights['confidence']:g} × attack confidence + {weights['impact']:g} × "
                        f"category impact + {weights['evidence']:g} × evidence strength).")
    has_probability = pd.notna(result["confidence"])
    if has_probability:
        st.progress(float(result["attack_probability"]), text=f"P(ATTACK) = {result['attack_probability']:.3f}")

    if verdict != "ATTACK":
        if verdict == "BENIGN":
            st.success(f"No alert: P(ATTACK) {result['attack_probability']:.3f} is below the threshold." if has_probability
                       else "No alert: the model classifies this flow as BENIGN.")
        else:
            st.warning(f"Row not scored ({result['scored_by']}).")
        return

    attack = map_to_attack(result["attack_type"])
    left, right = st.columns(2)
    with left:
        st.markdown("#### Evidence")
        model_line = (f"gives P(ATTACK) = {result['attack_probability']:.3f}" if has_probability
                      else "classifies it as an attack (this model gives no probability)")
        lines = [f"- **Model:** `{result['scored_by']}` {model_line}",
                 f"- **Attack category:** {result['attack_type']}" + (
                     f" (confidence {result['attack_type_probability']:.2f}, {result['attack_type_from']})"
                     if pd.notna(result["attack_type_probability"]) else f" ({result['attack_type_from']})")]
        lines += [f"- {item['text']}" for item in result["evidence"]] or ["- No feature is outside the benign range."]
        st.markdown("\n".join(lines))
        scorer = model.scorer_for(analysis.logs.loc[index])
        if scorer is not None and analysis.reference:
            drivers = decision_drivers(scorer, analysis.logs.loc[[index]], analysis.reference)
            if not drivers.empty:
                st.markdown("**What drove the decision** (drop in P(ATTACK) when the feature is set to its typical "
                            "benign value):")
                st.dataframe(drivers, hide_index=True, column_config={
                    "probability_change": st.column_config.NumberColumn("P(ATTACK) drop", format="%+.3f")})
        st.markdown("#### MITRE ATT&CK")
        if attack["technique_id"]:
            st.markdown(f"[{attack['technique_id']} {attack['technique_name']}]({attack['url']}) · "
                        f"{attack['tactic']} ({attack['tactic_id']}) · mapping confidence {attack['confidence']}")
        else:
            st.markdown("Unmapped: no defensible ATT&CK technique for this attack type.")
    with right:
        st.markdown("#### Playbook similarity")
        ranked = playbook_similarity(result["attack_type"], tuple(i["feature"] for i in result["evidence"][:3]))
        st.dataframe(ranked[["playbook", "title", "similarity", "reasons"]], hide_index=True,
                     column_config={"similarity": st.column_config.ProgressColumn("similarity", min_value=0.0,
                                                                                  max_value=1.0, format="%.2f")})
        best = PLAYBOOK_BY_ID[ranked["playbook"].iloc[0]]
        st.markdown(f"**Recommended: {best['id']} {best['title']}** (similarity {ranked['similarity'].iloc[0]:.2f})")
        st.markdown("\n".join(f"{n}. {s}" for n, s in enumerate(best["steps"], start=1)))


def _render_single_entry(st, model: DetectionModel, threshold: float, family_model, analysis: Analysis | None) -> None:
    st.caption("Type the 7 features of one flow and click Classify.")
    policy = model.policy
    not_specified = f"(not specified: use {policy.default_source})" if policy.default_source else "(not specified)"
    known = [s for s in (policy.benign_reference or {}) if s != "ALL"] or ["CIC-IDS2017", "UNSW-NB15"]
    sources = list(model.scorers) if model.routed else [not_specified, *known]
    with st.form("single_entry"):
        source = st.selectbox("Dataset (optional: only used to pick the benign reference for the evidence)", sources,
                              key="single_source")
        cols = st.columns(4)
        values = {f: cols[i % 4].number_input(f, value=0.0, min_value=0.0, format="%.6f", key=f"single_{f}")
                  for i, f in enumerate(HARMONIZED_FEATURES)}
        extra = st.text_area("Other columns, only if your model needs more inputs (name=value per line)",
                             key="single_extra", height=68)
        submitted = st.form_submit_button("Classify", type="primary")
    if not submitted:
        return
    record = dict(values) if source == not_specified else {"dataset_source": source, **values}
    for line in extra.splitlines():
        if "=" in line:
            name, value = line.split("=", 1)
            record[name.strip()] = value.strip()
    try:
        single = analyze_logs(pd.DataFrame([record]), model, threshold, family_model,
                              _single_reference(analysis, model))
    except ValueError as exc:
        st.error(str(exc))
        return
    _render_row(st, single, int(single.results.index[0]), model)


def _single_reference(analysis: Analysis | None, model: DetectionModel) -> pd.DataFrame | None:
    """Single entry: the bundle's reference if it has one, else the benign flows of the last analyzed log."""
    if model.policy.benign_reference or analysis is None or not analysis.reference:
        return None
    benign = analysis.results["verdict"].eq("BENIGN")
    return analysis.logs[benign.to_numpy()] if benign.sum() >= MIN_BATCH_REFERENCE_ROWS else None


if __name__ == "__main__":
    main()
