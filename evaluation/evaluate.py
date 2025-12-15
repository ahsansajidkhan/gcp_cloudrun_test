import os
import re
import json
from datetime import datetime
from typing import Dict, List, Tuple
from fastapi import FastAPI
from pydantic import BaseModel
import jiwer

# Optional GCP support
try:
    from google.cloud import storage
    _HAS_GCP = True
except ImportError:
    _HAS_GCP = False


# ==============================
# Parsing & normalization
# ==============================
_SPK_LINE_RE = re.compile(r"^<speaker:(\\d+)>[ \t]+(.+)$")
_TS_LINE_RE  = re.compile(r"^\[\\s*\\d+(?:\\.\\d+)?\\s*-\\s*\\d+(?:\\.\\d+)?\\s*\]\\s*SPEAKER_(\\d{2}):\\s*(.*)$")

def is_diarized_timestamp_format(text: str) -> bool:
    """Detect diarized timestamp format like [1.00 - 2.00] SPEAKER_00: ..."""
    return any(_TS_LINE_RE.match(line.strip()) for line in text.strip().splitlines())

def convert_diarized_to_inline(text: str) -> str:
    """Convert timestamped diarization text to inline <speaker:n> format."""
    out = []
    for line in text.strip().splitlines():
        m = _TS_LINE_RE.match(line.strip())
        if m:
            spk_idx0 = int(m.group(1))
            content = m.group(2).strip()
            out.append(f"<speaker:{spk_idx0 + 1}> {content}")
    return "\n".join(out)

def parse_inline(text: str) -> List[Tuple[int, str]]:
    """Parse lines like: <speaker:1> hello world -> [(1, 'hello world'), ...]"""
    pairs = []
    for line in text.strip().splitlines():
        m = _SPK_LINE_RE.match(line.strip())
        if m:
            spk = int(m.group(1))
            utt = m.group(2).strip()
            if utt:
                pairs.append((spk, utt))
    return pairs


# ==============================
# Metrics: WER / cpWER / WDER
# ==============================
def wer(ref_text: str, hyp_text: str) -> float:
    """Compute WER, with version fallback for jiwer."""
    trans = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip()
    ])
    try:
        return jiwer.wer(ref_text, hyp_text, truth_transform=trans, hypothesis_transform=trans)
    except TypeError:
        return jiwer.wer(trans(ref_text), trans(hyp_text))

def aggregate_by_speaker(pairs: List[Tuple[int, str]]) -> Dict[int, str]:
    """Group utterances by speaker."""
    bag = {}
    for spk, utt in pairs:
        bag.setdefault(spk, []).append(utt)
    return {k: " ".join(v) for k, v in bag.items()}

def best_mapping_cpwer(ref_by_spk: Dict[int, str], hyp_by_spk: Dict[int, str]) -> float:
    """Find best speaker mapping minimizing average speaker WER."""
    import itertools
    ref_ids, hyp_ids = sorted(ref_by_spk), sorted(hyp_by_spk)
    if not ref_ids or not hyp_ids:
        return 1.0
    cache = {(h, r): wer(ref_by_spk[r], hyp_by_spk[h]) for h in hyp_ids for r in ref_ids}
    best_score = float("inf")
    for perm in itertools.permut(ref_ids, r=min(len(hyp_ids), len(ref_ids))):
        pairs = list(zip(hyp_ids, perm))
        per_spk = [cache[(h, r)] for h, r in pairs]
        score = sum(per_spk) / len(per_spk)
        best_score = min(best_score, score)
    return best_score

def compute_wer_cpwer_wder(ref_pairs: List[Tuple[int, str]], hyp_pairs: List[Tuple[int, str]]) -> Dict[str, float]:
    """Compute WER, cpWER, and WDER."""
    ref_text = " ".join(utt for _, utt in ref_pairs)
    hyp_text = " ".join(utt for _, utt in hyp_pairs)
    wer_global = wer(ref_text, hyp_text)
    cpwer_val = best_mapping_cpwer(aggregate_by_speaker(ref_pairs), aggregate_by_speaker(hyp_pairs))
    wder_val = max(0.0, cpwer_val - wer_global)
    return {"WER": wer_global, "cpWER": cpwer_val, "WDER": wder_val}

def read_resources_json(filepath):
    """
    Reads a JSON file (local or GCS) containing resource stats and extracts key metrics.
    Returns duration, cpu, memory, and gpu info.
    """
    content = read_text_from_path(filepath) # Use the GCS-aware reader
    data = json.loads(content)

    result = {
        "duration_s": data.get("duration_s"),
        "cpu": data.get("cpu", {}),
        "memory": data.get("memory", {}),
        "gpu": data.get("gpu", {})
    }

    return result

# ==============================
# Local/GCS read/write
# ==============================
def read_text_from_path(path: str) -> str:
    """Read file from local or GCS"""
    if path.startswith("gs://"):
        if not _HAS_GCP:
            raise RuntimeError("google-cloud-storage not installed.")
        client = storage.Client()
        bucket_name, blob_name = path.replace("gs://", "").split("/", 1)
        bucket = client.bucket(bucket_name)
        return bucket.blob(blob_name).download_as_text()
    else:
        with open(path, "r") as f:
            return f.read()

def save_result_to_path(content: str, path: str):
    """Save JSON result to local or GCS"""
    if path.startswith("gs://"):
        if not _HAS_GCP:
            raise RuntimeError("google-cloud-storage not installed.")
        client = storage.Client()
        bucket_name, blob_name = path.replace("gs://", "").split("/", 1)
        bucket = client.bucket(bucket_name)
        bucket.blob(blob_name).upload_from_string(content, content_type="application/json")
        print(f"Uploaded to {path}")
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        print(f"Saved locally to {path}")


# ==============================
# FastAPI Endpoint
# ==============================
app = FastAPI(title="WER Evaluation API")

class AccuracyRequest(BaseModel):
    hyp_bucket: str
    ref_bucket: str
    res_bucket: str
    output_bucket: str
    hyp_prefix: str
    ref_prefix: str
    res_prefix: str
    output_prefix: str
    compute_der: bool = False


@app.post("/evaluate")
async def evaluate_accuracy(request: AccuracyRequest):
    """Evaluate via API"""
    hyp_path = f"{request.hyp_bucket.rstrip('/')}/{request.hyp_prefix}"
    ref_path = f"{request.ref_bucket.rstrip('/')}/{request.ref_prefix}"
    res_path = f"{request.res_bucket.rstrip('/')}/{request.res_prefix}"
    out_path = f"{request.output_bucket.rstrip('/')}/{request.output_prefix}/eval_result_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"

    ref_text = read_text_from_path(ref_path)
    hyp_text = read_text_from_path(hyp_path)

    if is_diarized_timestamp_format(ref_text):
        ref_text = convert_diarized_to_inline(ref_text)
    if is_diarized_timestamp_format(hyp_text):
        hyp_text = convert_diarized_to_inline(hyp_text)

    ref_pairs = parse_inline(ref_text)
    hyp_pairs = parse_inline(hyp_text)
    metrics = compute_wer_cpwer_wder(ref_pairs, hyp_pairs)
    resource_metrics = read_resources_json(res_path)
    combined = {**metrics,**resource_metrics}

    result = {
        "status": "success",
        "message": "Evaluation completed.",
        "result": combined,
        "timestamp": datetime.utcnow().isoformat(),
        "output_path": out_path
    }

    save_result_to_path(json.dumps(result, indent=2), out_path)
    return result


# ==============================
# Local CLI
# ==============================
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python evaluate_api.py ref.txt hyp.txt resource_metrics.json [output.json]")
        sys.exit(1)

    ref_path, hyp_path, res_path = sys.argv[1], sys.argv[2], sys.argv[3]
    out_path = sys.argv[4] if len(sys.argv) > 4 else "results/local_eval_result.json"


    ref_text = read_text_from_path(ref_path)
    hyp_text = read_text_from_path(hyp_path)

    if is_diarized_timestamp_format(ref_text):
        ref_text = convert_diarized_to_inline(ref_text)
    if is_diarized_timestamp_format(hyp_text):
        hyp_text = convert_diarized_to_inline(hyp_text)

    ref_pairs = parse_inline(ref_text)
    hyp_pairs = parse_inline(hyp_text)
    metrics = compute_wer_cpwer_wder(ref_pairs, hyp_pairs)
    resource_metrics = read_resources_json(res_path)
    combined = {**metrics,**resource_metrics}

    result = {
        "status": "success",
        "result": combined,
        "timestamp": datetime.utcnow().isoformat()
    }

    save_result_to_path(json.dumps(result, indent=2), out_path)
    print(json.dumps(result, indent=2))
    print(f"Saved results to {out_path}")
