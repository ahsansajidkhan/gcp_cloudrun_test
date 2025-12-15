# batch_evaluation_with_resources.py
import os
import json
import csv
from datetime import datetime
from typing import Optional, Dict, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from evaluation.evaluate import (
    read_text_from_path,
    save_result_to_path,
    is_diarized_timestamp_format,
    convert_diarized_to_inline,
    parse_inline,
    compute_wer_cpwer_wder,
    read_resources_json
)

def flatten_summary(data, parent_key=""):
    flat = {}
    for k, v in data.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        # If it's a dict and its values are not all (int or float), then flatten further
        if isinstance(v, dict) and not (v and all(isinstance(x, (int, float)) for x in v.values())):
            flat.update(flatten_summary(v, new_key))
        else:
            # Otherwise, it's a primitive value or a terminal dict of numbers, so add it directly
            flat[new_key] = v
    return flat

# ==============================
# Core batch evaluation logic
# ==============================
def batch_eval(
    hyp_dir_or_bucket: str,
    ref_dir_or_bucket: str,
    output_root_or_bucket: str,
    resources_dir_or_bucket: str
) -> Dict:
    """
    Batch evaluate all .txt files under hyp_dir_or_bucket, matched with corresponding
    reference files in ref_dir_or_bucket (same prefix or with _ref/_reference/_gt suffix).
    Supports both local paths and GCS (gs://...) using evaluate.py utilities.
    """

    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    report_name = f"report_{timestamp}"
    report_dir = os.path.join(output_root_or_bucket, report_name)

    # Create local folder only if it's not GCS
    if not report_dir.startswith("gs://"):
        os.makedirs(report_dir, exist_ok=True)

    # Gather hyp files (local or GCS)
    hyp_files = []
    if hyp_dir_or_bucket.startswith("gs://"):
        print(f"[DEBUG] Processing GCS path for hypotheses: {hyp_dir_or_bucket}")
        from google.cloud import storage
        client = storage.Client()
        bucket_parts = hyp_dir_or_bucket.replace("gs://", "").split("/", 1)
        bucket_name = bucket_parts[0]
        # Handle case where prefix might be empty (root of bucket)
        prefix = bucket_parts[1] if len(bucket_parts) > 1 else ""

        print(f"[DEBUG] GCS Bucket Name: {bucket_name}, Prefix: {prefix}")
        # list_blobs with a prefix will list objects that start with that prefix.
        # If prefix ends with '/', it effectively acts as a directory.
        blobs = list(client.bucket(bucket_name).list_blobs(prefix=prefix))
        print(f"[DEBUG] Raw blobs found: {[b.name for b in blobs]}")

        # Filter for .txt files that are directly under the prefix (not in subfolders)
        hyp_files = [b.name for b in blobs if b.name.lower().endswith(".txt") and (os.path.dirname(b.name.replace(prefix, '')) == '' or os.path.dirname(b.name.replace(prefix, '')) == '/')]

        print(f"[DEBUG] Filtered .txt hyp_files: {hyp_files}")
    else:
        print(f"[DEBUG] Processing local path for hypotheses: {hyp_dir_or_bucket}")
        hyp_files = [f for f in os.listdir(hyp_dir_or_bucket) if f.lower().endswith(".txt")]
        print(f"[DEBUG] Local .txt hyp_files: {hyp_files}")

    if not hyp_files:
        raise HTTPException(status_code=400, detail="No .txt files found under hyp/")

    results, success_files, failed_files = [], [], []

    for hyp_file_full_path in hyp_files: # hyp_file_full_path will be like 'github_test/hyp/audio1.txt'
        # Extract the base filename without extension to use as prefix for matching
        # os.path.basename will get 'audio1.txt'
        # os.path.splitext will get 'audio1'
        prefix = os.path.splitext(os.path.basename(hyp_file_full_path))[0]
        
        # Construct the full path for the hypothesis file (GCS or local)
        if hyp_dir_or_bucket.startswith("gs://"):
            # For GCS, hyp_file_full_path already contains the full object name
            hyp_path = f"gs://{bucket_name}/{hyp_file_full_path}"
        else:
            hyp_path = os.path.join(hyp_dir_or_bucket, hyp_file_full_path)

        # Try to find matching ref
        matched_ref = None
        candidates = [f"{prefix}.txt", f"{prefix}_ref.txt", f"{prefix}_reference.txt", f"{prefix}_gt.txt"]
        json_candidates = [f"{prefix}.json", f"{prefix}_ref.json"]

        print(f"[DEBUG] Searching for ref for hyp '{prefix}' with candidates: {candidates}")

        if ref_dir_or_bucket.startswith("gs://"):
            # Fix: Make ref_prefix derivation consistent with hyp_prefix derivation
            ref_bucket_parts = ref_dir_or_bucket.replace("gs://", "").split("/", 1)
            ref_bucket_name = ref_bucket_parts[0]
            ref_prefix = ref_bucket_parts[1] if len(ref_bucket_parts) > 1 else ""
            ref_client = storage.Client()
            
            print(f"[DEBUG] GCS Ref Bucket Name: {ref_bucket_name}, Ref Prefix: {ref_prefix}")
            ref_blobs = list(ref_client.bucket(ref_bucket_name).list_blobs(prefix=ref_prefix))
            print(f"[DEBUG] Raw Ref blobs found for prefix '{ref_prefix}': {[b.name for b in ref_blobs]}")

            for rb in ref_blobs:
                ref_blob_filename = os.path.basename(rb.name) # Get just the filename part
                for c in candidates:
                    print(f"[DEBUG_REF_MATCH] Comparing Ref blob filename '{ref_blob_filename}' with candidate '{c}'")
                    if ref_blob_filename == c: # Use exact match on basename
                        matched_ref = f"gs://{ref_bucket_name}/{rb.name}"
                        print(f"[DEBUG_REF_MATCH] Matched ref: {matched_ref}")
                        break
                if matched_ref:
                    break
        else:
            for fn in os.listdir(ref_dir_or_bucket):
                if fn in candidates:
                    matched_ref = os.path.join(ref_dir_or_bucket, fn)
                    break

        if not matched_ref:
            print(f"[WARNING] No matching reference found for {hyp_file_full_path}. Skipping.")
            failed_files.append(hyp_file_full_path)
            continue

        matched_res = None # Initialize matched_res for each iteration
        print(f"[DEBUG] Searching for resources for hyp '{prefix}' with candidates: {json_candidates}")
        if resources_dir_or_bucket.startswith("gs://"):
            # Fix: Make res_prefix derivation consistent with hyp_prefix derivation
            res_bucket_parts = resources_dir_or_bucket.replace("gs://", "").split("/", 1)
            res_bucket_name = res_bucket_parts[0]
            res_prefix = res_bucket_parts[1] if len(res_bucket_parts) > 1 else ""
            res_client = storage.Client()
            
            print(f"[DEBUG] GCS Res Bucket Name: {res_bucket_name}, Res Prefix: {res_prefix}")
            res_blobs = list(res_client.bucket(res_bucket_name).list_blobs(prefix=res_prefix))
            print(f"[DEBUG] Raw Res blobs found for prefix '{res_prefix}': {[b.name for b in res_blobs]}")

            for resources in res_blobs:
                res_blob_filename = os.path.basename(resources.name)
                for c in json_candidates:
                    print(f"[DEBUG_RES_MATCH] Comparing Res blob filename '{res_blob_filename}' with candidate '{c}'")
                    if res_blob_filename == c:
                        matched_res = f"gs://{res_bucket_name}/{resources.name}"
                        print(f"[DEBUG_RES_MATCH] Matched resource: {matched_res}")
                        break
                if matched_res:
                    break
        else:
            for fn in os.listdir(resources_dir_or_bucket):
                if fn in json_candidates:
                    matched_res = os.path.join(resources_dir_or_bucket, fn)
                    break

        if not matched_res:
            print(f"[WARNING] No matching resources JSON found for {hyp_file_full_path}. Skipping.")
            failed_files.append(hyp_file_full_path)
            continue

        try:
            print(f"[INFO] Evaluating {hyp_file_full_path} against {matched_ref} with resources {matched_res}")
            # Read and preprocess
            ref_text = read_text_from_path(matched_ref)
            hyp_text = read_text_from_path(hyp_path)

            if is_diarized_timestamp_format(ref_text):
                ref_text = convert_diarized_to_inline(ref_text)
            if is_diarized_timestamp_format(hyp_text):
                hyp_text = convert_diarized_to_inline(hyp_text)

            ref_pairs = parse_inline(ref_text)
            hyp_pairs = parse_inline(hyp_text)
            metrics = compute_wer_cpwer_wder(ref_pairs, hyp_pairs)
            resource_metrics = read_resources_json(matched_res)
            combined = {**metrics, **resource_metrics}

            # Save result
            result_name = f"{prefix}_result.json"
            result_path = f"{report_dir}/{result_name}"
            save_result_to_path(json.dumps(combined, indent=2), result_path)
            
            csv_result_path = f"{report_dir}/{prefix}_result.csv"
            # Ensure the directory exists before writing the CSV
            if not csv_result_path.startswith("gs://"):
                os.makedirs(os.path.dirname(csv_result_path), exist_ok=True)
            
            # Open the file for writing. If it's a GCS path, save_result_to_path will handle it.
            # For local, use standard open.
            if not csv_result_path.startswith("gs://"):
                with open(csv_result_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Metric", "Value"])
                    for key, value in combined.items():
                        if isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                writer.writerow([f"{key}.{subkey}", subvalue])
                        else:
                            writer.writerow([key, value])
            else:
                # For GCS, prepare CSV content and use save_result_to_path
                csv_content = []
                csv_content.append("Metric,Value")
                for key, value in combined.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            csv_content.append(f"{key}.{subkey},{subvalue}")
                    else:
                        csv_content.append(f"{key},{value}")
                save_result_to_path("\n".join(csv_content), csv_result_path)

            success_files.append(hyp_file_full_path)
            results.append(combined)

        except Exception as e:
            failed_files.append(hyp_file_full_path)
            print(f"[ERROR] Error on {hyp_file_full_path}: {e}")

    # Aggregate
    summary_data = {}
    if results:
        summary_data = {}
        for key in results[0]:
            # If the value is a dictionary (like cpu, memory, gpu)
            if isinstance(results[0][key], dict):
                summary_data[key] = {} 
                for subkey in results[0][key]:
                    values = [r[key][subkey] for r in results if key in r and subkey in r[key]]
                    if values:
                        summary_data[key][subkey] = {
                            "avg": sum(values) / len(values),
                            "min": min(values),
                            "max": max(values),
                        }
            else:
                # Regular flat metric
                values = [r[key] for r in results if key in r]
                if values:
                    summary_data[key] = {
                        "avg": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                    }
        # summary_data = avg
        flat_summary = flatten_summary(summary_data)

        summary_txt = (
            "Batch Evaluation Summary\n"
            f"Total files: {len(results)}\n"
            + "\n".join([
                f"{k}: avg={v['avg']:.4f}, min={v['min']:.4f}, max={v['max']:.4f}"
                for k, v in flat_summary.items()
            ])
        )
        summary_path = f"{report_dir}/summary.txt"
        save_result_to_path(summary_txt, summary_path)


    # Final report
    report_data = {
        "timestamp": timestamp,
        "total_files": len(hyp_files),
        "success_count": len(success_files),
        "failed_count": len(failed_files),
        "success_files": success_files,
        "failed_files": failed_files,
        "batch_metrics": summary_data,
        "report_dir": report_dir,
    }

    report_path = f"{report_dir}/report.json"
    save_result_to_path(json.dumps(report_data, indent=2), report_path)


    return report_data


# ==============================
# FastAPI Endpoint
# ==============================
app = FastAPI(title="Batch WER Evaluation API")

class BatchEvalRequest(BaseModel):
    hyp_dir_or_bucket: str
    ref_dir_or_bucket: str
    output_root_or_bucket: str = "results"
    resources_dir_or_bucket: str


@app.post("/batch-evaluate")
async def batch_evaluate(request: BatchEvalRequest):
    try:
        report = batch_eval(
            request.hyp_dir_or_bucket,
            request.ref_dir_or_bucket,
            request.output_root_or_bucket,
            request.resources_dir_or_bucket
        )
        return {"status": "success", "report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================
# Local CLI
# ==============================
if __name__ == "__main__":
    import sys, uvicorn

    if len(sys.argv) == 5:
        hyp_dir, ref_dir, out_dir, resources_dir = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
        print("\n============================")
        print("Starting Local Batch Evaluation")
        print("============================")
        print(f"Hypothesis directory (model outputs): {hyp_dir}")
        print(f"Reference directory (ground truth):   {ref_dir}")
        print(f"Resources directory: {resources_dir} ")
        print(f"Results will be saved under:          {out_dir}\n")

        result = batch_eval(hyp_dir, ref_dir, out_dir,resources_dir)

        print("\n============================")
        print("Batch Evaluation Completed")
        print("============================")
        print(f"Total files processed: {result['total_files']}")
        print(f"Successful evaluations: {result['success_count']}")
        print(f"Failed evaluations:     {result['failed_count']}\n")

        report_dir = result["report_dir"]
        print("Generated files:")
        print(f"  ├─ {report_dir}/summary.txt   ← Average metrics (WER, cpWER, WDER)")
        print(f"  ├─ {report_dir}/report.json   ← Full metadata (success, fail, avg)")
        print(f"  └─ {report_dir}/*_result.json ← Individual file results\n")
        print(f"  └─ {report_dir}/*_result.csv ← Individual file results in CSV format\n")

        print("Summary of average metrics:")
        for k, v in result.get("batch_metrics", {}).items(): # Changed to batch_metrics as per report_data structure
            if isinstance(v, dict) and 'avg' in v: # Handle cases where values might be nested dicts themselves
                print(f"  {k}: avg={v['avg']:.4f}, min={v['min']:.4f}, max={v['max']:.4f}")
            else:
                print(f"  {k}: {v}") # Print as is if not the expected avg/min/max dict


        print("\nAll results saved successfully.\n")

    else:
        print("Usage:")
        print("  python batch_evaluate_api.py <hyp_dir> <ref_dir> <output_dir>")
        print("\nExample:")
        print("  python batch_evaluate_api.py hyp ref results")
        print("\nDescription:")
        print("  <hyp_dir>     Folder containing hypothesis .txt files (model outputs).")
        print("  <ref_dir>     Folder containing reference .txt files (ground truth).")
        print("  <output_dir>  Destination folder for generated reports and summaries.")
        print("  <resources_dir>  Folder containing resources .json files.")
