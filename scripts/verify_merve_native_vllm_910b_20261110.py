"""Independent sample, runtime, artifact, and score verification."""

import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import sys

from verify_TextCapsBase_A100_FP16_disclosed_v3_0527_20261107 import rouge_l

MODEL = "merve/paligemma_vqav2"
STANDARD = {
    "eval_report.json", "sample_inputs.jsonl", "sample_outputs.jsonl",
    "run.log", "operator_lineage.json",
}
PATTERNS = (
    (r"<answer>\s*([A-Ea-e]|\d+(?:\.\d+)?)\s*</answer>", re.I),
    (r"\\boxed\{([^}]+)\}", 0),
    (r"(?:the\s+)?answer\s*(?:is|:)\s*([^\n.]+)", re.I),
    (r"correct\s+option\s+is\s*\**(\w+)\**", re.I),
    (r"correct\s+answer\s*(?:is|:)\s*([^\n.]+)", re.I),
    (r"(?:the\s+)?(?:number|result|value|amount|length)\s+is\s+([^\n.]+)", re.I),
    (r"(?:therefore|so|hence|thus|conclusion)[,.]?\s+.+?\s+is\s+([^\n.]+?)(?:\.|\n|$)", re.I),
    (r"\*\*([^*]+?)\*\*", 0),
    (r"=\s*(\d+(?:\.\d+)?)\s*$", re.M),
    (r"(?:^|\n)\s*([A-Ea-e]|\d+(?:\.\d+)?)\s*$", re.M),
    (r"([A-Ea-e])\s*$", re.M),
)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def extract(raw):
    if not raw:
        return raw
    clean = raw.strip().replace("\u0120", " ").replace("\u010a", " ")
    for expression, flags in PATTERNS:
        matches = re.findall(expression, clean, flags)
        if matches:
            candidate = matches[-1].strip().rstrip("*").strip()
            if candidate:
                return candidate
    return raw


def relative(score, reference):
    score, reference = Decimal(str(score)), Decimal(str(reference))
    return abs(score - reference) / abs(reference) if reference else (
        Decimal(0) if not score else Decimal("Infinity")
    )


def verify(folder, count):
    assert count in (3, 10, 200)
    assert folder.parent.parent.name == MODEL.replace("/", "_")
    assert re.fullmatch(r"\d{8}_\d{6}", folder.name)
    assert {file.name for file in folder.iterdir()} == STANDARD
    report = read(folder / "eval_report.json")
    cfg = report["pipeline_config"]
    assert cfg["model"]["name"] == MODEL and cfg["task_type"] == folder.parent.name
    assert cfg["sample_range"] == f"0:{count - 1}"
    assert cfg["algorithm"]["batch_size"] == 10 and cfg["algorithm"]["kwargs"]["extract_answer"] is True
    assert not cfg["model"]["download"] and not cfg["data"]["download"] and not cfg["data"]["filter_empty"]
    assert report["summary"]["primary_metric"] == "ROUGE-L"
    assert report["statistics"]["total_samples"] == count

    arguments = cfg["model"]["kwargs"]
    protocol_file = Path(arguments["protocol_file"])
    assert digest(protocol_file) == arguments["protocol_sha256"]
    protocol = read(protocol_file)
    assert protocol["model_id"] == MODEL and cfg["model"]["model_path"] == protocol["weight_path"]
    external = {str(protocol_file): digest(protocol_file)}
    for path, expected in protocol["static_evidence_sha256"].items():
        assert digest(path) == expected
        external[path] = expected
    archive = read(protocol["original_A100_cohort"])["rows"]
    reference_images = read(protocol["original_A100_deployment"])["media"]
    assert len(archive) == len(reference_images) == 200
    assert all(row["index"] == i and extract(row["raw_output"]) == row["prediction"]
               for i, row in enumerate(archive))
    original = sum(rouge_l(row["prediction"], row["reference"]) for row in archive) / len(archive)
    assert abs(original - protocol["original_A100_score"]) < 1e-10

    line = read(folder / "operator_lineage.json")["actual_execution"]
    assert line["device_type"] == "npu" and line["tensor_parallel_size"] == 1
    assert line["engine"] == "vLLM Ascend" and line["quantization"] is None
    assert line["max_output_tokens"] == 2048 and line["requested_physical_npu"] == arguments["npu_devices"][0]
    assert line["protocol_sha256"] == arguments["protocol_sha256"]
    assert line["source_resources_sha256"] == protocol["resource_sha256"]
    evidence = Path(line["evidence_directory"])
    assert evidence.is_relative_to(Path.cwd() / "workspace/diagnostics/model_runs")
    trace_file = Path(line["request_trace"])
    assert trace_file.parent == evidence
    trace = rows(trace_file)
    inputs = rows(folder / "sample_inputs.jsonl")
    outputs = rows(folder / "sample_outputs.jsonl")
    details = report["detailed_results"]
    assert len(inputs) == len(outputs) == len(details) == len(trace) == count

    scores = []
    empty, invalid, truncated = [], [], []
    for i, (a, b, c, actual) in enumerate(zip(inputs, outputs, details, trace)):
        expected = archive[i]
        assert a["index"] == b["index"] == c["index"] == actual["index"] == expected["index"] == i
        assert a["prompt"] == expected["prompt"]
        assert a["reference"] == b["reference"] == c["reference"] == expected["reference"]
        assert b["raw_output"] == c["raw_output"] == actual["raw_output"] and b["error"] is None
        assert b["prediction"] == c["prediction"] == extract(b["raw_output"])
        assert actual["prompt"] == expected["prompt"] + "\n\nProvide only the final answer."
        media = Path(a["media_path"])
        assert media.name == expected["media_basename"] == reference_images[i]["basename"]
        assert actual["media_uri"] == f"file://{media.resolve()}"
        assert digest(media) == reference_images[i]["sha256"]
        assert actual["group_index"] == i // 10
        assert actual["group_size"] == min(10, count - 10 * (i // 10))
        assert actual["prompt_token_ids"].count(257152) == 256
        tokens = actual["generated_token_ids"]
        assert len(tokens) <= 2048 and all(0 <= token < 257216 for token in tokens)
        if not actual["raw_output"].strip():
            empty.append(i)
        if not b["prediction"] or not b["prediction"].strip():
            invalid.append(i)
        if actual["finish_reason"] == "length":
            truncated.append(i)
        else:
            assert actual["finish_reason"] == "stop"
        scores.append(rouge_l(b["prediction"], b["reference"]))
    assert len({row["request_id"] for row in trace}) == count

    kernels = {}
    for file in sorted(evidence.rglob("*")):
        if not file.is_file():
            continue
        external[str(file)] = digest(file)
        if file.name in ("kernel_details.csv", "op_summary.csv"):
            with file.open(encoding="utf-8-sig", newline="") as stream:
                operations = list(csv.DictReader(stream))
            if operations:
                kernels[str(file)] = len(operations)
    assert sum(kernels.values()) > 0, "No real NPU profiler kernel records"
    score = sum(scores) / count
    assert abs(score - report["metrics"]["ROUGE-L"]) < 1e-10
    archived_subset = sum(rouge_l(row["prediction"], row["reference"]) for row in archive[:count]) / count
    differences = {key: relative(score, value) for key, value in protocol["A100_D"].items()}
    return {
        "passed": True, "report_dir": str(folder), "sample_count": count,
        "metrics": {"ROUGE-L": score}, "historical_record_reuse": False,
        "actual_NPU_inference": True, "error_indices": [], "empty_indices": empty,
        "invalid_indices": invalid, "truncated_indices": truncated,
        "report_sha256": {name: digest(folder / name) for name in sorted(STANDARD)},
        "external_evidence_sha256": external, "NPU_profiler_kernel_events": sum(kernels.values()),
        "kernel_counts": kernels, "generated_token_count": sum(len(row["generated_token_ids"]) for row in trace),
        "archived_A100_first_n_ROUGE_L": archived_subset,
        "archived_A100_first_n_relative_error": str(relative(score, archived_subset)),
        "raw_output_matches_archived_A100": sum(row["raw_output"] == archive[i]["raw_output"]
                                               for i, row in enumerate(trace)),
        "relative_error_to_A100_table_D": float(differences["current_remote"]),
        "relative_error_exact_three_D": {key: str(value) for key, value in differences.items()},
        "formal_alignment_within_5pct": count == 200 and all(
            value <= Decimal("0.05") for value in differences.values()
        ),
    }


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1]).resolve(), int(sys.argv[2])), ensure_ascii=False))
