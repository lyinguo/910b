"""Verify original MMBench rows, real 910B output, and unfiltered Accuracy."""

import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import sys

MODEL = "CohereLabs/aya-vision-8b"
FILES = {"eval_report.json", "sample_inputs.jsonl", "sample_outputs.jsonl", "run.log", "operator_lineage.json"}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def option(text):
    if text is None:
        return None
    clean = str(text).replace("\u0120", " ").replace("\u010a", " ")
    normalized = clean.strip().strip("*").rstrip('.,;:!?*\'"()[]{}').strip().upper()
    if normalized in ("A", "B", "C", "D", "E"):
        return normalized
    expressions = (
        r"<answer>\s*([A-E])\s*</answer>", r"<[^>]*?>\s*([A-E])\s*</[^>]*?>",
        r"(?:the\s+)?answer\s*(?:is|:)\s*([A-E])", r"(?:correct\s+)?option\s+(?:is\s+)?([A-E])\b",
        r"(?:choose|select|pick)\s+([A-E])\b", r"(?:^|\n)\s*([A-E])\s*$",
        r"^\s*([A-E])\s*\.", r"\b([A-E])\b",
    )
    for i, expression in enumerate(expressions):
        hits = re.findall(expression, clean, re.I | (re.M if i == 5 else 0))
        if hits:
            return (hits[0] if i == 7 else hits[-1]).upper()
    return None

def verify(report_dir, count):
    assert count in (3, 200) and {p.name for p in report_dir.iterdir()} == FILES
    assert re.fullmatch(r"\d{8}_\d{6}", report_dir.name)
    assert report_dir.parent.parent.name == MODEL.replace("/", "_")
    report = load(report_dir / "eval_report.json")
    cfg = report["pipeline_config"]
    assert cfg["model"]["name"] == MODEL and cfg["task_type"] == report_dir.parent.name
    assert cfg["sample_range"] == f"0:{count - 1}" and cfg["algorithm"]["batch_size"] == 10
    assert not cfg["data"]["filter_empty"] and cfg["algorithm"]["kwargs"]["extract_answer"] is True
    assert report["statistics"]["total_samples"] == count and report["summary"]["primary_metric"] == "Accuracy"

    kwargs = cfg["model"]["kwargs"]
    protocol_file = kwargs["protocol_file"]
    assert digest(protocol_file) == kwargs["protocol_sha256"]
    protocol = load(protocol_file)
    evidence_hashes = {protocol_file: kwargs["protocol_sha256"]}
    for path, expected in protocol["static_evidence_sha256"].items():
        assert digest(path) == expected
        evidence_hashes[path] = expected
    archive = load(protocol["original_A100_cohort"])["cohort"]
    assert len(archive) == 200 and sum(
        option(row["prediction"]) == option(row["reference"]) for row in archive
    ) == 169

    line = load(report_dir / "operator_lineage.json")["actual_execution"]
    assert line["device"] == f"npu:{kwargs['npu_devices'][0]}"
    assert line["runtime_dtype"] == "torch.float16" and line["actual_native_serial"]
    assert line["generation_limit"] == 2048 and line["batch_size"] == 10
    assert line["source_resources_sha256"] == protocol["resource_sha256"]
    assert line["protocol_sha256"] == kwargs["protocol_sha256"] and line["parameter_tensor_count"] > 0
    trace_root = Path(line["evidence_directory"])
    assert trace_root.is_relative_to(Path.cwd() / "workspace/diagnostics/model_runs")
    trace_file = Path(line["sample_trace"])
    assert trace_file.parent == trace_root
    inputs = rows(report_dir / "sample_inputs.jsonl")
    outputs = rows(report_dir / "sample_outputs.jsonl")
    details = report["detailed_results"]
    traced = rows(trace_file)
    assert len(inputs) == len(outputs) == len(details) == len(traced) == count
    correct = 0
    empty, invalid, truncated, different = [], [], [], []
    for i, (a, b, c, actual) in enumerate(zip(inputs, outputs, details, traced)):
        reference = archive[i]
        assert a["index"] == b["index"] == c["index"] == actual["index"] == reference["index"] == i
        assert a["prompt"] == reference["prompt"]
        assert a["reference"] == b["reference"] == c["reference"] == reference["reference"]
        assert b["prediction"] == c["prediction"]
        assert b["error"] is None and b["raw_output"] == c["raw_output"] == actual["raw_output"]
        image = Path(a["media_path"])
        assert image.name == reference["media_basename"] and digest(image) == reference["media_sha256"]
        assert Path(actual["media_path"]) == image.resolve()
        assert len(actual["input_ids"]) > 0 and (actual["attention_mask"] is None or len(actual["input_ids"]) == len(actual["attention_mask"]))
        assert actual["pixel_shape"] and actual["generation"]["max_new_tokens"] == 2048
        assert actual["generation"]["do_sample"] is False
        generated = actual["generated_token_ids"]
        assert len(generated) <= 2048
        if not actual["raw_output"].strip():
            empty.append(i)
        prediction = option(b["prediction"])
        if prediction is None:
            invalid.append(i)
        eos = actual["generation"]["eos_token_id"]
        eos_ids = eos if isinstance(eos, list) else [eos]
        if len(generated) == 2048 and generated[-1] not in eos_ids:
            truncated.append(i)
        if actual["raw_output"] != reference["raw_output"]:
            different.append(i)
        assert option(reference["reference"]) is not None
        correct += prediction == option(reference["reference"])

    kernels = {}
    for file in sorted(trace_root.rglob("*")):
        if file.is_file():
            evidence_hashes[str(file)] = digest(file)
            if file.name == "kernel_details.csv":
                with file.open(encoding="utf-8-sig", newline="") as stream:
                    kernels[str(file)] = sum(1 for _ in csv.DictReader(stream))
    assert sum(kernels.values()) > 0
    score = correct / count
    assert score == report["metrics"]["Accuracy"], "Original score must agree with unfiltered option comparison"
    relative = {}
    for key, reference in protocol["A100_D"].items():
        value = Decimal(str(reference))
        assert value > 0
        relative[key] = abs(Decimal(str(score)) - value) / value
    a100 = sum(option(row["prediction"]) == option(row["reference"]) for row in archive[:count]) / count
    same = abs(Decimal(str(score)) - Decimal(str(a100))) / Decimal(str(a100)) if a100 else None
    return {
        "passed": True, "report_dir": str(report_dir), "sample_count": count,
        "metrics": {"Accuracy": score}, "historical_record_reuse": False,
        "actual_NPU_inference": True, "generated_token_count": sum(len(row["generated_token_ids"]) for row in traced),
        "NPU_profiler_kernel_events": sum(kernels.values()), "kernel_counts": kernels,
        "report_sha256": {name: digest(report_dir / name) for name in sorted(FILES)},
        "external_evidence_sha256": evidence_hashes,
        "archived_A100_first_n_Accuracy": a100,
        "archived_A100_first_n_relative_error": None if same is None else str(same),
        "different_A100_raw_indices": different,
        "error_indices": [], "empty_indices": empty, "invalid_indices": invalid, "truncated_indices": truncated,
        "relative_error_to_A100_table_D": float(relative["current_remote"]),
        "relative_error_exact_three_D": {key: str(error) for key, error in relative.items()},
        "formal_alignment_within_5pct": count == 200 and all(
            error <= Decimal(".05") for error in relative.values()
        ),
    }


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1]).resolve(), int(sys.argv[2])), ensure_ascii=False))
