"""Independent sample, runtime, artifact, and score verification."""

import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import sys


MODEL = "ibm-granite/granite-vision-3.1-2b-preview"
STANDARD = {
    "eval_report.json", "sample_inputs.jsonl", "sample_outputs.jsonl",
    "run.log", "operator_lineage.json",
}


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
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
        r"^\s*([A-E])\s*\.",
    )
    for i, expression in enumerate(expressions):
        hits = re.findall(expression, clean, re.I | (re.M if i == 5 else 0))
        if hits:
            return hits[-1].upper()
    return None

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
    assert report["summary"]["primary_metric"] == "Accuracy"
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
    original_deployment = read(protocol["original_A100_deployment"])
    assert original_deployment["frozen_A100_five"]["eval_report.json"] == protocol["original_A100_report_SHA256"]
    archive = original_deployment["cohort"]
    assert len(archive) == 200
    assert original_deployment["configuration"]["data"]["filter_empty"] is True
    assert protocol["fixed_original_sample_set_verified_nonempty"]
    assert all(row["prompt"].strip() and row["reference"].strip() for row in archive)
    assert all(row["index"] == i and option(row["raw_output"]) == option(row["prediction"]) and option(row["reference"]) is not None
               for i, row in enumerate(archive))
    original = sum(int(option(row["prediction"]) == option(row["reference"])) for row in archive) / len(archive)
    assert abs(original - protocol["original_A100_score"]) < 1e-10

    line = read(folder / "operator_lineage.json")["actual_execution"]
    assert line["device_type"] == "npu" and line["tensor_parallel_size"] == 1
    assert line["engine"] == "vLLM Ascend" and line["quantization"] is None
    assert line["runtime_dtype"] == "torch.bfloat16" and line["worker_start_method"] == "spawn"
    assert line["max_output_tokens"] == 256 and line["effective_max_model_len"] == 16384 and line["requested_physical_npu"] == arguments["npu_devices"][0]
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
        assert b["prediction"] == c["prediction"]
        assert option(b["prediction"]) == option(b["raw_output"])
        assert actual["prompt"].startswith(expected["prompt"])
        media = Path(a["media_path"])
        assert media.name == expected["media_basename"]
        assert actual["media_uri"] == f"file://{media.resolve()}"
        assert digest(media) == expected["media_sha256"]
        assert actual["group_index"] == i // 10
        assert actual["group_size"] == min(10, count - 10 * (i // 10))
        assert actual["prompt_token_ids"].count(49155) > 0
        tokens = actual["generated_token_ids"]
        assert len(tokens) <= 256 and all(0 <= token < 49156 for token in tokens)
        if not actual["raw_output"].strip():
            empty.append(i)
        if option(b["prediction"]) is None:
            invalid.append(i)
        if actual["finish_reason"] == "length":
            truncated.append(i)
        else:
            assert actual["finish_reason"] == "stop"
        scores.append(int(option(b["prediction"]) == option(b["reference"])))
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
    assert abs(score - report["metrics"]["Accuracy"]) < 1e-10
    archived_subset = sum(int(option(row["prediction"]) == option(row["reference"])) for row in archive[:count]) / count
    differences = {key: relative(score, value) for key, value in protocol["A100_D"].items()}
    return {
        "passed": True, "report_dir": str(folder), "sample_count": count,
        "metrics": {"Accuracy": score}, "historical_record_reuse": False,
        "actual_NPU_inference": True, "error_indices": [], "empty_indices": empty,
        "invalid_indices": invalid, "truncated_indices": truncated,
        "report_sha256": {name: digest(folder / name) for name in sorted(STANDARD)},
        "external_evidence_sha256": external, "NPU_profiler_kernel_events": sum(kernels.values()),
        "kernel_counts": kernels, "generated_token_count": sum(len(row["generated_token_ids"]) for row in trace),
        "archived_A100_first_n_Accuracy": archived_subset,
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
