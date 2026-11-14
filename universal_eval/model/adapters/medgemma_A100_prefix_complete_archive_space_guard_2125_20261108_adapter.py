"""Keep room for complete MedGemma evidence without touching inference or array bytes."""
import json
import shutil
from pathlib import Path

from universal_eval.model.adapters.medgemma_vllm_A100_native_lineage_v8_original_prefix_2125_20261108_adapter import (
    MedGemmaVllmA100NativeLineageV3Adapter,
)


class MedGemmaOriginalPrefixCompleteArchiveAdapter(MedGemmaVllmA100NativeLineageV3Adapter):
    reserve_bytes = 8 * 1024**3
    pending_log_budget = 2 * 1024**3

    def load(self, model_path, **kwargs):
        super().load(model_path, **kwargs)
        self._space_evidence = self.evidence / "archive_space_guard.jsonl"
        self._space_evidence.touch(exist_ok=False)
        self.lineage["complete_archive_space_guard"] = {
            "receipt": str(self._space_evidence),
            "reserve_bytes": self.reserve_bytes,
            "pending_log_budget": self.pending_log_budget,
            "checks_during_actual_execution": "Before each original group and on every full-vocabulary forward",
        }

    def _check_archive_space(self):
        free = shutil.disk_usage(self.evidence).free
        required = self.reserve_bytes + self.pending_log_budget + self._archive_bound
        if free < required:
            raise OSError(f"Complete archive space is insufficient: {free} bytes free, {required} required")
        self._last_free_before_archive = free
        self._archive_space_checks += 1

    def _tensor_record(self, value, capture_key=None, full_hash=False):
        if capture_key is not None and capture_key not in self.arrays:
            size = value.numel() * value.element_size()
            # Conservative DEFLATE overhead and both ZIP/NPY headers per uint8 array.
            self._archive_bound += size + (size // 16) + 1024 + 4 * len(capture_key)
            if capture_key.endswith("full262208_BF16_logits"):
                self._check_archive_space()
        return super()._tensor_record(value, capture_key, full_hash)

    def batch_predict(self, batch):
        self._archive_bound = 65536
        self._archive_space_checks = 0
        self._check_archive_space()
        outputs = super().batch_predict(batch)
        group = self.groups[-1]
        actual_bytes = Path(group["archive"]).stat().st_size
        assert actual_bytes <= self._archive_bound
        free_after = shutil.disk_usage(self.evidence).free
        assert free_after >= self.reserve_bytes
        with self._space_evidence.open("a", encoding="utf-8") as stream:
            json.dump({"group": len(self.groups) - 1, "archive": group["archive"],
                       "archive_SHA256": group["archive_SHA256"], "actual_archive_bytes": actual_bytes,
                       "archive_byte_upper_bound": self._archive_bound,
                       "checks": self._archive_space_checks, "last_free_before_archive": self._last_free_before_archive,
                       "free_after_archive": free_after}, stream, ensure_ascii=False)
            stream.write("\n")
        return outputs
