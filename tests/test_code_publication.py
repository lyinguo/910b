"""Publication boundaries and source relocation checks; no model inference."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("publication", ROOT/"scripts/publish_code_daily.py")
pub = importlib.util.module_from_spec(spec); spec.loader.exec_module(pub)


class PublicationTests(unittest.TestCase):
    def test_no_artifact_or_credential_path(self):
        for name in ["workspace/a.py", "weights/a.py", ".env", "a.safetensors", "a.npz", "a.xlsx", "../a.py", "a\\b.py", ".git/config", "datasets/a.json"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                pub.validate_path(name, "runtime_code")

    def test_json_needs_a_configuration_role(self):
        with self.assertRaises(ValueError):
            pub.validate_path("diagnostic.json", "documentation")
        pub.validate_path("configs/protocols/a.json", "runtime_config")

    def test_no_embedded_secret_or_payload(self):
        for data in [b"password = 'not-a-real-secret'", ("x = " + repr("x"*32769)).encode(), b"\x00x", b"def broken("]:
            with self.subTest(kind=data[:16]), self.assertRaises((ValueError, SyntaxError)):
                pub.validate_content("scripts/a.py", data)

    def test_valid_small_source(self):
        pub.validate_path("scripts/a.py", "runtime_code")
        pub.validate_content("scripts/a.py", b"from pathlib import Path\nprint(Path.cwd())\n")

    def test_dirty_checkout_stops_before_any_snapshot(self):
        responses = [SimpleNamespace(stdout=b"codex/910b-source-layout-20261114\n"),
                     SimpleNamespace(stdout=b"https://github.com/lyinguo/910b.git\n"),
                     SimpleNamespace(stdout=b"?? unrelated.py\0")]
        with patch.object(pub, "git", side_effect=responses), self.assertRaisesRegex(ValueError, "must be clean"):
            pub.plan_snapshot(ROOT, ROOT, ROOT/"configs/publication/code_only.json")

    def test_hovle_methods_are_original_source_segments(self):
        path=ROOT/"universal_eval/model/reference_sources/hovle_original_patches.py"
        text=path.read_text(encoding="utf-8"); tree=ast.parse(text)
        expected={
            "_patch_hovle_prepare_inputs":"38465a246fb4f1302c8d3e64b32c1fe3d3dfcb1cd26bb3a35dc8bf65d3d3bc61",
            "_patch_hovle_causal_mask":"9fef499ca79e15a514efbc64b606c8d006aa5b7c8f09119803562fac3ecff786",
            "_patch_hovle":"2b1f8507de596a9ea567d5655abae61bd351de21fe7e28c68c0ed1407e582310",
        }
        functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        self.assertEqual(set(functions),set(expected))
        for name, sha in expected.items():
            self.assertEqual(hashlib.sha256(ast.get_source_segment(text,functions[name]).encode()).hexdigest(),sha)

    def test_new_hovle_entry_points_to_versioned_source(self):
        # JSON is valid YAML; this new YAML needs no third-party parser for the check.
        cfg=json.loads((ROOT/"configs/InternVLChatModel/OpenGVLab--HoVLE-HD-910b-original-BF16-source-layout-v5.yaml").read_text(encoding="utf-8"))
        prefix="/root/projects/comperror/"
        rel=cfg["model"]["kwargs"]["protocol_file"].removeprefix(prefix)
        raw=(ROOT/rel).read_bytes(); p=json.loads(raw)
        self.assertEqual(hashlib.sha256(raw).hexdigest(),cfg["model"]["kwargs"]["protocol_sha256"])
        source=p["original_runtime_patch_source"].removeprefix(prefix)
        self.assertFalse(source.startswith("workspace/"))
        self.assertEqual(hashlib.sha256((ROOT/source).read_bytes()).hexdigest(),p["original_runtime_patch_sha256"])


if __name__ == "__main__":
    unittest.main()
