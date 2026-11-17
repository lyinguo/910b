"""Publication rejects unreviewed JavaScript, artifacts and credential literals."""
import hashlib
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('publication_js_boundary', ROOT / 'scripts/publish_code_daily.py')
pub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pub)


class JavaScriptPublicationTests(unittest.TestCase):
    def test_mjs_needs_code_role(self):
        pub.validate_path('scripts/cells.mjs', 'verification_code')
        for role in ('documentation', 'runtime_config', 'publication_policy'):
            with self.subTest(role=role), self.assertRaises(ValueError):
                pub.validate_path('scripts/cells.mjs', role)

    def test_unreviewed_or_changed_javascript_rejected(self):
        raw = b'export const value = 0.88;\n'
        with self.assertRaises(ValueError):
            pub.validate_content('scripts/cells.mjs', raw)
        entry = {'reviewed_javascript_source_SHA256': [hashlib.sha256(raw).hexdigest()]}
        pub.validate_content('scripts/cells.mjs', raw, entry)
        with self.assertRaises(ValueError):
            pub.validate_content('scripts/cells.mjs', raw + b'// change\n', entry)

    def test_javascript_credential_literal_rejected_even_with_digest(self):
        raw = b'const api_key = "not-a-real-secret";\n'
        with self.assertRaises(ValueError):
            pub.validate_content('scripts/cells.mjs', raw, {'reviewed_javascript_source_SHA256': [hashlib.sha256(raw).hexdigest()]})

    def test_large_literal_or_binary_rejected_even_with_digest(self):
        for raw in [b'const payload = "' + b'A' * 32769 + b'";', b'\x00export {}']:
            with self.subTest(size=len(raw)), self.assertRaises(ValueError):
                pub.validate_content('scripts/cells.mjs', raw, {'reviewed_javascript_source_SHA256': [hashlib.sha256(raw).hexdigest()]})

    def test_javascript_in_workspace_still_forbidden(self):
        with self.assertRaises(ValueError):
            pub.validate_path('workspace/diagnostics/code.mjs', 'verification_code')


if __name__ == '__main__':
    unittest.main()
