"""Publication boundaries use only synthetic bytes and temporary Git repositories."""
from pathlib import Path
import subprocess
import tempfile
import unittest

from tools.check_publish import check_index, check_outgoing, inspect


class PublicationTests(unittest.TestCase):
    def test_document_families_are_case_insensitive(self):
        for extension in ('HWP', 'HWPX', 'HPWX', 'PDF', 'XLS', 'XLSX', 'XLSM', 'XLSB',
                          'XLTX', 'XLTM', 'XLAM', 'DOC', 'DOCX', 'RTF', 'PPTX', 'ODS',
                          'ODT', 'ZIP', '7Z', 'RAR', 'TGZ', 'XZ'):
            with self.subTest(extension=extension):
                issues = inspect('examples/original.' + extension, b'synthetic probe')
                self.assertIn('forbidden document, data or runtime extension', issues)

    def test_source_placement_and_exact_json_allowlist(self):
        accepted = ['README.md', '.gitignore', 'requirements.txt', 'requirements-inference.txt',
                    'support_program_similarity/module.py', 'tests/test_module.py',
                    'tools/check.py', 'release/notes.md', 'examples/README.md',
                    '.github/workflows/check.yml', '.github/workflows/check.yaml',
                    'examples/rankings.synthetic.json', 'examples/positives.synthetic.json',
                    'release/selection.json']
        for path in accepted:
            with self.subTest(path=path):
                self.assertEqual(inspect(path, b'{}\n'), [])
        rejected = ['HANDOFF.md', 'tools/source.txt', 'examples/source.txt', 'release/source.xml',
                    'examples/labels.json', 'release/selection.JSON', 'examples/script.py',
                    'support_program_similarity/source.md', '.github/source.yml',
                    'tools/data/source.py', 'outside/script.py', '../README.md', '']
        for path in rejected:
            with self.subTest(path=path):
                self.assertTrue(inspect(path, b'synthetic probe'))

    def test_renamed_pdf_header_in_first_1024_bytes(self):
        for offset in (0, 512, 1023):
            with self.subTest(offset=offset):
                payload = b'A' * offset + b'%' + b'PDF-1.7\n'
                self.assertIn('document/archive signature, including renamed payload',
                              inspect('tools/renamed.py', payload))

    def test_renamed_archive_signatures(self):
        signatures = [b'PK\x03\x04', b'PK\x05\x06', b'PK\x07\x08',
                      b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1', b'\x1f\x8b',
                      b'7z\xbc\xaf\x27\x1c', b'Rar!\x1a\x07', b'BZh', b'\xfd7zXZ\x00']
        for signature in signatures:
            with self.subTest(signature=signature):
                self.assertIn('document/archive signature, including renamed payload',
                              inspect('tools/renamed.py', signature))

    def test_binary_controls_and_environment_names(self):
        for byte in [*range(9), 11, 12, *range(14, 32), 127]:
            with self.subTest(byte=byte):
                self.assertIn('binary control bytes in source file',
                              inspect('tools/renamed.py', b'# probe ' + bytes([byte])))
        self.assertEqual(inspect('tools/source.py', b'# source\r\n\tpass\n'), [])
        for path in ('tests/.env', 'tests/.ENV', 'tests/.Env.local'):
            with self.subTest(path=path):
                self.assertIn('forbidden document, data or runtime extension',
                              inspect(path, b'PLACEHOLDER=value'))


class PublicationGitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='.publication_test_',
                                               dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git('init', '--quiet')
        self.git('config', 'user.name', 'Publication gate test')
        self.git('config', 'user.email', 'publication@example.invalid')
        self.git('config', 'commit.gpgsign', 'false')
        self.git('config', 'core.hooksPath', '.disabled-hooks')
        self.stage('README.md', b'Synthetic repository fixture.\n')
        self.git('commit', '--quiet', '-m', 'Synthetic base')
        self.base = self.git('rev-parse', 'HEAD').strip()

    def git(self, *args):
        return subprocess.check_output(['git', *args], cwd=self.root, stderr=subprocess.PIPE,
                                       text=True, encoding='utf-8')

    def stage(self, path, content):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        self.git('add', '--', path)

    def test_reads_index_blob_instead_of_repaired_working_file(self):
        path = 'tools/probe.py'
        self.stage(path, b'%' + b'PDF-1.7\n')
        (self.root / path).write_text('# now safe in working tree\n', encoding='utf-8')
        issues, count = check_index(self.root)
        self.assertEqual(count, 2)
        self.assertTrue(any('document/archive signature' in issue for issue in issues))
        self.git('add', '--', path)
        self.assertEqual(check_index(self.root), ([], 2))

    def test_outgoing_history_detects_document_deleted_before_push(self):
        path = 'examples/removed.HPWX'
        self.stage(path, b'Synthetic document probe; no original data.\n')
        self.git('commit', '--quiet', '-m', 'Synthetic prohibited intermediate file')
        (self.root / path).unlink()
        self.git('add', '--', path)
        self.git('commit', '--quiet', '-m', 'Remove synthetic probe')
        self.assertEqual(check_index(self.root), ([], 1))
        issues, commits = check_outgoing(self.base, self.root)
        self.assertEqual(commits, 2)
        self.assertTrue(any(path in issue and 'forbidden document' in issue for issue in issues))

    def test_clean_outgoing_source_and_gitlink_rejection(self):
        self.stage('tools/example.py', b'# synthetic source\n')
        self.git('commit', '--quiet', '-m', 'Synthetic source update')
        self.assertEqual(check_outgoing(self.base, self.root), ([], 1))
        self.assertEqual(check_outgoing('HEAD', self.root), ([], 0))
        self.git('update-index', '--add', '--cacheinfo', '160000,' + self.base + ',tools/submodule.py')
        issues, count = check_index(self.root)
        self.assertEqual(count, 3)
        self.assertTrue(any('symlink/submodule is not allowed' in issue for issue in issues))


if __name__ == '__main__':
    unittest.main()
