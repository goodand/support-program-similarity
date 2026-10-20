"""Check staged blob contents, not just working files, before publication."""
import re
import subprocess
from pathlib import PurePosixPath

ROOTS = {'support_program_similarity', 'tests', 'examples', 'tools', 'release', '.github'}
ROOT_FILES = {'.gitignore', 'README.md', 'HANDOFF.md', 'requirements.txt', 'requirements-inference.txt'}
PUBLIC_JSON = {'examples/rankings.synthetic.json', 'examples/positives.synthetic.json', 'release/selection.json'}
PRIVATE_DIRS = {'raw', 'runs', 'data', 'models', '__pycache__', '.git'}
FORBIDDEN = {'.hwp', '.hwpx', '.pdf', '.xls', '.xlsx', '.xlsm', '.xlsb', '.xlt', '.xltx', '.xltm', '.xla', '.xlam',
             '.npy', '.npz', '.gguf', '.safetensors', '.bin', '.pt', '.zip', '.7z', '.tar', '.gz', '.csv', '.tsv', '.jsonl'}


def inspect(path, blob, mode='100644'):
    errors = []
    p = PurePosixPath(path)
    if any(part.lower() in PRIVATE_DIRS for part in p.parts):
        errors.append('private/generated directory')
    if mode not in ('100644', '100755'):
        errors.append('symlink/submodule is not allowed')
    if path not in ROOT_FILES and p.parts[0] not in ROOTS:
        errors.append('outside curated source roots')
    if p.suffix.lower() in FORBIDDEN or p.name.startswith('.env'):
        errors.append('forbidden document, data or runtime extension')
    if len(blob) > 250_000:
        errors.append('unexpectedly large source file')
    try:
        text = blob.decode('utf-8')
    except UnicodeDecodeError:
        return errors + ['non UTF-8/binary payload']
    # Screening is supplemental to manual review; do not print matched secrets.
    if re.search(r'(?i)(?:sk-[a-z0-9_-]{20,}|gh[pousr]_[a-z0-9]{20,}|-{5}BEGIN [A-Z ]*PRIVATE KEY-{5})', text):
        errors.append('possible secret')
    if re.search(r'(?i)[A-Z]:[\\/](?:Users|dev_eval_team)[\\/]', text):
        errors.append('private workstation path')
    if p.suffix.lower() == '.json' and path not in PUBLIC_JSON:
        errors.append('JSON is not an explicitly reviewed synthetic/provenance file')
    return errors


def main():
    entries = subprocess.check_output(['git', 'ls-files', '--stage', '-z']).decode().split('\0')
    errors, count = [], 0
    for entry in filter(None, entries):
        info, path = entry.split('\t', 1)
        mode, oid, stage = info.split()
        blob = subprocess.check_output(['git', 'cat-file', 'blob', oid])
        issues = inspect(path, blob, mode)
        if stage != '0':
            issues.append('unmerged index entry')
        errors.extend(path + ': ' + error for error in issues)
        count += 1
    if not count:
        errors.append('empty index')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'PASS: {count} staged files checked')


if __name__ == '__main__':
    main()

