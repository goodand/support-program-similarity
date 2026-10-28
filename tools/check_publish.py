"""Check indexed files and optionally outgoing commit trees before publication."""
import argparse
import re
import subprocess
from pathlib import PurePosixPath

SOURCE_SUFFIXES = {
    'support_program_similarity': {'.py'},
    'tests': {'.py'},
    'tools': {'.py'},
    'examples': {'.md'},
    'release': {'.md'},
    'experiments': {'.py', '.md', '.sh'},
}
ROOT_FILES = {'.gitignore', 'README.md', 'requirements.txt', 'requirements-inference.txt'}
PUBLIC_JSON = {'examples/rankings.synthetic.json', 'examples/positives.synthetic.json', 'release/selection.json',
               'experiments/kim_db_chunks/config.json', 'experiments/kim_db_chunks/chunk_config.json'}
PRIVATE_DIRS = {'raw', 'runs', 'data', 'models', '__pycache__', '.git'}
FORBIDDEN = {
    '.hwp', '.hwpx', '.hpwx', '.hwt', '.hml', '.pdf',
    '.xls', '.xlsx', '.xlsm', '.xlsb', '.xlt', '.xltx', '.xltm', '.xla', '.xlam', '.xlw', '.xlm',
    '.doc', '.docx', '.docm', '.dot', '.dotx', '.dotm', '.rtf',
    '.ppt', '.pptx', '.pptm', '.pot', '.potx', '.potm', '.pps', '.ppsx', '.ppsm',
    '.odt', '.ods', '.odp', '.ott', '.ots', '.otp', '.fodt', '.fods', '.fodp',
    '.npy', '.npz', '.gguf', '.safetensors', '.bin', '.pt', '.csv', '.tsv', '.jsonl',
    '.zip', '.7z', '.rar', '.tar', '.gz', '.tgz', '.bz2', '.xz', '.lz', '.lzma', '.zst',
}
ARCHIVE_SIGNATURES = (
    b'PK\x03\x04', b'PK\x05\x06', b'PK\x07\x08',
    b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1',  # OLE: HWP and older Office documents.
    b'\x1f\x8b', b'7z\xbc\xaf\x27\x1c', b'Rar!\x1a\x07', b'BZh', b'\xfd7zXZ\x00',
)


def allowed_source(path):
    p = PurePosixPath(path)
    if not p.parts or p.is_absolute() or '..' in p.parts or '\\' in path:
        return False
    if path in ROOT_FILES or path in PUBLIC_JSON:
        return True
    if p.parent == PurePosixPath('.github/workflows'):
        return p.suffix.lower() in {'.yml', '.yaml'}
    return len(p.parts) > 1 and p.suffix.lower() in SOURCE_SUFFIXES.get(p.parts[0], set())


def inspect(path, blob, mode='100644'):
    errors = []
    p = PurePosixPath(path)
    if any(part.lower() in PRIVATE_DIRS for part in p.parts):
        errors.append('private/generated directory')
    if mode not in ('100644', '100755'):
        errors.append('symlink/submodule is not allowed')
    if not allowed_source(path):
        errors.append('not an allowed source path and extension')
    if p.suffix.lower() in FORBIDDEN or p.name.casefold().startswith('.env'):
        errors.append('forbidden document, data or runtime extension')
    if len(blob) > 250_000:
        errors.append('unexpectedly large source file')
    # PDF readers can accept a header starting anywhere in the first 1024 bytes.
    pdf_start = blob.find(b'%' + b'PDF-')
    if 0 <= pdf_start < 1024 or blob.startswith(ARCHIVE_SIGNATURES):
        errors.append('document/archive signature, including renamed payload')
    if any(byte < 32 and byte not in (9, 10, 13) or byte == 127 for byte in blob):
        errors.append('binary control bytes in source file')
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


def git_output(args, cwd=None):
    return subprocess.check_output(['git', *args], cwd=cwd)


def scan_entries(entries, cwd, *, tree=False, cache=None):
    cache = {} if cache is None else cache
    errors, count = [], 0
    for entry in filter(None, entries):
        info, path = entry.split('\t', 1)
        if tree:
            mode, object_type, oid = info.split()
            stage = '0'
        else:
            mode, oid, stage = info.split()
        key = (path, oid, mode)
        if key not in cache:
            # A gitlink points to a commit rather than a blob; reject its mode directly.
            blob = git_output(['cat-file', 'blob', oid], cwd) if mode in ('100644', '100755') else b''
            cache[key] = inspect(path, blob, mode)
        issues = list(cache[key])
        if stage != '0':
            issues.append('unmerged index entry')
        errors.extend(path + ': ' + error for error in issues)
        count += 1
    return errors, count


def check_index(cwd=None):
    entries = git_output(['ls-files', '--stage', '--full-name', '-z'], cwd).decode('utf-8').split('\0')
    errors, count = scan_entries(entries, cwd)
    if not count:
        errors.append('empty index')
    return errors, count


def check_outgoing(base_ref, cwd=None):
    """Check every outgoing commit, including files deleted by a later commit."""
    base = git_output(['rev-parse', '--verify', '--end-of-options', base_ref + '^{commit}'], cwd).decode('ascii').strip()
    ancestor = subprocess.run(['git', 'merge-base', '--is-ancestor', base, 'HEAD'], cwd=cwd, check=False)
    if ancestor.returncode != 0:
        raise ValueError('base ref must resolve to an ancestor of HEAD; reconcile history before pushing')
    commits = git_output(['rev-list', '--reverse', base + '..HEAD', '--'], cwd).decode('ascii').splitlines()
    errors, cache = [], {}
    for commit in commits:
        entries = git_output(['ls-tree', '-r', '-z', '--full-tree', commit], cwd).decode('utf-8').split('\0')
        issues, count = scan_entries(entries, cwd, tree=True, cache=cache)
        if not count:
            issues.append('empty source tree')
        errors.extend(commit[:12] + ': ' + error for error in issues)
    return errors, len(commits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-ref', help='Also check every commit in BASE..HEAD before pushing; BASE must be an ancestor')
    args = parser.parse_args()
    cwd = git_output(['rev-parse', '--show-toplevel']).decode('utf-8').strip()
    errors, count = check_index(cwd)
    commits = None
    if args.base_ref:
        try:
            history_errors, commits = check_outgoing(args.base_ref, cwd)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        errors.extend(history_errors)
    if errors:
        raise SystemExit('\n'.join(errors))
    detail = f'; {commits} outgoing commit trees checked' if commits is not None else ''
    print(f'PASS: {count} indexed files checked{detail}')


if __name__ == '__main__':
    main()

