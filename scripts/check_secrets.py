#!/usr/bin/env python3
"""Check staged Git blobs without printing credential values (stdlib only)."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    'api-key-format': re.compile(rb'(?<![A-Za-z0-9])(?:sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}|tvly-[A-Za-z0-9_-]{15,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16})(?![A-Za-z0-9])'),
    'private-key': re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
    'literal-credential': re.compile(rb'''(?im)["']?(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|secret[_-]?key|password)["']?\s*[:=]\s*["']([^\r\n"']{16,})["']'''),
}


def placeholder(value: bytes) -> bool:
    value = value.strip()
    low = value.lower()
    return (
        value.startswith((b'$', b'<', b'{', b'process.', b'os.', b'self.'))
        or value.endswith(b'...')
        or low.startswith((b'your-', b'your_', b'sk-ant-oat01-test-', b'sk-ant-api03-test-', b'sk-test', b'test-', b'fake-', b'dummy-', b'example-', b'placeholder'))
        or low in {b'your_api_key_here', b'your-api-key-here'}
        or bool(re.fullmatch(rb'[A-Z][A-Z0-9]*_(?:[A-Z0-9]+_)*[A-Z0-9]+', value))
    )


def known_local_secrets(root: Path) -> set[bytes]:
    values: set[bytes] = set()
    for relative in ['.env', 'flow/config.yaml', 'flow/extensions_config.json', 'flow/frontend/.env']:
        path = root / relative
        if not path.is_file():
            continue
        data = path.read_bytes()
        for match in PATTERNS['api-key-format'].finditer(data):
            if not placeholder(match.group()):
                values.add(match.group())
        # Also match local nonstandard credential formats, including env values.
        for match in re.finditer(rb'''(?im)^\s*["']?([\w.-]*(?:KEY|TOKEN|SECRET|PASSWORD))["']?\s*[:=]\s*([^\r\n#]+)''', data):
            value = match.group(2).strip().rstrip(b',').strip(b'\"\x27')
            if len(value) >= 12 and not placeholder(value):
                values.add(value)
        if path.suffix == '.json':
            try:
                config = json.loads(data)
            except ValueError:
                continue
            def walk(node):
                if isinstance(node, dict):
                    for key, value in node.items():
                        if isinstance(value, str) and re.search(r'(?:key|token|secret|password)$', key, re.I):
                            encoded = value.encode()
                            if len(encoded) >= 12 and not placeholder(encoded):
                                values.add(encoded)
                        walk(value)
                elif isinstance(node, list):
                    for value in node:
                        walk(value)
            walk(config)
    return values


def forbidden_path(path: str) -> bool:
    p = Path(path)
    if any(part in {'.git', '.local-backups', '.venv', 'node_modules', '.flow', '.langgraph_api'} for part in p.parts):
        return True
    if p.name == '.env' or (p.name.startswith('.env.') and not p.name.endswith('.example')):
        return True
    return p.name in {'config.yaml', 'config.yml', 'extensions_config.json', 'mcp_config.json', 'credentials.json'} or p.suffix in {'.pem', '.key', '.p12', '.pfx'}


def inspect_blob(path: str, data: bytes, known: set[bytes]) -> list[tuple[str, int, str]]:
    findings: set[tuple[str, int, str]] = set()
    if forbidden_path(path):
        findings.add((path, 1, 'private-file-path'))
    if b'\0' in data[:8192]:
        return sorted(findings)
    for name, pattern in PATTERNS.items():
        for match in pattern.finditer(data):
            value = match.group(1) if name == 'literal-credential' else match.group()
            if name != 'private-key' and placeholder(value):
                continue
            findings.add((path, data.count(b'\n', 0, match.start()) + 1, name))
    for value in known:
        start = data.find(value)
        if start >= 0:
            findings.add((path, data.count(b'\n', 0, start) + 1, 'known-local-credential'))
    return sorted(findings)


def check(root: Path) -> tuple[int, list[tuple[str, int, str]]]:
    raw = subprocess.check_output(['git', 'ls-files', '--stage', '-z'], cwd=root)
    entries = []
    findings: list[tuple[str, int, str]] = []
    for row in raw.split(b'\0'):
        if not row:
            continue
        header, name = row.split(b'\t', 1)
        mode, oid, stage = header.split()
        path = name.decode('utf8', 'replace')
        if mode == b'160000':
            findings.append((path, 1, 'embedded-git-repository'))
        elif stage != b'0':
            findings.append((path, 1, 'unmerged-index-entry'))
        else:
            entries.append((oid, path))
    known = known_local_secrets(root)
    with subprocess.Popen(['git', 'cat-file', '--batch'], cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as proc:
        assert proc.stdin is not None and proc.stdout is not None
        for oid, path in entries:
            proc.stdin.write(oid + b'\n')
            proc.stdin.flush()
            header = proc.stdout.readline().split()
            if len(header) != 3 or header[1] != b'blob':
                raise RuntimeError('Cannot read staged blob for ' + path)
            data = proc.stdout.read(int(header[2]))
            proc.stdout.read(1)
            findings.extend(inspect_blob(path, data, known))
        proc.stdin.close()
    return len(entries), sorted(set(findings))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staged', action='store_true', help='Check the exact staged Git content (the default).')
    parser.parse_args()
    count, findings = check(ROOT)
    for path, line, rule in findings:
        print(f'{path}:{line}: {rule}')
    print(f'Staged files checked: {count}; findings: {len(findings)}')
    return 1 if findings else 0


if __name__ == '__main__':
    raise SystemExit(main())
