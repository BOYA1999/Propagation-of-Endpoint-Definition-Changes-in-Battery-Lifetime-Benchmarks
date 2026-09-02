"""Validate the anonymous code-only release before publication."""

import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_FILES = {
    ".gitignore",
    "DATA_AVAILABILITY.md",
    "DATA_SOURCES.md",
    "FIGURE_CONTRACT.md",
    "INPUTS.md",
    "LICENSE",
    "MANIFEST.sha256",
    "README.md",
    "SECURITY_AND_PRIVACY.md",
    "THIRD_PARTY_NOTICES.md",
    "requirements.txt",
    "scripts/verify_public_repo.py",
    "src/academic.mplstyle",
    "src/analyze_endpoint_contracts.py",
    "src/audit_endpoint_labels.py",
    "src/build_paper_figures.py",
    "src/build_tables.py",
    "src/run_depth_campaign.py",
    "src/run_identifiability_diagnostics.py",
    "src/validate_depth_campaign.py",
    "src/validate_paper_figures.py",
    "src/validate_results.py",
}
FORBIDDEN_PARTS = {"__pycache__", ".pytest_cache", ".git", ".venv", "inputs", "outputs"}
FORBIDDEN_SUFFIXES = {".csv", ".docx", ".eps", ".ipynb", ".joblib", ".pdf", ".pkl", ".png", ".pt", ".pth", ".svg", ".zip"}
TEXT_NAMES = {".gitignore", "LICENSE", "MANIFEST.sha256"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest():
    entries = {}
    for line in (ROOT / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        entries[name] = digest
    return entries


def text_files(files):
    return [path for path in files if path.suffix in {".py", ".md", ".txt", ".mplstyle"} or path.name in TEXT_NAMES]


def privacy_hits(files):
    windows_path = re.compile(r"(?i)(?<![a-z0-9])[a-z]:[\\/]")
    home_path = re.compile(r"/(?:Users|home)/[^/\s]+")
    email = re.compile(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b")
    secret = re.compile(r"(?im)^\s*(?:api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*['\"][^'\"]{8,}")
    markers = tuple("".join(parts) for parts in (
        ("chat", "gpt"),
        ("open", "ai"),
        ("language", " model"),
        ("as an", " ai"),
        ("the user", " requested"),
        ("co", "dex"),
        ("reviewer", "_v"),
        ("gate", "23_v"),
        ("source", "_docx"),
        ("pass", "_with_"),
        ("formal", "_upload_stop"),
    ))
    hits = []
    for path in text_files(files):
        relative = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in (("windows_path", windows_path), ("home_path", home_path), ("email", email), ("secret", secret)):
            if pattern.search(text):
                hits.append(f"{relative}:{label}")
        lowered = text.lower()
        for marker in markers:
            if marker in lowered:
                hits.append(f"{relative}:marker:{marker}")
    return sorted(set(hits))


def command_help(scripts):
    failures = []
    for path in scripts:
        run = subprocess.run([sys.executable, str(path), "--help"], capture_output=True, text=True)
        if run.returncode != 0:
            failures.append(path.relative_to(ROOT).as_posix())
    return failures


def main():
    files = sorted(path for path in ROOT.rglob("*") if path.is_file())
    relative_files = {path.relative_to(ROOT).as_posix() for path in files}
    scripts = sorted((ROOT / "src").glob("*.py"))
    source_docs = "\n".join((ROOT / name).read_text(encoding="utf-8") for name in ("README.md", "DATA_SOURCES.md", "DATA_AVAILABILITY.md", "THIRD_PARTY_NOTICES.md"))
    manifest = read_manifest()
    manifest_expected = {name for name in relative_files if name != "MANIFEST.sha256"}
    checks = {
        "allowlisted_file_set": relative_files == ALLOWED_FILES,
        "no_forbidden_directories": not any(FORBIDDEN_PARTS & set(path.relative_to(ROOT).parts) for path in files),
        "no_forbidden_file_types": not any(path.suffix.lower() in FORBIDDEN_SUFFIXES for path in files),
        "no_symbolic_links": not any(path.is_symlink() for path in ROOT.rglob("*")),
        "python_syntax": all(ast.parse(path.read_text(encoding="utf-8")) is not None for path in scripts),
        "command_line_help": not command_help(scripts),
        "privacy_and_release_scan": not privacy_hits(files),
        "mit_licence_present": (ROOT / "LICENSE").read_text(encoding="utf-8").startswith("MIT License\n") and "Permission is hereby granted" in (ROOT / "LICENSE").read_text(encoding="utf-8"),
        "source_identifiers_present": all(token in source_docs for token in ("21149533", "22582234", "10714284", "10648587", "3b17bb53202e10711deae0448c91f84144bb3dd9")),
        "manifest_file_set": set(manifest) == manifest_expected,
        "manifest_hashes": all((ROOT / name).is_file() and sha256(ROOT / name) == digest for name, digest in manifest.items()),
    }
    details = {
        "unexpected_or_missing_files": sorted(relative_files ^ ALLOWED_FILES),
        "privacy_hits": privacy_hits(files),
        "help_failures": command_help(scripts),
    }
    report = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "details": details, "passed": sum(checks.values()), "total": len(checks)}
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()

