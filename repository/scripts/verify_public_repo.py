import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def verify(root, manifest_name="MANIFEST.json"):
    manifest = json.loads((root / manifest_name).read_text(encoding="utf-8"))
    expected = manifest["files"]
    ignored = {".git", ".venv", "__pycache__", "derived_data_v14", "results"}
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and not any(part in ignored for part in p.relative_to(root).parts)
              and p.name != manifest_name}
    if actual != set(expected):
        raise SystemExit(f"File inventory differs: missing={sorted(set(expected) - actual)}, extra={sorted(actual - set(expected))}")
    for name, checksum in expected.items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or digest(path) != checksum:
            raise SystemExit(f"Checksum mismatch: {name}")
    return len(expected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the release file inventory and SHA-256 hashes.")
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    report = {"code_files": verify(Path(__file__).resolve().parents[1])}
    if args.data_root:
        report["data_files"] = verify(args.data_root)
    print(json.dumps({"status": "PASS", **report}, indent=2))
