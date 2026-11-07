"""Package the Wikipedia pilot without model weights, other datasets or dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output/package/Wikipedia_Dataset_Pilot.zip"
TOP_FILES = ("README.md", "requirements.txt", "dataset_pipeline.py", "retrieval_baseline.py",
             "offline_check.py", "run_demo.ps1", "package_pilot.py")
DIRECTORIES = ("data", "evaluation", "tests", "reports", "output/presentation")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check_package(path: Path):
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC integrity check failed")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP member names")
        for name in names:
            parsed = PurePosixPath(name)
            if parsed.is_absolute() or ".." in parsed.parts or "\\" in name:
                raise ValueError(f"Unsafe ZIP path: {name}")
        manifest = json.loads(archive.read("bundle_manifest.json"))
        if set(names) != set(manifest["files"]) | {"bundle_manifest.json"}:
            raise ValueError("Package manifest does not describe all ZIP files")
        for name, expected in manifest["files"].items():
            if digest(archive.read(name)) != expected:
                raise ValueError(f"Package checksum mismatch: {name}")
        with tempfile.TemporaryDirectory(prefix="wikipedia_package_") as temporary:
            archive.extractall(temporary)
            completed = subprocess.run([sys.executable, "-X", "utf8", str(Path(temporary) / "offline_check.py")],
                                       cwd=temporary, capture_output=True, text=True, encoding="utf-8", timeout=120)
            if completed.returncode:
                raise ValueError(f"Extracted package offline check failed: {completed.stderr or completed.stdout}")
            offline = json.loads(completed.stdout)
            if offline["status"] != "passed":
                raise ValueError("Extracted package offline verification did not pass")
    return {"status": "passed", "zip_path": str(path), "zip_bytes": path.stat().st_size,
            "zip_sha256": digest(path.read_bytes()), "files_checked": len(manifest["files"]),
            "extracted_package_offline_check": offline,
            "runtime_requirement": "Installed Python 3.10 or newer. No pip packages are required for the included pipeline."}


def create_package(path: Path):
    from dataset_pipeline import verify
    verify(ROOT)
    files = {name: (ROOT / name).read_bytes() for name in TOP_FILES}
    for directory in DIRECTORIES:
        for candidate in sorted((ROOT / directory).rglob("*")):
            if not candidate.is_file() or "__pycache__" in candidate.parts:
                continue
            relative = candidate.relative_to(ROOT).as_posix()
            if relative == "reports/package_verification.json":
                continue
            files[relative] = candidate.read_bytes()
    manifest = {"title": "Wikipedia dataset pilot", "status": "dataset_and_lexical_baseline_only",
                "runtime_requirement": "Python >= 3.10, standard library only",
                "files": {name: digest(content) for name, content in sorted(files.items())}}
    files["bundle_manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, content in sorted(files.items()):
            archive.writestr(name, content)
    result = check_package(path)
    (ROOT / "reports/package_verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", type=Path, help="Verify and run an existing ZIP instead of creating one")
    args = parser.parse_args()
    try:
        result = check_package(args.check) if args.check else create_package(args.output)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, subprocess.TimeoutExpired, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Package failed: {exc}\n")


if __name__ == "__main__":
    main()
