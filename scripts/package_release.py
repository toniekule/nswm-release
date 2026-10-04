"""Create a source-only archive and per-file checksum manifest."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


def release_files(root, include_docs=False):
    """Source files for the public archive.

    The release carries code and a single installation/usage guide: README.md
    is the only top-level markdown file taken. `docs/` is excluded by default
    and is only packaged when include_docs is set.
    """
    files = [root / name for name in ("README.md", "pyproject.toml", ".gitignore")]
    directories = ["src", "tests", "scripts", "examples", "configs", ".github"]
    if include_docs:
        directories.append("docs")
    for directory in directories:
        files.extend(p for p in (root / directory).rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts and not any(part.endswith(".egg-info") for part in p.parts)
                     and p.suffix != ".pyc" and p.name != ".DS_Store"
                     and p.suffix != ".md" and not p.name.endswith(".manifest.json"))
    return sorted(files, key=lambda p: str(p.relative_to(root)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--include-docs", action="store_true",
                        help="also package the local notes in docs/")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("release destination exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    hashes = {}
    with output.open("wb") as stream, gzip.GzipFile(fileobj=stream, mode="wb", mtime=0, filename="") as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as archive:
            for path in release_files(root, include_docs=args.include_docs):
                content = path.read_bytes()
                name = str(path.relative_to(root))
                hashes[name] = hashlib.sha256(content).hexdigest()
                info = tarfile.TarInfo(name)
                info.size, info.mode = len(content), 0o755 if path.suffix == ".sh" else 0o644
                archive.addfile(info, io.BytesIO(content))
    manifest = {"archive_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "files": hashes}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"archive": str(output), "sha256": manifest["archive_sha256"], "files": len(hashes)}))


if __name__ == "__main__":
    main()
