"""Extract EPIC .tgz containers, retaining the .gz files consumed by notebooks.

Existing files must match the archive byte for byte; they are never overwritten.
Run: python scripts/unpack_raw_archives.py
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile


RAW = Path(__file__).resolve().parents[1] / "data/raw"
DESTINATION = (RAW / "unpacked").resolve()
BLOCK = 8 * 1024 * 1024


def safe_target(member, assembly):
    parts = PurePosixPath(member.name).parts
    if (
        not parts or parts[0] != assembly or ".." in parts
        or "\\" in member.name or ":" in member.name
        or member.name.startswith("/")
        or not (member.isdir() or member.isfile())
    ):
        raise ValueError(f"Unsupported archive entry: {member.name}")
    target = DESTINATION.joinpath(*parts).resolve()
    if not target.is_relative_to(DESTINATION):
        raise ValueError(f"Archive entry escapes destination: {member.name}")
    return target


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_archive(archive):
    records, seen = [], set()
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            target = safe_target(member, archive.stem)
            if member.isdir():
                if target.exists() and not target.is_dir():
                    raise ValueError(f"Directory path already occupied: {target}")
                continue
            if target in seen:
                raise ValueError(f"Duplicate archive target: {target}")
            seen.add(target)
            if target.exists() and (not target.is_file() or target.stat().st_size != member.size):
                raise ValueError(f"Existing file has different size: {target}")
            records.append({"name": member.name, "bytes": member.size, "existing": target.exists()})
    if not records:
        raise ValueError(f"Archive has no files: {archive}")
    return records


def extract_archive(archive):
    records = []
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            target = safe_target(member, archive.stem)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            digest, size = hashlib.sha256(), 0
            temporary = None
            existed = target.exists()
            try:
                if existed:
                    output = target.open("rb")
                else:
                    output = tempfile.NamedTemporaryFile(
                        mode="wb", dir=target.parent, prefix=".extract-", suffix=".partial", delete=False,
                    )
                    temporary = Path(output.name)
                with output, stream.extractfile(member) as source:
                    for block in iter(lambda: source.read(BLOCK), b""):
                        size += len(block)
                        digest.update(block)
                        if existed:
                            if output.read(len(block)) != block:
                                raise ValueError(f"Existing file differs from archive: {target}")
                        else:
                            output.write(block)
                    if existed and output.read(1):
                        raise ValueError(f"Existing file has trailing data: {target}")
                if size != member.size:
                    raise ValueError(f"Truncated archive member: {member.name}")
                expected_hash = digest.hexdigest()
                if temporary is not None:
                    if sha256_file(temporary) != expected_hash:
                        raise ValueError(f"Extracted file failed checksum: {member.name}")
                    if target.exists():
                        raise FileExistsError(target)
                    temporary.rename(target)
                    temporary = None
                records.append({"name": member.name, "bytes": size, "sha256": expected_hash,
                                "action": "verified_existing" if existed else "extracted"})
            finally:
                # This invocation's uniquely named, unfinished temporary file only.
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    return records


def main():
    if not DESTINATION.is_relative_to(RAW.resolve()):
        raise ValueError("Destination must remain inside the requested raw directory")
    archives = sorted(RAW.glob("*.tgz"))
    if not archives:
        raise FileNotFoundError(f"No .tgz archives in {RAW}")
    plans = {}
    for archive in archives:
        print(f"Inspecting {archive.name}", flush=True)
        plans[archive] = inspect_archive(archive)
        records = plans[archive]
        print(f"  {len(records)} files, {sum(r['bytes'] for r in records):,} bytes", flush=True)
    missing_bytes = sum(r["bytes"] for rows in plans.values() for r in rows if not r["existing"])
    if shutil.disk_usage(RAW).free < missing_bytes + 512 * 1024 * 1024:
        raise OSError("Not enough free disk space")
    print(f"New disk space required: {missing_bytes / 1024**3:.3f} GiB", flush=True)

    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "destination": str(DESTINATION), "inner_gzip_preserved": True, "archives": []}
    for archive in archives:
        print(f"Extracting / verifying {archive.name}", flush=True)
        records = extract_archive(archive)
        if [(r["name"], r["bytes"]) for r in records] != [
            (r["name"], r["bytes"]) for r in plans[archive]
        ]:
            raise ValueError(f"Archive changed after inspection: {archive}")
        new = sum(r["action"] == "extracted" for r in records)
        report["archives"].append({"archive": archive.name, "files": records})
        print(f"  OK: {new} extracted, {len(records) - new} existing verified", flush=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report_path = DESTINATION / f"extraction_report_{timestamp}.json"
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
