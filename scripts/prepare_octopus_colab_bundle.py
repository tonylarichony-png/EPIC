from pathlib import Path
import hashlib
import json
import os
import sys
import zipfile

EXPECTED_PYTHON = Path(
    r"C:\ML_Progs\Runtimes\Miniforge3\envs\epic-eda\python.exe"
).resolve()
if Path(sys.executable).resolve() != EXPECTED_PYTHON:
    raise RuntimeError(
        f"Use exactly {EXPECTED_PYTHON}, current={sys.executable}"
    )

PROJECT_ROOT = Path(r"D:\Projects\EPIC\EPIC")
ASSEMBLY = "ASM119413v2"

EXPECTED_AGENTS_SHA = "bbc36fcf2270d4ce806e25a5be0d6b7eeab5fe80bc91acd040e71459baac68c3"
EXPECTED_LOADER_SHA = "37c9b0ed4ccecc02b484010c7b5b02fa35fa35f4c976e76e6d0e00096380b468"

def sha256_file(path, block=8 * 1024 * 1024):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            d.update(chunk)
    return d.hexdigest()

assert PROJECT_ROOT.is_dir(), PROJECT_ROOT
assert (PROJECT_ROOT / "README.md").is_file()
assert (PROJECT_ROOT / "src/ml_project").is_dir()

agents_candidates = [
    PROJECT_ROOT / "AGENTS.md",
    PROJECT_ROOT.parent / "AGENTS.md",
]
AGENTS = next((p for p in agents_candidates if p.is_file()), None)
assert AGENTS is not None
assert sha256_file(AGENTS) == EXPECTED_AGENTS_SHA

LOADER = PROJECT_ROOT / "src/ml_project/epic_data.py"
assert sha256_file(LOADER) == EXPECTED_LOADER_SHA

source = PROJECT_ROOT / "data/raw/unpacked" / ASSEMBLY
participants = source / "for_participants"
genome = source / "genome"

# Exact files our current pipeline needs. Do NOT include unfiltered tracks.
required = [
    genome / "genome.fa.gz",
    genome / "genome.fa.fai",
    participants / "whitelist.train.bed.gz",
    participants / "whitelist.test.bed.gz",
    participants / "template.train.bed.gz",
    participants / "template.test.bed.gz",
]

for p in required:
    assert p.is_file(), p

r1 = sorted(
    p for p in participants.glob("csRNA*-r1-train.bed*")
    if "unfiltered" not in p.name.lower()
)
r2 = sorted(
    p for p in participants.glob("csRNA*-r2-train.bed*")
    if "unfiltered" not in p.name.lower()
)
assert len(r1) == 1, r1
assert len(r2) == 1, r2
required += [r1[0], r2[0]]

source_files = []
source_files += [
    PROJECT_ROOT / "README.md",
    LOADER,  # src tree below also includes it; kept only via tree in archive
]
source_files += required

# Full ml_project package.
for p in sorted((PROJECT_ROOT / "src/ml_project").rglob("*")):
    if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
        source_files.append(p)

# Optional project metadata.
for name in ("pyproject.toml", "requirements.txt", "setup.cfg"):
    p = PROJECT_ROOT / name
    if p.is_file():
        source_files.append(p)

# Deduplicate.
seen = set()
files = []
for p in source_files:
    p = p.resolve()
    if p not in seen:
        seen.add(p)
        files.append(p)

OUT = PROJECT_ROOT.parent / "octopus_colab_bundle.zip"
MANIFEST = PROJECT_ROOT.parent / "octopus_colab_bundle_manifest.json"

def archive_name(path):
    if path == AGENTS.resolve():
        return "EPIC/AGENTS.md"
    rel = path.relative_to(PROJECT_ROOT.resolve())
    return str(Path("EPIC") / rel).replace("\\", "/")

total_bytes = sum(p.stat().st_size for p in files) + AGENTS.stat().st_size
print(f"Files: {len(files)+1}")
print(f"Raw bundle bytes: {total_bytes / 1024**3:.2f} GiB")
print("Filtered replicates:", r1[0].name, "+", r2[0].name)
print("Writing:", OUT)

tmp = OUT.with_suffix(".zip.tmp")
with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
    zf.write(AGENTS, "EPIC/AGENTS.md")
    for i, p in enumerate(files, 1):
        zf.write(p, archive_name(p))
        if i % 50 == 0 or i == len(files):
            print(f"  {i}/{len(files)} files")

os.replace(tmp, OUT)

manifest = {
    "assembly": ASSEMBLY,
    "bundle": str(OUT),
    "bundle_sha256": sha256_file(OUT),
    "bundle_bytes": OUT.stat().st_size,
    "agents_sha256": sha256_file(AGENTS),
    "loader_sha256": sha256_file(LOADER),
    "filtered_r1": r1[0].name,
    "filtered_r2": r2[0].name,
    "included_files": [archive_name(p) for p in files] + ["EPIC/AGENTS.md"],
}
MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print("DONE")
print("Bundle:", OUT)
print("SHA256:", manifest["bundle_sha256"])
print("Manifest:", MANIFEST)
