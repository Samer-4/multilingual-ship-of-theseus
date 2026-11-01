"""Write Sam/notebooks/00_generate_translations.ipynb (the notebook is generated; edit this file, not the .ipynb).

Usage (from the repo root):
  Sam/.venv/bin/python Sam/scripts/build_notebook00.py
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "00_generate_translations.ipynb"
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


md(r"""
# 00 · Generate round-trip translations (Kaggle)

**What this experiment does, in plain English.** We take 200 short English texts (T0) from the XSum part of the Ship of Theseus corpus. Each text is translated into Spanish and back into English three times in a row:

`T0 (English) → S1 (Spanish) → T1 (English) → S2 → T2 → S3 → T3`

Each round starts from the previous round's English, never from the original. We do this with four translators, each kept completely separate: **MADLAD-400-3B** and **NLLB-200** (dedicated translation models), and **Qwen3-4B** and **Llama-3.2-3B** (general chat models). Later notebooks measure how much wording, meaning and style survive (RQ1), whether the original source can still be recognised (RQ2), and whether each translator leaves its own fingerprint (RQ3). All of them read the table this notebook produces.

**How this notebook works.**
- You choose **one translator per session** in the configuration cell.
- The shared runner (`Sam/src/sot_mt/runner.py`) finds the next unfinished step for every document. It translates, checks the output, and saves a checkpoint every few documents.
- Bad outputs (wrong language, empty, repetition loops, cut off by the length limit) are **kept, with the reason**, and that document's chain stops there. A failed output is never translated further.
- If the session dies, re-running the notebook continues where it stopped. At the end, the export folder holds everything the next session needs.

**What it never does.** It never overwrites the staged Stage 2 data, never mixes translators in one chain, never repairs generated text, and never downloads a model unless `ALLOW_MODEL_DOWNLOAD = True`.

**Default settings = the Stage 3B smoke test:** MADLAD, the five items in `Sam/data/manifests/stage3b_madlad_smoke5.csv`, continuing T1 → S2 → T2 → S3 → T3 (20 operations). Step-by-step instructions: `Sam/reports/stage3/KAGGLE_SMOKE_TEST.md`.
""")

md("## 1 · Configuration: the only cell you need to edit")
code(r'''
# ---- choose ONE pathway per session -------------------------------------------------------------
TRANSLATOR         = "madlad-3b"   # "madlad-3b" | "nllb" | "qwen3-4b" | "llama"
PROMPT_VARIANT     = None          # None = default ("mt-default" for MT, "free" for LLMs); "literal" = Qwen ablation
ITEM_IDS_FILE      = "data/manifests/stage3b_madlad_smoke5.csv"   # relative to Sam/; None = use MAX_ITEMS
MAX_ITEMS          = None          # first N documents in manifest order (when ITEM_IDS_FILE is None); None = all 200
ROUNDS             = (1, 3)        # first and last round trip to work on
STOP_AFTER         = None          # e.g. "S2" to stop after one stage
CHECKPOINT_EVERY   = 5             # operations between atomic checkpoint writes
EXPORT_EVERY       = None          # operations between rolling exports (RESUME.json) to the export folder; None = at the end only
TIME_LIMIT_MINUTES = 600           # stop cleanly well before Kaggle's session limit
MOCK               = False         # True = fake translator: checks the pipeline with no GPU and no model

# ---- permissions: nothing is downloaded unless you say so -------------------------------------
ALLOW_MODEL_DOWNLOAD = False       # set True for the first real run of a model (needs Internet ON)
USE_HF_TOKEN         = False       # True only for gated models (Llama): reads the Kaggle Secret "HF_TOKEN"

# ---- MADLAD consistency with round 1 ------------------------------------------------------------
PIN_MADLAD_ROUND1_LIBS = False     # True = install transformers 4.46.3 / sentencepiece 0.2.0 (round 1) before import
CHECK_TOKENIZER_PARITY = True      # MADLAD: re-segment all 200 texts and require round 1's segment counts
DRIFT_PROBE            = False     # MADLAD: re-translate T0 -> S1 for the selected items and compare with the
                                   # stored S1 (extra GPU translations, saved to drift_probe.csv, never recorded)
DRIFT_PROBE_MAX_ITEMS  = None      # probe only the first N selected items (None = all); batches use 2
REFERENCE_GATE         = True      # with DRIFT_PROBE: stop before translating if any probe output is degenerate
REFERENCE_MIN_SIMILARITY = 0.8     # ... or shares less than this word similarity with the round-1 S1
                                   # (verified wiring: 1.0; broken wirings: 0.0)
MADLAD_DIAGNOSTIC      = False     # T0 -> S1 under each embedding wiring, compared with round 1
                                   # (madlad_diagnostic.json; no records); stops unless the CONFIGURED wiring
                                   # (translators.yaml runtime_patches) passes

# ---- other translators (NLLB, Qwen, Llama) ------------------------------------------------------
SANITY_PROBE      = True           # load the model and translate two fixed sentences (en->es, es->en) before any
                                   # record; stop unless both are clean target-language translations (no records)
EXPECTED_WEIGHTS_ID = None         # weights id measured by this model's preflight (load_report.weights_id); a
                                   # full run stops if the loaded weights differ

# ---- inputs (placeholders; if left as is, the notebook searches /kaggle/input for them) -------
CODE_DATASET     = "<YOUR-CODE-DATASET-SLUG>"     # contains Sam/src/sot_mt and Sam/configs
STAGED_DATASET   = "<YOUR-STAGED-DATASET-SLUG>"   # contains t0_manifest.parquet, generations.parquet, STAGED_FILES.json
PREVIOUS_EXPORTS = "auto"  # "auto" = every attached export of THIS pathway (found by RESUME.json); or a list of
                           # folders, e.g. ["/kaggle/input/<prev-output>/sam_outputs/export/madlad-3b__mt-default"]; [] = none
EXPECTED_PREVIOUS_MERGED_SHA256 = None   # list of sha256: the attached exports must be exactly these
                                          # (set by prepare_kaggle_deploy.py when it plans a reuse)
QWEN_LEGACY_CSV  = None    # chain_es_Qwen3-4B-Instruct-2507.csv if it is ever recovered (imported, never continued)

OUTPUT_DIR = "/kaggle/working/sam_outputs"
EXPECTED_CODE_SHA = None   # set by scripts/prepare_kaggle_deploy.py: stop at once if the attached code is a different build
''')

md(r"""
## 2 · Dependencies (controlled setup)

This cell runs **before** torch or transformers is imported. It reads installed versions without importing them, installs any missing support package (never torch, and nothing at all when `MOCK = True`), and, only if `PIN_MADLAD_ROUND1_LIBS = True`, pins the MADLAD round-1 versions. **vLLM is never installed.** If transformers was already imported in this kernel, a pin needs a restart (Run → Restart & clear outputs), then Run All again.
""")
code(r'''
import importlib.metadata as md_, os, subprocess, sys

def installed(dist):
    try:
        return md_.version(dist)
    except md_.PackageNotFoundError:
        return None

SUPPORT = ["transformers", "accelerate", "sentencepiece", "protobuf", "pandas", "pyarrow", "pyyaml"]
if not MOCK and installed("torch") is None:
    raise RuntimeError("torch is not installed: use a Kaggle GPU notebook image (it ships torch)")
missing = [] if MOCK else [d for d in SUPPORT if installed(d) is None]   # MOCK needs no model libraries
pins = []
if PIN_MADLAD_ROUND1_LIBS and TRANSLATOR == "madlad-3b" and not MOCK:
    # transformers 4.46.3 requires tokenizers <0.21 and huggingface-hub <1.0; pip resolves both
    pins = ["transformers==4.46.3", "sentencepiece==0.2.0"]
    pins = [p for p in pins if installed(p.split("==")[0]) != p.split("==")[1]]
todo = sorted(set(missing) - {p.split("==")[0] for p in pins}) + pins
if todo:
    if "transformers" in sys.modules:
        raise RuntimeError("transformers is already imported in this kernel: restart it, then Run All")
    print("pip install", *todo)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *todo], check=True)
for d in ["torch", *SUPPORT]:
    print(f"{d:14s}", installed(d) or "MISSING")
if pins:
    assert installed("transformers") == "4.46.3", "pin failed: see the pip output above"
''')

md("## 3 · Environment: GPU and package versions")
code(r'''
import importlib, platform
from pathlib import Path

KAGGLE_INPUT = Path(os.environ.get("SAM_KAGGLE_INPUT", "/kaggle/input"))   # override only for local tests
ON_KAGGLE = KAGGLE_INPUT.is_dir()
print("Kaggle:", ON_KAGGLE, "| Python", platform.python_version())
try:
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv"],
                         capture_output=True, text=True).stdout)
except FileNotFoundError:
    print("nvidia-smi not found")
CUDA = False
try:
    import torch
    CUDA = torch.cuda.is_available()
    print("torch", torch.__version__, "| CUDA:", CUDA, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    if CUDA and not torch.cuda.is_bf16_supported():
        print("note: no bf16 on this GPU (e.g. T4); all configs use float16")
except ImportError:
    print("torch not installed")
if not MOCK and not CUDA:
    raise RuntimeError("No CUDA GPU in this session. Kaggle: Settings > Accelerator > GPU T4 x1 (or P100), "
                       "then restart. To check the pipeline without a GPU set MOCK = True.")
if not MOCK:
    import transformers
    print("transformers", transformers.__version__)
''')

md("## 4 · Locate the staged inputs and load the project code")
code(r'''
import hashlib, json, zipfile

import pandas as pd

# Modules this notebook imports from the code dataset; checked before the first import (stale-build guard).
REQUIRED_MODULES = ["checks", "files", "legacy_qwen", "progress", "runner", "schema", "session", "smoke", "summaries", "translators"]

def code_digest_of(sam_root):
    """Digest of a packaged code tree from its CODE_MANIFEST.json, verifying every listed file (stdlib only:
    must work before any sot_mt import). None if the tree has no manifest (a build older than Stage 3B)."""
    mf = Path(sam_root) / "CODE_MANIFEST.json"
    if not mf.exists():
        return None
    listed = json.loads(mf.read_text())
    bad = [k for k, v in listed.items()
           if not (Path(sam_root) / k).is_file() or hashlib.sha256((Path(sam_root) / k).read_bytes()).hexdigest() != v]
    if bad:
        raise RuntimeError(f"code dataset files differ from its CODE_MANIFEST.json: {bad[:10]}")
    return hashlib.sha256(json.dumps(listed, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]

def pick_code_root(candidates):
    """Among every attached copy of sot_mt (code dataset, an older output's sam_code/ ...), take the one whose
    digest is EXPECTED_CODE_SHA; without an expected digest, prefer the code dataset's own mount."""
    found = [(c, code_digest_of(c.parent)) for c in candidates]
    for c, d in found:
        print(f"  code candidate {c}  digest {d or 'none (no CODE_MANIFEST.json: older build)'}")
    if EXPECTED_CODE_SHA:
        match = [c for c, d in found if d == EXPECTED_CODE_SHA]
        if not match:
            raise RuntimeError(f"no attached code matches EXPECTED_CODE_SHA {EXPECTED_CODE_SHA}: the code dataset "
                               "mounted here is a different (stale?) version. Re-run scripts/launch_kaggle.py, which "
                               "waits for the new dataset version before pushing.")
        return match[0]
    own = [c for c, _ in found if CODE_DATASET.strip("<>") in c.parts or "sam-sot-mt-code" in c.parts]
    return (own or [c for c, _ in found] or [None])[0]

def find_dir(slug, marker, up):
    """Folder that contains `marker` (going `up` levels), under <input>/<slug> or anywhere in the input root."""
    roots = ([KAGGLE_INPUT / slug] if slug and not slug.startswith("<") else []) + [KAGGLE_INPUT]
    for root in roots:
        if root.exists():
            hits = sorted(root.rglob(marker))
            if hits:
                p = hits[0]
                for _ in range(up):
                    p = p.parent
                return p
    return None

def unpack_code_zip():
    """If Kaggle kept the uploaded Sam.zip as a zip, extract it into /kaggle/working (read-only input)."""
    for z in sorted(KAGGLE_INPUT.rglob("*.zip")):
        with zipfile.ZipFile(z) as f:
            if any(n.endswith("sot_mt/__init__.py") for n in f.namelist()):
                dest = Path(OUTPUT_DIR).parent / "sam_code"
                f.extractall(dest)
                print("extracted", z, "->", dest)
                return next(dest.rglob("sot_mt/__init__.py")).parents[1]
    return None

if ON_KAGGLE:
    cands = sorted(p.parents[1] for p in KAGGLE_INPUT.rglob("sot_mt/__init__.py"))
    SAM_SRC = pick_code_root(cands) if cands else unpack_code_zip()
    STAGED = find_dir(STAGED_DATASET, "STAGED_FILES.json", 1)
    os.environ["SAM_OUTPUT_DIR"] = OUTPUT_DIR
    if not (SAM_SRC and STAGED):
        print("contents of", KAGGLE_INPUT, ":", *[str(p) for p in sorted(KAGGLE_INPUT.glob("*/*"))[:40]], sep="\n  ")
else:   # local: this notebook lives in Sam/notebooks
    SAM_SRC = Path.cwd().resolve().parent / "src"
    STAGED = SAM_SRC.parent / "data" / "staged"
assert SAM_SRC and (SAM_SRC / "sot_mt").is_dir(), "project code not found: attach the code dataset (sam-sot-mt-code)"
assert STAGED and (STAGED / "STAGED_FILES.json").exists(), "staged inputs not found: attach sam-sot-mt-staged"
os.environ["SAM_STAGED_DIR"] = str(STAGED)
SAM_ROOT = SAM_SRC.parent
CODE_DIGEST = code_digest_of(SAM_ROOT)
if EXPECTED_CODE_SHA and CODE_DIGEST != EXPECTED_CODE_SHA:
    raise RuntimeError(f"code dataset digest {CODE_DIGEST} != EXPECTED_CODE_SHA {EXPECTED_CODE_SHA}")
missing_mods = [m for m in REQUIRED_MODULES if not (SAM_SRC / "sot_mt" / f"{m}.py").is_file()]
if missing_mods:
    raise RuntimeError(f"code dataset at {SAM_SRC} lacks sot_mt modules {missing_mods}: it is an older build. "
                       "Re-run scripts/launch_kaggle.py (it uploads the current build and waits for it).")
sys.path.insert(0, str(SAM_SRC))
for name in [n for n in sys.modules if n == "sot_mt" or n.startswith("sot_mt.")]:
    del sys.modules[name]            # a re-run must not keep modules imported from another copy

from sot_mt import legacy_qwen, progress, runner, schema, session, smoke, summaries
from sot_mt.files import load_yaml
print("code:", SAM_SRC, "\nstaged:", STAGED)
print("code digest:", CODE_DIGEST)
PROGRESS = progress.Progress.stdout(heartbeat_s=30)   # "SOTMT_PROGRESS {json}" lines, read by scripts/monitor_kaggle.py
PROGRESS.event("notebook_start", translator=TRANSLATOR, mock=MOCK)

PV = PROMPT_VARIANT or ("free" if TRANSLATOR in {"qwen3-4b", "llama", "phi4-mini"} else "mt-default")
if PREVIOUS_EXPORTS == "auto":
    PREVIOUS_EXPORTS = sorted(str(p.parent) for p in KAGGLE_INPUT.rglob("RESUME.json")
                              if json.loads(p.read_text()).get("pathway") == [TRANSLATOR, PV]) if ON_KAGGLE else []
print("previous exports:", PREVIOUS_EXPORTS or "none")
prev_tables = [Path(d) / runner.MERGED for d in PREVIOUS_EXPORTS]
prev_failures = [Path(d) / runner.FAILURES for d in PREVIOUS_EXPORTS]
for p in prev_tables:
    assert p.exists(), f"missing {p}"
if EXPECTED_PREVIOUS_MERGED_SHA256 is not None:
    found = sorted(hashlib.sha256(p.read_bytes()).hexdigest() for p in prev_tables)
    if found != sorted(EXPECTED_PREVIOUS_MERGED_SHA256):
        raise RuntimeError(f"attached previous exports {found} are not the planned ones "
                           f"{sorted(EXPECTED_PREVIOUS_MERGED_SHA256)}: a different or re-run kernel output is attached")
    print("previous exports match the planned sha256:", [x[:16] for x in found])
manifest, base = session.load_inputs(STAGED, prev_tables)       # verifies STAGED_FILES.json hashes
print(f"manifest: {len(manifest)} texts, {manifest.article_id.nunique()} articles | base table: {len(base)} records")

ITEM_IDS = smoke.read_item_ids(SAM_ROOT / ITEM_IDS_FILE) if ITEM_IDS_FILE else None
if ITEM_IDS:
    unknown = set(ITEM_IDS) - set(manifest.item_id)
    assert not unknown, f"item ids not in the manifest: {unknown}"
    print(f"{len(ITEM_IDS)} selected items from {ITEM_IDS_FILE}")
    display(pd.read_csv(SAM_ROOT / ITEM_IDS_FILE))
''')

code(r'''
# Optional: import the legacy Qwen round 1 into its own import-only pathway (never continued by this runner)
if QWEN_LEGACY_CSV:
    cfg, tcfg = load_yaml("experiment.yaml"), load_yaml("translators.yaml")
    try:
        rows, coverage = legacy_qwen.import_qwen_chain(QWEN_LEGACY_CSV, manifest, cfg, tcfg, "free")
        base, stats = schema.upsert(base, rows, manifest, allow_invalid_predecessor=True)
        print("imported legacy Qwen:", stats, "| missing items:", len(coverage["missing"]))
    except legacy_qwen.ArtifactNotFoundError as e:
        print(e)
else:
    print("No legacy Qwen artifacts supplied: qwen3-4b will run from T0.")
''')

md("## 5 · Available translators and current chain completeness")
code(r'''
display(session.translators_overview())
display(session.completeness(base, manifest))
''')

md("## 6 · Build the runner and preview the plan (nothing is loaded yet)")
code(r'''
if USE_HF_TOKEN and ON_KAGGLE:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

WORK_DIR = Path(OUTPUT_DIR if ON_KAGGLE else SAM_ROOT) / ("runs_mock" if MOCK else "runs") / f"{TRANSLATOR}__{PV}"
EXPORT_DIR = Path(OUTPUT_DIR if ON_KAGGLE else SAM_ROOT / ("runs_mock" if MOCK else "runs")) / "export" / f"{TRANSLATOR}__{PV}"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
run = session.open_runner(
    TRANSLATOR, PV, manifest, base, work_dir=WORK_DIR, prior_failure_paths=[p for p in prev_failures if p.exists()],
    mock=MOCK, allow_download=ALLOW_MODEL_DOWNLOAD, use_hf_token=USE_HF_TOKEN,
    rounds=ROUNDS, stop_after=STOP_AFTER, item_ids=ITEM_IDS, max_items=MAX_ITEMS,
    checkpoint_every=CHECKPOINT_EVERY, time_limit_minutes=TIME_LIMIT_MINUTES, progress=PROGRESS,
    export_every=EXPORT_EVERY, export_dir=str(EXPORT_DIR))
print(f"pathway {TRANSLATOR}/{PV} | model {run.tr.model_checkpoint} | config {run.tr.generation_config_id}")
print(f"work dir {run.work} | run_id {run.run_id} | items {len(run.items)}")
if TRANSLATOR == "madlad-3b":
    st_ = run.tr.settings
    print("MADLAD settings (frozen, as round 1):", st_["decoding"], "| budget", st_["chunking"]["token_budget"],
          "| batch", st_["batch_size"], "| dtype", st_["dtype"])
display(run.plan())
''')

md(r"""
## 7 · MADLAD checks before translating

**Tokenizer parity** loads only the tokenizer (no GPU memory, no model weights). It re-cuts every round-1 text into segments and compares the counts with those round 1 recorded. A mismatch stops the notebook, because rounds 2–3 would then cut text differently from round 1. Equal counts are necessary, not sufficient.

**Drift probe** (optional, `DRIFT_PROBE = True`) loads the model and re-translates T0 → S1 for the selected items, then compares the result with the stored S1. It costs a few extra GPU translations, which are saved to `drift_probe.csv` and never enter the generation table. An exact match means this session reproduces round 1. A mismatch means library or GPU drift; rounds 2–3 are still valid as a continuation, but the difference must be reported.
""")
code(r'''
if TRANSLATOR == "madlad-3b" and not MOCK and CHECK_TOKENIZER_PARITY:
    PROGRESS.set_phase("tokenizer_parity")
    tok = run.tr.load_tokenizer()
    print("tokenizer:", type(tok).__name__)
    other = {"es": "en", "en": "es"}
    parity = smoke.madlad_segment_parity(run.table, lambda s, tgt: run.tr.count_tokens(s, other[tgt], tgt),
                                         run.tr.budget, run.exp)
    parity.to_csv(EXPORT_DIR / "tokenizer_parity.csv", index=False)
    print(f"segment counts equal to round 1: {int(parity.match.sum())}/{len(parity)}")
    PROGRESS.event("parity_done", matched=int(parity.match.sum()), checked=len(parity), tokenizer=type(tok).__name__)
    if not parity.match.all():
        display(parity[~parity.match])
        raise RuntimeError("MADLAD tokenizer differs from round 1 (segment counts). Set PIN_MADLAD_ROUND1_LIBS = True, "
                           "restart and Run All; or set CHECK_TOKENIZER_PARITY = False to continue knowingly.")

if TRANSLATOR == "madlad-3b" and not MOCK and MADLAD_DIAGNOSTIC:
    import time
    from sot_mt.translators import t5_embedding_report
    import transformers
    PROGRESS.set_phase("loading_model", note="diagnostic; includes the download on first use")
    t_ = time.monotonic()
    run.tr.load()                                    # applies the configured wiring and records load_report
    PROGRESS.event("probe_model_loaded", probe_load_s=round(time.monotonic() - t_, 1), load_report=run.tr.load_report)
    m, tok = run.tr.model, run.tr.tokenizer
    configured = run.tr.wiring
    diag = {"transformers": transformers.__version__, "torch": __import__("torch").__version__,
            "configured_wiring": configured, "runtime_patch": run.tr.runtime_patch, "load_report": run.tr.load_report,
            "prefix_tokens": {l: tok.convert_ids_to_tokens(tok(f"<2{l}> Hello world.").input_ids) for l in ("es", "en")}}
    pw = run._pathway()
    item = run.items.item_id.iloc[0]
    s1 = pw[(pw.item_id == item) & (pw.output_stage == "S1")].iloc[0]
    diag["item_id"], diag["round1_S1"], diag["variants"] = item, s1.output_text, {}
    for variant in ["none", "shared", "decoder"]:          # "none" = as transformers built it
        run.tr.rewire(variant)                             # restores the loaded modules first, then verifies
        wiring = t5_embedding_report(m)
        t_ = time.monotonic()
        res = run.tr.translate(s1.input_text, "en", "es")
        v = smoke.probe_verdict(s1.input_text, s1.output_text, res.text, res.finish, run.cfg["validation"],
                                REFERENCE_MIN_SIMILARITY)
        diag["variants"][variant] = {**v, "finish": res.finish, "seconds": round(time.monotonic() - t_, 2),
                                     "segment_output_tokens": res.meta.get("segment_output_tokens"),
                                     "segment_hit_eos": res.meta.get("segment_hit_eos"), "wiring": wiring,
                                     "text": res.text}
        PROGRESS.event("probe_done", probe="diagnostic", item_id=item, variant=variant, identical=v["identical"],
                       similarity=v["similarity"], ok=v["ok"], op_seconds=diag["variants"][variant]["seconds"])
        print(f"{variant:8s} ok={v['ok']} identical={v['identical']} similarity={v['similarity']} "
              f"blocking={v['blocking']} | {str(res.text)[:200]}")
    run.tr.rewire(configured)                              # back to the configured, verified wiring
    diag["restored_wiring"] = {"wiring": run.tr.wiring, "report": t5_embedding_report(m)}
    (EXPORT_DIR / "madlad_diagnostic.json").write_text(json.dumps(diag, indent=1, default=str))
    passing = [k for k, v in diag["variants"].items() if v["ok"]]
    print("wirings that reproduce round 1:", passing, "| configured:", configured)
    if configured not in passing:
        raise RuntimeError(f"MADLAD diagnostic: the configured wiring {configured!r} does not reproduce round 1 "
                           f"(passing: {passing}); see madlad_diagnostic.json. Stopping before any S2 work.")

if TRANSLATOR == "madlad-3b" and not MOCK and DRIFT_PROBE:
    import time
    PROGRESS.set_phase("loading_model", note="drift probe; includes the download on first use")
    t_ = time.monotonic()
    run.tr.load()
    PROGRESS.event("probe_model_loaded", probe_load_s=round(time.monotonic() - t_, 1),
                   load_report=run.tr.load_report)
    PROGRESS.set_phase("drift_probe", emit=False)
    pw = run._pathway()
    rows = []
    for item in list(run.items.item_id)[:DRIFT_PROBE_MAX_ITEMS]:
        s1 = pw[(pw.item_id == item) & (pw.output_stage == "S1")].iloc[0]
        t_ = time.monotonic()
        try:
            res = run.tr.translate(s1.input_text, "en", "es")
            text, finish, err = res.text, res.finish, res.error
        except Exception as e:   # recorded, not fatal: the probe is diagnostic only
            text, finish, err = None, "error", f"{type(e).__name__}: {str(e)[:200]}"
        v = smoke.probe_verdict(s1.input_text, s1.output_text, text, finish, run.cfg["validation"],
                                REFERENCE_MIN_SIMILARITY)
        rows.append({"item_id": item, "identical_to_round1_S1": v["identical"], "similarity": v["similarity"],
                     "ok": v["ok"], "blocking": v["blocking"], "finish": finish,
                     "error": err, "seconds": round(time.monotonic() - t_, 2),
                     "round1_words": len(s1.output_text.split()), "probe_words": len((text or "").split()),
                     "probe_text": text})
        PROGRESS.event("probe_done", probe="drift", item_id=item, identical=v["identical"], similarity=v["similarity"],
                       ok=v["ok"], op_seconds=rows[-1]["seconds"], error=err)
        if REFERENCE_GATE and not v["ok"]:
            break                                     # one broken probe is enough evidence; save the GPU time
    probe = pd.DataFrame(rows)
    probe.to_csv(EXPORT_DIR / "drift_probe.csv", index=False)
    display(probe.drop(columns="probe_text"))
    if REFERENCE_GATE and not probe.ok.all():
        PROGRESS.event("quality_gate", source="drift_probe", failed=int((~probe.ok).sum()))
        raise RuntimeError("REFERENCE GATE: a T0 -> S1 re-translation is degenerate or unlike round 1 "
                           f"({probe[~probe.ok][['item_id', 'blocking', 'similarity']].to_dict('records')}); "
                           "the engine does not reproduce round 1. No S2-T3 operation was run.")
''')

md(r"""
## 7b · Sanity gate for the other translators (NLLB, Qwen, Llama)

These models have no round 1 to compare with. Before any record is written, the notebook loads the model (load time and GPU memory are measured) and translates one fixed sentence in each direction. It stops unless each translation ended normally, passes the blocking checks, is in the target language and contains the expected content words. The results go to `sanity_probe.csv` and never enter the generation table.
""")
code(r'''
SANITY = [("en", "es", "The government said on Tuesday that the new bridge over the river will open next year.",
           ["gobierno", "puente", "río"]),
          ("es", "en", "El gobierno dijo el martes que el nuevo puente sobre el río se abrirá el próximo año.",
           ["government", "bridge", "river"])]
if TRANSLATOR != "madlad-3b" and not MOCK and SANITY_PROBE:
    import time
    from sot_mt.checks import check_step
    PROGRESS.set_phase("loading_model", note="sanity gate; includes the download on first use")
    t_ = time.monotonic()
    run.tr.load()
    PROGRESS.event("probe_model_loaded", probe_load_s=round(time.monotonic() - t_, 1), load_report=run.tr.load_report)
    print("load report:", json.dumps(run.tr.load_report, indent=1, default=str)[:3000])
    if EXPECTED_WEIGHTS_ID and run.tr.load_report.get("weights_id") != EXPECTED_WEIGHTS_ID:
        raise RuntimeError(f"loaded weights {run.tr.load_report.get('weights_id')} != EXPECTED_WEIGHTS_ID "
                           f"{EXPECTED_WEIGHTS_ID} (the preflight's): not the same checkpoint")
    rows = []
    for src_, tgt_, text_, words_ in SANITY:
        t_ = time.monotonic()
        res = run.tr.translate(text_, src_, tgt_)
        chk = check_step(text_, res.text, tgt_, run.cfg["validation"]) if res.text else {"blocking": ["missing"], "warnings": []}
        hits = [w for w in words_ if w in (res.text or "").lower()]
        ok = res.finish == "stop" and not chk["blocking"] and len(hits) >= 2
        rows.append({"direction": f"{src_}->{tgt_}", "ok": ok, "finish": res.finish, "blocking": ";".join(chk["blocking"]),
                     "expected_words_found": "|".join(hits), "seconds": round(time.monotonic() - t_, 2), "output": res.text})
        PROGRESS.event("probe_done", probe="sanity", item_id=f"sanity {src_}->{tgt_}", direction=f"{src_}->{tgt_}",
                       ok=ok, blocking=rows[-1]["blocking"], expected_words_found=rows[-1]["expected_words_found"],
                       op_seconds=rows[-1]["seconds"])
    sanity = pd.DataFrame(rows)
    sanity.to_csv(EXPORT_DIR / "sanity_probe.csv", index=False)
    display(sanity)
    if not sanity.ok.all():
        PROGRESS.event("quality_gate", source="sanity_probe", failed=int((~sanity.ok).sum()))
        raise RuntimeError("SANITY GATE: a fixed test sentence was not translated cleanly "
                           f"({sanity[~sanity.ok].drop(columns='output').to_dict('records')}); no records written.")
''')

md("## 8 · Translate\nLoads the model (from cache, or downloads it if allowed), runs every ready operation and checkpoints as it goes. Interrupting the kernel is safe: completed operations are saved, and the next run continues.")
code(r'''
try:
    summary = run.run()
except BaseException:
    # keep every completed operation resumable: write the export (RESUME.json) before the error propagates
    run.export(EXPORT_DIR)
    print("partial export written to", EXPORT_DIR, "- attach this output to resume")
    raise
summary
''')

md("## 9 · Progress, validation and the generated text")
code(r'''
pw = run.table[(run.table.translator == TRANSLATOR) & (run.table.prompt_variant == PV)]
display(schema.attrition(pw, manifest))
this_run = pw[pw.run_id == run.run_id]
print(f"this run: {len(this_run)} records, {int((this_run.validation_status == 'invalid').sum())} invalid")
display(runner.validation_report(this_run))
display(summaries.item_flags(this_run))
display(run.failures.tail(20))
SHOW_CHARS = 400
for item in this_run.item_id.unique()[:5]:
    chain = pw[pw.item_id == item].set_index("output_stage")
    print("=" * 100, "\n", item)
    print("T0:", manifest.set_index("item_id").t0_text[item][:SHOW_CHARS])
    for st in schema.GENERATED_STAGES:
        if st in chain.index:
            print(f"{st} [{chain.loc[st, 'validation_status']}]:", str(chain.loc[st, "output_text"])[:SHOW_CHARS])
''')

md(r"""
## 10 · Export for the next session

The export folder holds:
- `generations_merged.parquet`: Stage 2 plus every record so far. This is the next session's input.
- `generations_delta.parquet`: only this work dir's records.
- `failures.parquet`: failed attempts, including those from earlier sessions.
- `runs.parquet`: settings, library versions and status for each run.
- `validation_report.csv`, `attrition.csv`, `new_translations.csv`.
- `tokenizer_parity.csv` / `drift_probe.csv`, when those checks ran.
- `RESUME.json`: file hashes and attrition.

**Files in `/kaggle/working` do not survive a new session by themselves.** To resume:
1. **Save Version** ("Save & Run All" or "Quick Save"), so `/kaggle/working` is kept as that version's output.
2. In the new session, **Add Input** and choose that notebook's output.
3. Leave `PREVIOUS_EXPORTS = "auto"` (or list the export folder) and run again.

The last cell also writes `<export folder>.zip`, which you can download from the Output panel in one click.
""")
code(r'''
import shutil
try:
    import torch
    if torch.cuda.is_available():         # peak memory of this session, per GPU (kept after the model is unloaded)
        gpu = [{"device": i, "name": torch.cuda.get_device_name(i),
                "max_allocated_gb": round(torch.cuda.max_memory_allocated(i) / 2**30, 2),
                "max_reserved_gb": round(torch.cuda.max_memory_reserved(i) / 2**30, 2)} for i in range(torch.cuda.device_count())]
        (EXPORT_DIR / "gpu_memory.json").write_text(json.dumps(gpu, indent=1))
        PROGRESS.event("gpu_memory", gpu=gpu)
except ImportError:
    pass
resume = run.export(EXPORT_DIR)
print("exported to", EXPORT_DIR)
for name, sha in resume["files"].items():
    print(f"  {name:30s} {(EXPORT_DIR / name).stat().st_size:>10,d} B  {sha[:16]}")
print("download:", shutil.make_archive(str(EXPORT_DIR), "zip", EXPORT_DIR))
PROGRESS.event("notebook_done", status=summary["status"], export_dir=str(EXPORT_DIR))
''')

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
if __name__ == "__main__":
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print("wrote", OUT, f"({len(cells)} cells)")
