"""Regression tests for the first real Kaggle failure: the notebook mounted code dataset version 1 (built before
progress.py existed) because the launcher pushed 2 s after uploading version 2, trusting a "ready" status that
still described version 1. Everything here runs before any GPU submission, with a fake Kaggle CLI."""
import ast
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sot_mt import paths
from sot_mt.files import CODE_MANIFEST, code_digest, code_manifest

SAM = paths.SAM_ROOT
REAL_MOUNT = Path("datasets") / "samahmed17"          # Kaggle mounts /kaggle/input/datasets/<owner>/<slug>/


def _script(name):
    sys.path.insert(0, str(SAM / "scripts"))
    spec = importlib.util.spec_from_file_location(name, SAM / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def staged_ok():
    if not (paths.staged_dir() / "STAGED_FILES.json").exists():
        pytest.skip("staged Stage 2 data unavailable")


@pytest.fixture
def deploy(tmp_path, staged_ok):
    """A fresh deployment, rebuilt by the launcher exactly as `launch_kaggle.py` (dry run) would."""
    prep, launch = _script("prepare_kaggle_deploy"), _script("launch_kaggle")
    dep = tmp_path / "deploy"
    prep.prepare("samahmed17", "sam-sot-mt-madlad-smoke5", dep, prep.SMOKE["overrides"], 5400,
                 build_out=tmp_path / "seed")
    d = launch.launch(dep, None, "kaggle", yes=False, build_out=tmp_path / "build", out=lambda *a: None)
    return launch, dep, d, Path(d["code_build_dir"]) / "Sam"


def notebook_imports(nb_path: Path) -> dict:
    """{module: [names]} for every `from sot_mt[.x] import ...` / `import sot_mt.x` in the notebook's code cells."""
    nb = json.loads(nb_path.read_text())
    found = {}
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        tree = ast.parse("\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("!", "%"))))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "sot_mt":
                if node.module == "sot_mt":
                    for a in node.names:
                        found.setdefault(a.name, [])
                else:
                    found.setdefault(node.module.split(".", 1)[1], []).extend(a.name for a in node.names)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.startswith("sot_mt."):
                        found.setdefault(a.name.split(".", 1)[1], [])
    return found


# -- the exact artifact -------------------------------------------------------------------------------

def test_artifact_contains_and_imports_every_module_the_notebook_needs(deploy):
    _, dep, d, build = deploy
    nb = dep / "00_generate_translations.ipynb"
    imports = notebook_imports(nb)
    assert {"progress", "runner", "session", "smoke", "files"} <= set(imports)
    for mod in imports:
        assert (build / "src" / "sot_mt" / f"{mod}.py").is_file(), f"package lacks sot_mt/{mod}.py"
    # the notebook's own pre-import guard lists every module it imports
    src = "".join("".join(c["source"]) for c in json.loads(nb.read_text())["cells"] if c["cell_type"] == "code")
    guard = ast.literal_eval(src.split("REQUIRED_MODULES = ", 1)[1].split("\n", 1)[0])
    assert set(imports) <= set(guard)
    # import them from the PACKAGE (isolated interpreter, package first on the path), names included
    prog = ("import importlib, json, sys; sys.path.insert(0, sys.argv[1]); spec = json.loads(sys.argv[2]); "
            "mods = {m: importlib.import_module('sot_mt.' + m) for m in spec}; "
            "missing = [f'{m}.{n}' for m, ns in spec.items() for n in ns if not hasattr(mods[m], n)]; "
            "assert not missing, missing; "
            "print(json.dumps({m: mods[m].__file__ for m in mods}))")
    out = subprocess.run([sys.executable, "-I", "-c", prog, str(build / "src"), json.dumps(imports)],
                         capture_output=True, text=True, cwd=dep)
    assert out.returncode == 0, out.stderr
    assert all(f.startswith(str(build / "src")) for f in json.loads(out.stdout).values())


def test_fingerprint_matches_the_artifact(deploy):
    _, dep, d, build = deploy
    listed = json.loads((build / CODE_MANIFEST).read_text())
    assert listed == code_manifest(build)                        # manifest describes the real files
    assert code_digest(listed) == d["code_digest"]
    nb = (dep / "00_generate_translations.ipynb").read_text()
    assert f'EXPECTED_CODE_SHA = \\"{d["code_digest"]}\\"' in nb
    src = {p.relative_to(SAM / "src").as_posix(): p.read_bytes() for p in (SAM / "src").rglob("*.py")
           if "__pycache__" not in p.parts}
    assert {k: (build / "src" / k).read_bytes() for k in src} == src      # rebuilt from the current sources


# -- launcher: rebuild, wait for the NEW version, verify the listing, refuse pins ----------------------

class FakeKaggle:
    """Kaggle CLI double. `versions` are successive (status, current_version_number) answers."""

    def __init__(self, versions, listing, launch, build):
        self.versions, self.listing, self.calls = list(versions), listing, []
        self.launch, self.build, self.uploaded = launch, build, None

    def __call__(self, *args):
        self.calls.append(args)
        if args[:2] == ("datasets", "status"):
            st, v = self.versions.pop(0) if len(self.versions) > 1 else self.versions[0]
            return 0, json.dumps({"status": st, "current_version_number": v})
        if args[:2] == ("datasets", "version"):
            p = Path(args[args.index("-p") + 1]) / "Sam"
            self.uploaded = {"digest": code_digest(code_manifest(p)),
                             "has_progress": (p / "src" / "sot_mt" / "progress.py").is_file()}
            return 0, "Dataset version is being created."
        if args[:2] == ("datasets", "files"):
            return 0, json.dumps([{"name": k, "size": s, "creationDate": "x"} for k, s in self.listing().items()])
        if args[:2] == ("kernels", "push"):
            return 0, "Kernel version 2 successfully pushed.  Please check progress at https://x"
        return 1, "unexpected"


def test_launch_rebuilds_uploads_waits_for_new_version_then_pushes(deploy, tmp_path):
    launch, dep, d, build = deploy
    fake = FakeKaggle([("ready", 1), ("ready", 1), ("running", 2), ("ready", 2)],
                      lambda: launch.expected_listing(build), launch, build)
    slept = []
    rec = launch.launch(dep, fake, "kaggle", yes=True, sleep=slept.append, build_out=tmp_path / "build",
                        out=lambda *a: None)
    assert fake.uploaded == {"digest": rec["code_digest"], "has_progress": True}
    order = [c[:2] for c in fake.calls]
    assert order.index(("datasets", "version")) < order.index(("kernels", "push"))
    assert len(slept) == 2                                        # did not trust the stale "ready" of version 1
    assert rec["code_dataset_version"] == 2 and rec["version"] == 2
    assert json.loads((dep / "last_push.json").read_text())["code_dataset_version"] == 2
    assert ("datasets", "files", d["code_dataset"] + "/2", "--format", "json", "--page-size", "200") in fake.calls


def test_launch_refuses_when_the_new_version_never_appears(deploy, tmp_path):
    launch, dep, _, build = deploy
    fake = FakeKaggle([("ready", 1)], lambda: launch.expected_listing(build), launch, build)
    t = [0.0]

    def sleep(s):
        t[0] += s
    with pytest.raises(SystemExit, match="not ready"):
        launch.launch(dep, fake, "kaggle", yes=True, sleep=sleep, build_out=tmp_path / "build", out=lambda *a: None,
                      clock=lambda: t[0])
    assert not any(c[:2] == ("kernels", "push") for c in fake.calls)


def test_launch_refuses_a_remote_listing_that_is_not_the_local_build(deploy, tmp_path):
    launch, dep, _, build = deploy

    def stale():                     # what version 1 looked like: no progress.py, no CODE_MANIFEST.json
        want = launch.expected_listing(build)
        return {k: v for k, v in want.items() if k not in {"src/sot_mt/progress.py", CODE_MANIFEST}}
    fake = FakeKaggle([("ready", 1), ("ready", 2)], stale, launch, build)
    with pytest.raises(SystemExit, match="does not match the local build"):
        launch.launch(dep, fake, "kaggle", yes=True, sleep=lambda s: None, build_out=tmp_path / "build",
                      out=lambda *a: None)
    assert not any(c[:2] == ("kernels", "push") for c in fake.calls)


def test_skip_code_upload_still_verifies_the_remote_version(deploy, tmp_path):
    launch, dep, _, build = deploy
    fake = FakeKaggle([("ready", 7)], lambda: {}, launch, build)
    with pytest.raises(SystemExit, match="does not match"):
        launch.launch(dep, fake, "kaggle", yes=True, skip_code_upload=True, build_out=tmp_path / "build",
                      out=lambda *a: None)


def test_pinned_code_dataset_version_is_refused(deploy):
    launch, dep, _, _ = deploy
    meta_p = dep / "kernel-metadata.json"
    meta = json.loads(meta_p.read_text())
    meta["dataset_sources"][0] = "samahmed17/sam-sot-mt-code/1"
    meta_p.write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="pinned"):
        launch.check(dep)


def test_dry_run_sends_nothing(deploy, tmp_path):
    launch, dep, _, build = deploy
    fake = FakeKaggle([("ready", 1)], lambda: {}, launch, build)
    lines = []
    launch.launch(dep, fake, "kaggle", yes=False, build_out=tmp_path / "build", out=lines.append)
    assert fake.calls == [] and any("DRY RUN" in x for x in lines)


# -- the notebook on Kaggle's real mount layout, with stale and duplicate code copies ------------------

def _kaggle_layout(build_root: Path, inp: Path, name="sam-sot-mt-code") -> Path:
    """Kaggle unpacks Sam.zip without the Sam/ prefix under /kaggle/input/datasets/<owner>/<slug>/."""
    dest = inp / REAL_MOUNT / name
    shutil.copytree(build_root / "Sam", dest)
    return dest


def _run_nb(dep: Path, inp: Path, tmp: Path, extra=""):
    nb = json.loads((dep / "00_generate_translations.ipynb").read_text())
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    script = tmp / "nb.py"
    script.write_text("\n\n".join(["display = print", cells[0],
                                   f"MOCK = True\nOUTPUT_DIR = {str(tmp / 'working' / 'sam_outputs')!r}\n{extra}",
                                   *cells[1:]]))
    return subprocess.run([sys.executable, "-I", str(script)], capture_output=True, text=True, cwd=tmp,
                          env={"SAM_KAGGLE_INPUT": str(inp), "PATH": "/usr/bin:/bin", "HOME": str(tmp)})


def _stage(inp: Path, d: dict):
    shutil.copytree(Path(d["code_build_dir"]).parent / "sam-sot-mt-staged", inp / REAL_MOUNT / "sam-sot-mt-staged")


def test_stale_code_dataset_fails_clearly_before_any_import(deploy, tmp_path):
    _, dep, d, _ = deploy
    inp = tmp_path / "input"
    code = _kaggle_layout(Path(d["code_build_dir"]), inp)
    (code / "src" / "sot_mt" / "progress.py").unlink()            # reproduce code dataset version 1
    (code / CODE_MANIFEST).unlink()
    _stage(inp, d)
    res = _run_nb(dep, inp, tmp_path)
    assert res.returncode != 0
    assert "no attached code matches EXPECTED_CODE_SHA" in res.stderr and "ImportError" not in res.stderr
    assert "older build" in res.stdout                            # the candidate listing says why


def test_stale_build_without_expected_digest_names_the_missing_module(deploy, tmp_path):
    _, dep, d, _ = deploy
    inp = tmp_path / "input"
    code = _kaggle_layout(Path(d["code_build_dir"]), inp)
    (code / "src" / "sot_mt" / "progress.py").unlink()
    (code / CODE_MANIFEST).unlink()
    _stage(inp, d)
    res = _run_nb(dep, inp, tmp_path, extra="EXPECTED_CODE_SHA = None")
    assert res.returncode != 0 and "lacks sot_mt modules ['progress']" in res.stderr


def test_correct_copy_is_chosen_when_a_stale_one_is_also_attached(deploy, tmp_path):
    _, dep, d, _ = deploy
    inp = tmp_path / "input"
    old = inp / "aaa-older-notebook-output" / "sam_code"           # sorts first, like an attached old output
    shutil.copytree(Path(d["code_build_dir"]) / "Sam", old)
    (old / "src" / "sot_mt" / "progress.py").unlink()
    (old / CODE_MANIFEST).unlink()
    good = _kaggle_layout(Path(d["code_build_dir"]), inp)
    _stage(inp, d)
    res = _run_nb(dep, inp, tmp_path)
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
    assert f"code: {good / 'src'}" in res.stdout and f"code digest: {d['code_digest']}" in res.stdout
    assert "this run: 20 records" in res.stdout
