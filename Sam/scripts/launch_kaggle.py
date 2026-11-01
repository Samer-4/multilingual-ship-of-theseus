"""Launch a prepared deployment on Kaggle (remote actions only with --yes; otherwise a dry run).

Steps:
  1. REBUILD from the current Sam/ sources: re-run prepare_kaggle_deploy.prepare() with the parameters in
     <deploy>/deploy.json, so the code build, its CODE_MANIFEST.json digest and the notebook's
     EXPECTED_CODE_SHA always describe today's code; then re-validate the metadata. A dataset source pinned
     to a version (owner/slug/N) is refused: it would mount that old version whatever is uploaded.
  2. note the code dataset's current version number, upload the build as a new version
     (`kaggle datasets version -r zip -t`), and wait until `kaggle datasets status --format json` reports
     status "ready" AND a current_version_number above the one noted. (Status alone is not enough: it
     stays "ready" from the previous version while the new one is still processing. That is how the first
     smoke run mounted the old code.)
  3. check that the file listing of that version, and of the unversioned latest, equals the local build
     (names relative to Sam/, sizes); Kaggle unpacks Sam.zip without the Sam/ prefix;
  4. `kaggle kernels push -p <deploy> --accelerator NvidiaTeslaT4 -t <timeout>` (save + run);
  5. record kernel version and code dataset version in <deploy>/last_push.json.
With --skip-code-upload, steps 2 is skipped but step 3 must pass against the current remote version.
The notebook itself re-checks the digest before importing anything (a last line of defence).

Usage (from the repo root):
  Sam/.venv/bin/python Sam/scripts/launch_kaggle.py           # dry run: rebuild + checks, prints the commands
  Sam/.venv/bin/python Sam/scripts/launch_kaggle.py --yes     # rebuild, upload, wait, verify, push and run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sot_mt import kaggle_cli, paths  # noqa: E402
from sot_mt.files import CODE_MANIFEST, code_digest, code_manifest, sha256_file  # noqa: E402
import prepare_kaggle_deploy as prep  # noqa: E402

DEFAULT_DEPLOY = paths.SAM_ROOT / "kaggle" / "madlad_smoke5"


class LaunchError(SystemExit):
    pass


def rebuild(deploy_dir: Path, build_out=None) -> dict:
    """Re-run prepare() from deploy.json so the upload is always the current source tree."""
    old = json.loads((deploy_dir / "deploy.json").read_text())
    user, slug = old["kernel"].split("/")
    kw = {"build_out": Path(build_out)} if build_out else {}
    return prep.prepare(user, slug, deploy_dir, old["overrides"], old["timeout_s"], old.get("kernel_sources", []),
                        results_dir=old.get("results_dir"), generated_root=old.get("generated_root", "data/generated"),
                        model_sources=old.get("model_sources", []), **kw)


def check(deploy_dir: Path) -> dict:
    d = json.loads((deploy_dir / "deploy.json").read_text())
    meta = json.loads((deploy_dir / "kernel-metadata.json").read_text())
    errs = prep.validate(deploy_dir, meta)
    for src in meta.get("dataset_sources", []):
        if src.split("/")[:2] == d["code_dataset"].split("/") and len(src.split("/")) == 3:
            errs.append(f"dataset source {src!r} is pinned to a version: it would mount that version, not the upload")
    if sha256_file(deploy_dir / prep.CODE_FILE) != d["notebook_sha256"]:
        errs.append("notebook changed since prepare_kaggle_deploy.py: re-run it")
    build = Path(d["code_build_dir"]) / "Sam"
    digest = code_digest(code_manifest(build))
    if digest != d["code_digest"] or json.loads((build / CODE_MANIFEST).read_text()) != code_manifest(build):
        errs.append(f"code build is {digest}, notebook expects {d['code_digest']}: re-run prepare_kaggle_deploy.py")
    if errs:
        raise LaunchError(f"not launching: {errs}")
    return d


def expected_listing(build_sam: Path) -> dict:
    """{name: size} as Kaggle lists the unpacked Sam.zip: paths relative to Sam/."""
    return {p.relative_to(build_sam).as_posix(): p.stat().st_size for p in sorted(build_sam.rglob("*")) if p.is_file()}


def remote_listing(cli_run, dataset: str) -> dict:
    rc, out = cli_run("datasets", "files", dataset, "--format", "json", "--page-size", "200")
    if rc != 0:
        raise LaunchError(f"cannot list {dataset}: {out.strip()[:300]}")
    try:
        return {f["name"]: int(f["size"]) for f in json.loads(out[out.index("["):])}
    except (ValueError, KeyError) as e:
        raise LaunchError(f"unexpected listing for {dataset}: {e}")


def dataset_state(cli_run, dataset: str) -> tuple:
    rc, out = cli_run("datasets", "status", dataset, "--format", "json")
    try:
        js = json.loads(out[out.index("{"):])
        return js.get("status"), int(js.get("current_version_number") or 0)
    except ValueError:
        raise LaunchError(f"cannot read status of {dataset} (rc {rc}): {out.strip()[:300]}")


def wait_for_version(cli_run, dataset: str, after: int, timeout_s=900, poll_s=15, sleep=time.sleep,
                     clock=None) -> int:
    clock = clock or time.monotonic
    t0 = clock()
    while True:
        status, version = dataset_state(cli_run, dataset)
        print(f"  {dataset}: status {status}, current version {version} (waiting for > {after})", flush=True)
        if status == "ready" and version > after:
            return version
        if status and "error" in status:
            raise LaunchError(f"{dataset} processing failed ({status}); not pushing")
        if clock() - t0 > timeout_s:
            raise LaunchError(f"{dataset} version {after + 1} not ready after {timeout_s}s; not pushing")
        sleep(poll_s)


def verify_remote(cli_run, dataset: str, version: int, build_sam: Path) -> None:
    want = expected_listing(build_sam)
    for ref in (f"{dataset}/{version}", dataset):          # that version, and what an unpinned source mounts
        got = remote_listing(cli_run, ref)
        if got != want:
            diff = sorted(k for k in set(got) | set(want) if got.get(k) != want.get(k))
            raise LaunchError(f"{ref} does not match the local build (differs: {diff[:12]}); not pushing")


def launch(deploy_dir: Path, cli_run, cli: str, yes: bool, skip_code_upload: bool = False, sleep=time.sleep,
           build_out=None, out=print, clock=None, new_version: bool = False) -> dict:
    pushed = Path(deploy_dir) / "last_push.json"
    if pushed.exists() and not new_version:
        prev = json.loads(pushed.read_text())
        msg = (f"{prev['kernel']} was already pushed (version {prev.get('version')}, {prev.get('pushed_at')}); pushing "
               "again starts a NEW GPU run. Monitor or download the existing one, or pass --new-version deliberately.")
        if yes:
            raise LaunchError(msg)
        out("WARNING: " + msg)
    rebuild(deploy_dir, build_out)
    d = check(deploy_dir)
    build_sam = Path(d["code_build_dir"]) / "Sam"
    upload = ["datasets", "version", "-p", d["code_build_dir"], "-r", "zip", "-t",
              "-m", f"code {d['code_digest']} for {d['kernel']}"]
    push = ["kernels", "push", "-p", str(deploy_dir), "--accelerator", d["accelerator"], "-t", str(d["timeout_s"])]
    out(f"rebuilt from current sources: code digest {d['code_digest']} | kernel {d['kernel']} | "
        f"accelerator {d['accelerator']} | timeout {d['timeout_s']} s")
    if not yes:
        out("DRY RUN (nothing sent). With --yes this runs:")
        if not skip_code_upload:
            out(f"  {cli} " + " ".join(upload))
            out(f"  {cli} datasets status {d['code_dataset']} --format json   (until ready with a NEW version number)")
        out(f"  {cli} datasets files {d['code_dataset']}/<version> --format json   (must equal the local build)")
        out(f"  {cli} " + " ".join(push))
        return {"dry_run": True, **d}
    _, before = dataset_state(cli_run, d["code_dataset"])
    if skip_code_upload:
        version = before
    else:
        out(f"uploading code version (current version {before}) ...")
        rc, text = cli_run(*upload)
        out(text.strip())
        if rc != 0:
            raise LaunchError("code upload failed; not pushing the notebook")
        version = wait_for_version(cli_run, d["code_dataset"], before, sleep=sleep, clock=clock)
    verify_remote(cli_run, d["code_dataset"], version, build_sam)
    out(f"code dataset version {version} matches the local build")
    rc, text = cli_run(*push)
    out(text.strip())
    kv = kaggle_cli.parse_push(text)
    if rc != 0 or kv is None:
        raise LaunchError("push failed (see output above)")
    rec = {"kernel": d["kernel"], "version": kv, "pushed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "code_digest": d["code_digest"], "code_dataset_version": version, "notebook_sha256": d["notebook_sha256"],
           "code_uploaded": not skip_code_upload}
    (deploy_dir / "last_push.json").write_text(json.dumps(rec, indent=1) + "\n")
    out(f"pushed kernel version {kv} (code dataset v{version}). Monitor: "
        "Sam/.venv/bin/python Sam/scripts/monitor_kaggle.py")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy-dir", default=str(DEFAULT_DEPLOY))
    ap.add_argument("--kaggle", help="path to the Kaggle CLI (default: $KAGGLE_CLI or ~/.venvs/kaggle-cli/bin/kaggle)")
    ap.add_argument("--skip-code-upload", action="store_true", help="the current build is already the latest version")
    ap.add_argument("--yes", action="store_true", help="really upload and push (otherwise a dry run)")
    ap.add_argument("--new-version", action="store_true", help="push again although this deployment was already "
                                                               "pushed (starts a new GPU run of the same kernel)")
    a = ap.parse_args()
    cli = kaggle_cli.find_cli(a.kaggle)
    launch(Path(a.deploy_dir), lambda *x: kaggle_cli.run(cli, *x, timeout=600), cli, a.yes, a.skip_code_upload,
           new_version=a.new_version)


if __name__ == "__main__":
    main()
