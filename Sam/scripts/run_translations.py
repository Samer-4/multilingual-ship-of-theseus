"""Command-line runner (the notebook does the same through sot_mt.session).

Examples (from the repo root):
  # what would run, without loading anything (the Stage 3B smoke-test items)
  Sam/.venv/bin/python Sam/scripts/run_translations.py --translator madlad-3b \
      --item-ids-file Sam/data/manifests/stage3b_madlad_smoke5.csv --plan-only
  # full local dry run with the deterministic mock (writes to Sam/runs_mock/, run_ids prefixed MOCK-)
  Sam/.venv/bin/python Sam/scripts/run_translations.py --translator nllb --max-items 5 --mock --export
  # on Kaggle, a real run (downloads allowed explicitly)
  python Sam/scripts/run_translations.py --translator madlad-3b --max-items 5 --allow-download --export
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sot_mt import progress, session, smoke  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--translator", required=True)
    ap.add_argument("--prompt-variant", default=None, help="default: mt-default for MT models, free for LLMs")
    ap.add_argument("--max-items", type=int)
    ap.add_argument("--article-ids", nargs="*")
    ap.add_argument("--item-ids", nargs="*")
    ap.add_argument("--item-ids-file", help="CSV with an item_id column, e.g. data/manifests/stage3b_madlad_smoke5.csv")
    ap.add_argument("--rounds", nargs=2, type=int, default=[1, 3])
    ap.add_argument("--stop-after")
    ap.add_argument("--checkpoint-every", type=int, default=5)
    ap.add_argument("--max-operations", type=int)
    ap.add_argument("--time-limit-minutes", type=float)
    ap.add_argument("--base", nargs="*", default=[], help="earlier session exports (generations_merged.parquet)")
    ap.add_argument("--prior-failures", nargs="*", default=[])
    ap.add_argument("--work-dir")
    ap.add_argument("--staged-dir")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--use-hf-token", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--no-progress", action="store_true", help="do not print SOTMT_PROGRESS event lines")
    a = ap.parse_args()

    pv = a.prompt_variant or ("free" if a.translator in {"qwen3-4b", "llama", "phi4-mini"} else "mt-default")
    item_ids = (a.item_ids or []) + (smoke.read_item_ids(a.item_ids_file) if a.item_ids_file else [])
    manifest, base = session.load_inputs(a.staged_dir, a.base)
    r = session.open_runner(a.translator, pv, manifest, base, work_dir=a.work_dir,
                            prior_failure_paths=a.prior_failures, mock=a.mock, allow_download=a.allow_download,
                            use_hf_token=a.use_hf_token, rounds=tuple(a.rounds), stop_after=a.stop_after,
                            max_items=a.max_items, article_ids=a.article_ids, item_ids=item_ids or None,
                            checkpoint_every=a.checkpoint_every,
                            max_operations=a.max_operations, time_limit_minutes=a.time_limit_minutes,
                            progress=None if a.no_progress else progress.Progress.stdout())
    print(f"pathway {a.translator}/{pv} | config {r.tr.generation_config_id} | work dir {r.work}")
    print(r.plan().to_string(index=False))
    if a.plan_only:
        return
    print(r.run())
    if a.export:
        print(r.export())


if __name__ == "__main__":
    main()
