from pathlib import Path

from sklearn.model_selection import GroupKFold

from sot_mt import paths


def test_article_grouping(real):
    m = real["manifest"]
    assert (m.group_id == m.dataset + "|" + m.article_id).all()
    assert m.group_id.nunique() == m.article_id.nunique() == 31
    # every text about the same article shares one group, across different source labels
    per_group = m.groupby("group_id").source_label.nunique()
    assert per_group.max() == 7 and (m.groupby("group_id").article_id.nunique() == 1).all()


def test_group_split_never_shares_articles(real):
    m = real["manifest"]
    for train, test in GroupKFold(n_splits=5).split(m, groups=m.group_id):
        assert not set(m.group_id.iloc[train]) & set(m.group_id.iloc[test])


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("SOT_CORPUS_DIR", str(tmp_path / "c"))
    monkeypatch.setenv("SAM_STAGED_DIR", str(tmp_path / "s"))
    monkeypatch.setenv("SAM_OUTPUT_DIR", str(tmp_path / "o"))
    assert paths.corpus_dir() == tmp_path / "c"
    assert paths.staged_dir() == tmp_path / "s"
    assert paths.output_root() == tmp_path / "o"


def test_kaggle_resolution(monkeypatch):
    for v in ["SOT_CORPUS_DIR", "SAM_STAGED_DIR", "SAM_OUTPUT_DIR"]:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    assert paths.corpus_dir() == Path("/kaggle/input/sot-corpus-4423d8c")
    assert paths.staged_dir() == Path("/kaggle/input/sam-sot-mt-staged")
    assert paths.output_root() == Path("/kaggle/working/sam_outputs")


def test_local_defaults_stay_inside_sam(monkeypatch):
    for v in ["SOT_CORPUS_DIR", "SAM_STAGED_DIR", "SAM_OUTPUT_DIR", "KAGGLE_KERNEL_RUN_TYPE"]:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(paths, "on_kaggle", lambda: False)
    for p in [paths.corpus_dir(), paths.staged_dir(), paths.output_root()]:
        assert paths.SAM_ROOT in [p, *p.parents]


def test_configs_have_no_absolute_paths():
    for f in (paths.SAM_ROOT / "configs").glob("*.yaml"):
        text = f.read_text()
        assert "/Users/" not in text and "/home/" not in text and "C:\\" not in text, f
