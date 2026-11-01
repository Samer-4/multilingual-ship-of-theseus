"""Grouped-by-article classification shared by RQ2 (source attribution) and RQ3 (translator fingerprints).

Leakage rule: folds are formed over ORIGINAL ARTICLE IDS. Every text derived from an article (all source labels,
all translators, all rounds) is in the same fold, so no article is ever on both sides of a split.
Hyperparameters are fixed a priori (never tuned on test folds). Uncertainty: spread over repeated fold shufflings
plus a 95% article-cluster bootstrap over the pooled out-of-fold predictions.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LENGTH_COLS = ["log_words", "log_sentences", "sent_len_mean", "word_len_mean"]


def make_model(name: str, style_cols=None):
    """('text' | 'features', estimator). Fixed hyperparameters."""
    if name == "tfidf_word":
        return "text", Pipeline([("v", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                                 ("clf", LogisticRegression(C=10, max_iter=5000, class_weight="balanced"))])
    if name == "tfidf_char":
        return "text", Pipeline([("v", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                                       sublinear_tf=True, max_features=200_000)),
                                 ("clf", LogisticRegression(C=10, max_iter=5000, class_weight="balanced"))])
    if name in ("stylometric", "length_only"):
        # median imputation (fitted on the training fold only): MATTR is undefined for texts under 2 words,
        # which only occurs in the shared-opening ablation
        return "features", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                                     ("clf", LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced"))])
    raise ValueError(name)


def add_length_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_words"] = np.log1p(df["n_words"])
    df["log_sentences"] = np.log1p(df["n_sentences"])
    return df


def article_folds(articles, n_splits: int, seed: int) -> dict:
    """Shuffle the unique article ids and deal them round-robin into n_splits folds."""
    u = np.array(sorted(set(articles)))
    rng = np.random.default_rng(seed)
    u = u[rng.permutation(len(u))]
    return {a: i % n_splits for i, a in enumerate(u)}


def _xy(df, kind, cols, label):
    return (df["text"] if kind == "text" else df[cols].to_numpy(float)), df[label].to_numpy()


def grouped_cv(train_pool: pd.DataFrame, eval_sets: dict, label: str, model: str, cols=None, n_splits: int = 5,
               repeats: int = 5, seed: int = 6120, train_filter=None, keep=("text_id", "item_id", "article_id",
                                                                             "source_label", "translator", "stage")):
    """Train on `train_pool` rows of the training articles; predict every eval set's rows of the held-out articles.

    eval_sets: name -> DataFrame (same columns). Returns one row per (repeat, eval set, text) with y_true, y_pred.
    train_filter(train_df, fold_info) may further restrict the training rows (e.g. leave one source out).
    """
    out = []
    articles = set(train_pool.article_id)
    for df in eval_sets.values():
        articles |= set(df.article_id)
    for r in range(repeats):
        folds = article_folds(articles, n_splits, seed + r)
        for k in range(n_splits):
            tr = train_pool[train_pool.article_id.map(folds) != k]
            if train_filter is not None:
                tr = train_filter(tr)
            kind, est = make_model(model)
            X, y = _xy(tr, kind, cols, label)
            est.fit(X, y)
            for name, df in eval_sets.items():
                te = df[df.article_id.map(folds) == k]
                if not len(te):
                    continue
                Xt, yt = _xy(te, kind, cols, label)
                part = te[[c for c in keep if c in te]].copy()
                part["y_true"], part["y_pred"] = yt, est.predict(Xt)
                part["repeat"], part["fold"], part["eval_set"], part["model"] = r, k, name, model
                out.append(part)
    return pd.concat(out, ignore_index=True)


# -- scoring --------------------------------------------------------------------------------------------------

def _macro_f1(yt, yp, labels) -> float:
    f = []
    for c in labels:
        tp = np.sum((yt == c) & (yp == c))
        fp = np.sum((yt != c) & (yp == c))
        fn = np.sum((yt == c) & (yp != c))
        f.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(f))


def scores(preds: pd.DataFrame, by: list, labels: list, n_boot: int = 1000, seed: int = 6120) -> pd.DataFrame:
    """Accuracy and macro-F1 per group: mean and SD over repeats, and a 95% article-cluster bootstrap CI on the
    predictions pooled over repeats (articles resampled with replacement). n = unique texts behind the score."""
    rng = np.random.default_rng(seed)
    labels = list(labels)
    rows = []
    for g, part in preds.groupby(by, observed=True):
        g = g if isinstance(g, tuple) else (g,)
        per_rep = part.groupby("repeat").apply(
            lambda d: pd.Series({"acc": float((d.y_true == d.y_pred).mean()),
                                 "f1": _macro_f1(d.y_true.to_numpy(), d.y_pred.to_numpy(), labels)}),
            include_groups=False)
        arts = part.article_id.unique()
        idx = {a: np.flatnonzero(part.article_id.to_numpy() == a) for a in arts}
        yt, yp = part.y_true.to_numpy(), part.y_pred.to_numpy()
        boots_a, boots_f = [], []
        for _ in range(n_boot):
            sel = np.concatenate([idx[a] for a in rng.choice(arts, len(arts), replace=True)])
            boots_a.append((yt[sel] == yp[sel]).mean())
            boots_f.append(_macro_f1(yt[sel], yp[sel], labels))
        rows.append({**dict(zip(by, g)), "accuracy": per_rep.acc.mean(), "accuracy_sd": per_rep.acc.std(ddof=1),
                     "accuracy_ci_low": np.percentile(boots_a, 2.5), "accuracy_ci_high": np.percentile(boots_a, 97.5),
                     "macro_f1": per_rep.f1.mean(), "macro_f1_sd": per_rep.f1.std(ddof=1),
                     "macro_f1_ci_low": np.percentile(boots_f, 2.5), "macro_f1_ci_high": np.percentile(boots_f, 97.5),
                     "n_texts": part.text_id.nunique(), "n_articles": len(arts), "repeats": part.repeat.nunique(),
                     "chance": 1 / len(labels)})
    return pd.DataFrame(rows)


def per_class(preds: pd.DataFrame, by: list, labels: list) -> pd.DataFrame:
    """Precision / recall / F1 per class, pooled over repeats."""
    rows = []
    for g, part in preds.groupby(by, observed=True):
        g = g if isinstance(g, tuple) else (g,)
        for c in labels:
            tp = ((part.y_true == c) & (part.y_pred == c)).sum()
            fp = ((part.y_true != c) & (part.y_pred == c)).sum()
            fn = ((part.y_true == c) & (part.y_pred != c)).sum()
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            rows.append({**dict(zip(by, g)), "class": c, "precision": p, "recall": r,
                         "f1": 2 * p * r / (p + r) if p + r else 0.0,
                         "support_per_repeat": int((part.y_true == c).sum() / part.repeat.nunique())})
    return pd.DataFrame(rows)


def confusion(part: pd.DataFrame, labels: list, normalize: bool = True) -> pd.DataFrame:
    m = pd.crosstab(pd.Categorical(part.y_true, labels), pd.Categorical(part.y_pred, labels), dropna=False)
    m.index.name, m.columns.name = "true", "predicted"
    return m.div(m.sum(axis=1).replace(0, np.nan), axis=0) if normalize else m


def top_features(est, kind: str, cols=None, k: int = 15) -> pd.DataFrame:
    """Largest positive logistic-regression coefficients per class (fit on all data: interpretation only)."""
    clf = est.named_steps["clf"]
    names = est.named_steps["v"].get_feature_names_out() if kind == "text" else np.array(cols)
    rows = []
    for ci, c in enumerate(clf.classes_):
        coef = clf.coef_[ci] if clf.coef_.shape[0] > 1 else (clf.coef_[0] if ci == 1 else -clf.coef_[0])
        for j in np.argsort(coef)[::-1][:k]:
            rows.append({"class": c, "feature": names[j], "coef": float(coef[j])})
    return pd.DataFrame(rows)


# -- shared-opening ablation ----------------------------------------------------------------------------------

def shared_opening_lengths(df: pd.DataFrame, group=("article_id", "translator", "stage")) -> pd.Series:
    """Words at the start of each text that it shares VERBATIM with another source's text of the same article,
    translator and stage (the longest common whitespace-word prefix with any other text in that group).
    At T0 this recovers the 30-word human opening given to the generating LLMs. Nothing is assumed about its
    length in translations: it is measured, per text."""
    out = pd.Series(0, index=df.index, dtype=int)
    for _, g in df.groupby(list(group)):
        toks = {i: t.split() for i, t in zip(g.index, g.text)}
        for i in g.index:
            best = 0
            for j in g.index:
                if i == j:
                    continue
                a, b = toks[i], toks[j]
                n = 0
                for x, y in zip(a, b):
                    if x != y:
                        break
                    n += 1
                best = max(best, n)
            out[i] = best
    return out


def strip_opening(df: pd.DataFrame, lengths: pd.Series) -> pd.DataFrame:
    df = df.copy()
    df["text"] = [" ".join(t.split()[n:]) for t, n in zip(df.text, lengths.reindex(df.index).fillna(0).astype(int))]
    return df


def clean_feature_name(c: str) -> str:
    names = ["full stop", "comma", "semicolon", "colon", "exclamation", "question mark", "straight double quote",
             "straight apostrophe", "open paren", "close paren", "hyphen", "em dash", "en dash", "curly open quote",
             "curly close quote", "curly open single quote", "curly apostrophe"]
    if re.fullmatch(r"punct_\d\d", c):
        return f"punct: {names[int(c[6:])]}"
    return re.sub(r"^(fw|pos|dep)_", r"\1:", c)
