# -*- coding: utf-8 -*-
"""
Osteosarcoma AutoScore-Survival — complete analysis pipeline
============================================================

One script, one run, all reported results.

Input
-----
1. SEER + external cohort workbook
   sheet "SEER"        development cohort
   sheet "外部验证"     external validation cohort
2. Institutional pathology workbook (sheet "Sheet1") with Huvos grades

Output (written to ./results)
-----------------------------
summary.json                  every number reported in the manuscript
table1_baseline.csv           baseline characteristics
table2_score.csv              integer score and risk bands
table3_performance.csv        discrimination and calibration with 95% CIs
bands.csv                     observed vs predicted risk by band
deciles.csv                   decile-grouped calibration
dca.csv                       decision curve analysis
subgroups.csv                 subgroup discrimination
huvos.csv                     histological-response analysis
figures/Figure2.tif           parsimony curve + Kaplan-Meier by band
figures/Figure3.tif           calibration + decision curve
figures/Figure4.tif           subgroup forest plot
figures/SuppFigure_S1.tif     histological response
run.log                       full console log

Usage
-----
    python analysis_pipeline.py
Optional environment variables
    OS_FAST=1        fewer bootstrap replicates (quick check)
    OS_INPUT=path    override the main workbook
    OS_HUVOS=path    override the pathology workbook
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_LOCAL = Path(__file__).resolve().parent / "pylibs"
if _LOCAL.exists():
    sys.path.insert(0, str(_LOCAL))
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "_mplcache"))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import (
    brier_score,
    concordance_index_censored,
    cumulative_dynamic_auc,
)
from sksurv.util import Surv

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = RES / "figures"

BASE = Path(os.environ.get(
    "OS_BASE", r"C:\Users\秋心\Desktop\骨肿瘤SEER ML\临床数据"))
INPUT = Path(os.environ.get(
    "OS_INPUT", BASE / "骨肉瘤生存期预测--最终预测版本 - 副本.xlsx"))
HUVOS = Path(os.environ.get(
    "OS_HUVOS", BASE / "301---骨肉瘤随访  筛选完 - 副本.xlsx"))

FAST = os.environ.get("OS_FAST") == "1"
SEED = 2026
HORIZON = 36.0            # months; the single evaluation horizon
CV_REPEATS = 3 if FAST else 10
N_BOOT = 200 if FAST else 1000
CV_INNER = 5

# candidate variables, in rank order (race deliberately excluded)
CANDIDATES = ["lung_mets", "age", "primary_site", "grade", "tumor_size",
              "histology", "lymph_node", "other_mets", "sex", "delay"]
# four retained in the final score
FINAL = ["lung_mets", "age", "primary_site", "grade"]

# age and size are discretised exactly as in the score
AGE_BINS = [-np.inf, 10, 20, 40, 60, np.inf]
AGE_LABELS = ["<10", "10-19", "20-39", "40-59", ">=60"]
SIZE_BINS = [-np.inf, 50, 100, np.inf]
SIZE_LABELS = ["<=50", "51-100", ">100"]

SEER_MAP = {
    "年龄": "age_group",
    "性别": "sex",
    "种族": "race",
    "从诊断到治疗的时间（天）": "treatment_delay_days",
    "原发部位四肢骨0中轴骨1": "primary_site",
    "肿瘤分级(恶性低0高1)": "grade",
    "病理类型 .1": "histology",
    "肿瘤大小mm": "tumor_size_mm",
    "是否侵犯淋巴结": "lymph_node",
    "其他部位转移": "other_mets",
    "肺转移(否0是1)": "lung_mets",
    "化疗(否0术前1术后2前后3)": "chemo_seq",
    "放疗(否0术前1术后2前后3)": "radio_seq",
    "生存期（月）": "time_months",
    "死亡（否0是1）？": "event",
    "治疗情况": "treatment",
}
EXT_MAP = {
    "确诊时年龄": "age_group",
    "性别": "sex",
    "种族": "race",
    "从诊断到治疗的时间（天）": "treatment_delay_days",
    "原发部位四肢骨0中轴骨1": "primary_site",
    "肿瘤分级(低级别0高级别1)": "grade",
    "病理类型": "histology",
    "肿瘤大小(mm)": "tumor_size_mm",
    "淋巴结侵犯 否0是1": "lymph_node",
    "其他部位转移": "other_mets",
    "肺转移": "lung_mets",
    "化疗(否0术前1术后2前后3)": "chemo_seq",
    "放疗(否0术前1术后2)": "radio_seq",
    "治疗情况": "treatment",
    "生存期（月）": "time_months",
    "是否死亡(0否1是)": "event",
}

_LOG = []


def log(msg=""):
    print(msg)
    _LOG.append(str(msg))


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def age_midpoint(value):
    """Midpoint of a SEER five-year age band, e.g. '15-19 years' -> 17."""
    text = str(value).strip()
    m = stats_m = None
    import re
    m = re.match(r"^(\d+)-(\d+)", text)
    if m:
        return (int(m.group(1)) + int(m.group(2))) / 2
    m = re.match(r"^(\d+)\+", text)
    if m:
        return int(m.group(1)) + 2.5
    try:
        return float(text)
    except ValueError:
        return np.nan


def clean_sheet(df, mapping, cohort):
    df = df.rename(columns=mapping)
    df = df[list(mapping.values())].copy()
    df["cohort"] = cohort
    df["age_mid"] = df["age_group"].map(age_midpoint)
    for col in ("sex", "race", "histology", "treatment"):
        df[col] = df[col].map(
            lambda v: str(v).strip() if pd.notna(v) else np.nan).astype(object)
    df["sex"] = df["sex"].replace({"男": "Male", "女": "Female"})
    for col in ("treatment_delay_days", "tumor_size_mm", "time_months"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    df["event"] = pd.to_numeric(df["event"], errors="coerce")
    for col in ("primary_site", "grade", "lymph_node", "other_mets", "lung_mets"):
        num = pd.to_numeric(df[col], errors="coerce")
        num = num.where(num.isin({0, 1}))
        df[col] = num.map(lambda v: str(int(v)) if pd.notna(v) else np.nan)
    for col in ("chemo_seq", "radio_seq"):
        num = pd.to_numeric(df[col], errors="coerce")
        num = num.where(num.isin({0, 1, 2, 3}))
        df[col] = num.map(lambda v: str(int(v)) if pd.notna(v) else np.nan)
    df = df.dropna(subset=["time_months", "event"]).copy()
    df.loc[df["time_months"] <= 0, "time_months"] = 0.5
    df["event"] = df["event"].astype(int)
    df["age_cat"] = pd.cut(df["age_mid"], bins=AGE_BINS, labels=AGE_LABELS,
                           right=False).astype(object)
    df["size_cat"] = pd.cut(df["tumor_size_mm"], bins=SIZE_BINS,
                            labels=SIZE_LABELS, right=True).astype(object)
    df["delay_q"] = pd.qcut(df["treatment_delay_days"].rank(method="first"),
                            CV_INNER, labels=False)
    return df


def load_cohorts():
    raw = pd.read_excel(INPUT, sheet_name="SEER").iloc[1:].reset_index(drop=True)
    raw = raw.copy()
    raw["治疗情况"] = np.nan
    seer = clean_sheet(raw, SEER_MAP, "SEER")
    ext = clean_sheet(pd.read_excel(INPUT, sheet_name="外部验证"), EXT_MAP, "External")
    return seer, ext


# --------------------------------------------------------------------------- #
# Imputation (missForest via iterative random-forest regression)
# --------------------------------------------------------------------------- #
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

IMPUTE_NUM = ["age_mid", "treatment_delay_days", "tumor_size_mm"]
IMPUTE_CAT = ["sex", "race", "primary_site", "grade", "histology", "lymph_node",
              "other_mets", "lung_mets", "radio_seq", "chemo_seq"]


class MissForest:
    """Iterative random-forest imputation that leaves unseen categories alone."""

    def __init__(self, seed=SEED):
        self.seed = seed
        self.features = IMPUTE_NUM + IMPUTE_CAT

    def fit(self, df):
        self.maps_, self.levels_ = {}, {}
        for c in IMPUTE_CAT:
            vals = pd.Series(df[c]).dropna().astype(str).unique().tolist()
            vals = sorted(vals)
            self.levels_[c] = vals
            self.maps_[c] = {v: i for i, v in enumerate(vals)}
        enc = self._encode(df)
        self.imp_ = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=60, max_depth=8,
                                             min_samples_leaf=3, n_jobs=1,
                                             random_state=self.seed),
            max_iter=4, initial_strategy="most_frequent",
            imputation_order="ascending", skip_complete=True,
            random_state=self.seed)
        self.imp_.fit(enc)
        return self

    def _encode(self, df):
        out = pd.DataFrame(index=df.index)
        for c in IMPUTE_NUM:
            out[c] = pd.to_numeric(df[c], errors="coerce")
        for c in IMPUTE_CAT:
            s = df[c].astype(object)
            s = s.where(pd.notna(s), np.nan).map(
                lambda v: str(v) if pd.notna(v) else np.nan)
            out[c] = s.map(self.maps_[c]).astype(float)
        return out[self.features].to_numpy(float)

    def transform(self, df):
        enc = self.imp_.transform(self._encode(df))
        out = df.copy()
        idx = {f: i for i, f in enumerate(self.features)}
        for c in IMPUTE_NUM:
            out[c] = enc[:, idx[c]]
        for c in IMPUTE_CAT:
            levels = self.levels_[c]
            raw = np.rint(enc[:, idx[c]]).astype(int).clip(0, len(levels) - 1)
            out[c] = [levels[i] for i in raw]
        # rebuild derived variables from the imputed columns
        out["age_cat"] = pd.cut(out["age_mid"], bins=AGE_BINS, labels=AGE_LABELS,
                                right=False).astype(object)
        out["size_cat"] = pd.cut(out["tumor_size_mm"], bins=SIZE_BINS,
                                 labels=SIZE_LABELS, right=True).astype(object)
        out["delay_q"] = pd.qcut(out["treatment_delay_days"].rank(method="first"),
                                 CV_INNER, labels=False)
        return out


# --------------------------------------------------------------------------- #
# Design matrix for the Cox models
# --------------------------------------------------------------------------- #
ONEHOT = {
    "lung_mets": ("lung_mets", ["0", "1"]),
    "age": ("age_cat", AGE_LABELS),
    "primary_site": ("primary_site", ["0", "1"]),
    "grade": ("grade", ["0", "1"]),
    "tumor_size": ("size_cat", SIZE_LABELS),
    "histology": ("histology", None),
    "lymph_node": ("lymph_node", ["0", "1"]),
    "other_mets": ("other_mets", ["0", "1"]),
    "sex": ("sex", ["Female", "Male"]),
    "delay": ("delay_q", None),
}


def design(df, candidates):
    """One-hot design; first level of each variable is the reference."""
    frames, cols = {}, {}
    for name in candidates:
        src, levels = ONEHOT[name]
        s = df[src].astype(object)
        if levels is None:
            levels = sorted(pd.Series(s).dropna().astype(str).unique().tolist())
        names = []
        for lv in levels[1:]:
            key = f"{name}={lv}"
            frames[key] = (s.astype(str) == str(lv)).astype(float)
            names.append(key)
        cols[name] = names
    X = pd.DataFrame(frames, index=df.index)
    return X, cols


def cv_cindex(X, T, E, repeats=CV_REPEATS, seed=SEED):
    """Mean cross-validated C-index over repeated stratified 5-fold splits."""
    vals = []
    for r in range(repeats):
        skf = StratifiedKFold(CV_INNER, shuffle=True, random_state=seed + r)
        pred = np.zeros(len(T))
        for tr, te in skf.split(X, E):
            m = CoxPHSurvivalAnalysis(alpha=1.0).fit(
                X[tr], Surv.from_arrays(event=E[tr].astype(bool), time=T[tr]))
            pred[te] = m.predict(X[te])
        vals.append(float(concordance_index_censored(
            E.astype(bool), T, pred)[0]))
    return float(np.mean(vals)), float(np.std(vals)), np.asarray(vals)


# --------------------------------------------------------------------------- #
# Score construction
# --------------------------------------------------------------------------- #
def fit_points(df, features):
    """Fit the ridge Cox model and convert coefficients to integer points."""
    X, cols = design(df, features)
    y = Surv.from_arrays(event=df["event"].astype(bool),
                         time=df["time_months"].astype(float))
    model = CoxPHSurvivalAnalysis(alpha=1.0).fit(X.to_numpy(float), y)
    coef = pd.Series(model.coef_, index=X.columns)
    rows = []
    for name in features:
        values = [0.0] + [coef[c] for c in cols[name]]
        levels = ONEHOT[name][1]
        if levels is None:
            levels = sorted(df[ONEHOT[name][0]].dropna().astype(str).unique())
        shift = np.asarray(values) - min(values)
        pts = np.rint(shift / 0.25).astype(int)
        for lv, b, p in zip(levels, values, pts):
            rows.append({"feature": name, "category": lv, "beta": float(b),
                         "points": int(p)})
    table = pd.DataFrame(rows)
    table["variable"] = table["feature"].map(VAR_LABEL)
    return table, model, cols


def score(df, table, features):
    total = np.zeros(len(df), dtype=int)
    for name in features:
        sub = table[table["feature"] == name]
        src = ONEHOT[name][0]
        mapping = dict(zip(sub["category"].astype(str), sub["points"]))
        total += df[src].astype(str).map(mapping).fillna(0).to_numpy(int)
    return total


VAR_LABEL = {
    "lung_mets": "Lung metastases", "age": "Age at diagnosis, y",
    "primary_site": "Primary site", "grade": "Tumour grade",
    "tumor_size": "Tumour size, mm", "histology": "Histologic subtype",
    "lymph_node": "Lymph-node metastases", "other_mets": "Other-site metastases",
    "sex": "Sex", "delay": "Diagnosis-to-treatment interval, d",
}


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def km_curve(t, e):
    t, e = np.asarray(t, float), np.asarray(e, bool)
    order = np.argsort(t, kind="mergesort")
    tt, ee = t[order], e[order].astype(float)
    ut, idx = np.unique(tt, return_index=True)
    d = np.add.reduceat(ee, idx)
    n = (len(tt) - idx).astype(float)
    return ut, np.cumprod(np.where(n > 0, 1 - d / n, 1.0))


def km_risk(t, e, h=HORIZON):
    ut, s = km_curve(t, e)
    j = np.searchsorted(ut, h, side="right") - 1
    return 0.0 if j < 0 else float(1 - s[j])


def predict_risk(model, pts):
    sf = model.predict_survival_function(np.asarray(pts, float).reshape(-1, 1))
    return np.asarray([1 - f(HORIZON) for f in sf], dtype=float)


def metrics(p, T, E, y_ref, tmax):
    tt, ee = np.asarray(T, float), np.asarray(E, int)
    obs = (tt > HORIZON) | ((tt <= HORIZON) & (ee == 1))
    yb = ((tt <= HORIZON) & (ee == 1))[obs].astype(int)
    pb = np.clip(p[obs], 1e-6, 1 - 1e-6)
    out = {"n": int(len(tt)), "deaths": int(ee.sum()),
           "c_index": float(concordance_index_censored(ee.astype(bool), tt, p)[0]),
           "oe_ratio": float(yb.sum() / pb.sum())}
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(np.log(pb / (1 - pb)).reshape(-1, 1), yb)
    out["slope"] = float(lr.coef_[0][0])
    tt2 = np.minimum(tt, np.nextafter(tmax, 0))
    ee2 = np.where(tt >= tmax, False, ee.astype(bool))
    y2 = Surv.from_arrays(event=ee2, time=tt2)
    try:
        _, bs = brier_score(y_ref, y2, 1 - p, [HORIZON])
        out["brier"] = float(bs[0])
    except Exception:
        out["brier"] = float("nan")
    try:
        a, _ = cumulative_dynamic_auc(y_ref, y2, p, [HORIZON])
        out["auc"] = float(a[0])
    except Exception:
        out["auc"] = float("nan")
    return out


def bootstrap_metrics(p, T, E, y_ref, tmax, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        i = rng.integers(0, len(T), len(T))
        if E[i].sum() < 10:
            continue
        rows.append(metrics(p[i], T[i], E[i], y_ref, tmax))
    df = pd.DataFrame(rows)
    out = {}
    for key in ("c_index", "auc", "oe_ratio", "slope", "brier"):
        out[key] = (float(np.percentile(df[key].dropna(), 2.5)),
                    float(np.percentile(df[key].dropna(), 97.5)))
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
BLUE, TERRA, GOLD, GREY, GREEN = "#2E5F8A", "#C0653A", "#C9A227", "#7F7F7F", "#4F7A55"


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{name}.tif", dpi=300, bbox_inches="tight", facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    log(f"  figure written: {name}.tif")


def figure_flow(seer, ext, splits):
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.axis("off")
    ax.text(0.5, 0.97, "SEER registry 2010-2019", ha="center", fontsize=11,
            weight="bold")
    ax.text(0.5, 0.90, f"osteosarcoma, surgically treated\nn = {len(seer)} "
                       f"({int(seer['event'].sum())} deaths)",
            ha="center", fontsize=9)
    ax.annotate("", xy=(0.5, 0.72), xytext=(0.5, 0.84),
                arrowprops=dict(arrowstyle="->", color=GREY))
    ax.text(0.5, 0.77, "random 70:30 split", ha="center", fontsize=8.5, color=GREY)
    ax.text(0.25, 0.62, f"Training set\nn = {splits['n_train']}\n"
                        f"model fitting, cut-offs", ha="center", fontsize=9)
    ax.text(0.75, 0.62, f"Validation set\nn = {splits['n_valid']}\n"
                        f"evaluation", ha="center", fontsize=9)
    ax.annotate("", xy=(0.25, 0.30), xytext=(0.25, 0.53),
                arrowprops=dict(arrowstyle="->", color=GREY))
    ax.annotate("", xy=(0.75, 0.30), xytext=(0.75, 0.53),
                arrowprops=dict(arrowstyle="->", color=GREY))
    ax.text(0.25, 0.24, "four-variable\ninteger score", ha="center", fontsize=9)
    ax.text(0.75, 0.24, "internal estimate", ha="center", fontsize=9)
    ax.annotate("", xy=(0.5, 0.10), xytext=(0.25, 0.18),
                arrowprops=dict(arrowstyle="->", color=BLUE))
    ax.text(0.5, 0.06, f"External validation - Chinese cohort\nn = {len(ext)} "
                       f"({int(ext['event'].sum())} deaths)",
            ha="center", fontsize=9.5, weight="bold")
    save(fig, "Figure1")


def figure_parsimony(curve, km_bands):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    ax.errorbar(curve["k"], curve["c"], yerr=curve["sd"], fmt="o-", color=BLUE,
                ecolor="#9AAFC6", capsize=3, markersize=5, lw=1.4)
    ax.axvline(4, ls=":", color=TERRA, lw=1.2)
    ax.annotate("four variables\nretained",
                xy=(4.05, curve["c"].iloc[3]),
                xytext=(4.9, curve["c"].min() + 0.012), fontsize=8.5,
                color=TERRA, arrowprops=dict(arrowstyle="->", color=TERRA, lw=1))
    ax.set_xlabel("Number of ranked variables included", fontsize=10)
    ax.set_ylabel("Cross-validated C-index", fontsize=10)
    ax.set_xticks(curve["k"])
    ax.grid(alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("A  Cumulative performance in ranking order", fontsize=11)

    ax = axes[1]
    for label, col in (("Low", BLUE), ("Intermediate", GOLD), ("High", TERRA)):
        t, e, n = km_bands[label]
        ut, s = km_curve(t, e)
        ax.step(np.concatenate([[0], ut]), np.concatenate([[1], s]), where="post",
                color=col, lw=1.8, label=f"{label} (n={n})")
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time since diagnosis (months)", fontsize=10)
    ax.set_ylabel("Overall survival", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("B  Kaplan-Meier by risk band (external)", fontsize=11)
    fig.tight_layout()
    save(fig, "Figure2")


def figure_calibration(dec, dca):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    ax = axes[0]
    lim = max(0.85, float(dec["hi"].max()) * 1.08)
    ax.plot([0, lim], [0, lim], "--", color=GREY, lw=1)
    ax.errorbar(dec["pred"], dec["obs"],
                yerr=[dec["obs"] - dec["lo"], dec["hi"] - dec["obs"]],
                fmt="o", color=BLUE, ecolor="#9AAFC6", capsize=3, markersize=5)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Mean predicted event risk", fontsize=10)
    ax.set_ylabel("Kaplan-Meier observed event risk", fontsize=10)
    ax.set_title("A  Calibration at 3 years (external)", fontsize=11)
    ax.grid(alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    ax = axes[1]
    ax.plot(dca["threshold"], dca["net_benefit"], "-", lw=2, color=BLUE,
            label="Integer score")
    ax.plot(dca["threshold"], dca["treat_all"], "--", lw=1.4, color=TERRA,
            label="Treat all")
    ax.axhline(0, ls=":", color=GREY, lw=1, label="Treat none")
    ax.set_xlabel("Threshold probability", fontsize=10)
    ax.set_ylabel("Net benefit", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("B  Decision curve at 3 years", fontsize=11)
    ax.grid(alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    save(fig, "Figure3")


def figure_subgroups(sub, overall):
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    y = np.arange(len(sub))[::-1]
    ax.axvline(overall, ls="--", lw=1, color=GREY,
               label=f"Overall C-index = {overall:.3f}")
    ax.errorbar(sub["c"], y, xerr=[sub["c"] - sub["lo"], sub["hi"] - sub["c"]],
                fmt="o", color=BLUE, ecolor=BLUE, capsize=4, markersize=6, lw=0)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.subgroup}  (n={r.n}, {r.deaths} deaths)"
                        for r in sub.itertuples()], fontsize=9)
    ax.set_xlabel("C-index (95% CI), 3-year discrimination", fontsize=10)
    ax.set_xlim(0.40, 0.92)
    ax.grid(axis="x", alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("Subgroup discrimination in the external cohort", fontsize=11)
    fig.tight_layout()
    save(fig, "Figure4")


def figure_huvos(huv):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    ax = axes[0]
    for label, col, mask in (("Good response (Huvos 3-4)", BLUE, huv["good"]),
                             ("Poor response (Huvos 1-2)", TERRA, huv["poor"])):
        t, e = huv["t"][mask], huv["e"][mask]
        ut, s = km_curve(t, e)
        ax.step(np.concatenate([[0], ut]), np.concatenate([[1], s]), where="post",
                color=col, lw=1.9, label=f"{label}, n={int(mask.sum())}")
    ax.axvline(HORIZON, ls=":", color=GREY, lw=1)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time since diagnosis (months)", fontsize=10)
    ax.set_ylabel("Overall survival", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.grid(alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("A  Overall survival by histological response", fontsize=11)

    ax = axes[1]
    labs = ["Poor response\n(Huvos 1-2)", "Good response\n(Huvos 3-4)"]
    est = [huv["poor_risk"], huv["good_risk"]]
    lo = [huv["poor_lo"], huv["good_lo"]]
    hi = [huv["poor_hi"], huv["good_hi"]]
    x = np.arange(2)
    ax.bar(x, est, width=0.5, color=[TERRA, BLUE], alpha=0.9)
    ax.errorbar(x, est, yerr=[np.array(est) - lo, np.array(hi) - est],
                fmt="none", ecolor="black", capsize=4, lw=1)
    for i, v in enumerate(est):
        ax.text(i, hi[i] + 1.5, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=9)
    ax.set_ylabel("3-year event risk (%)", fontsize=10)
    ax.set_ylim(0, max(hi) * 1.35)
    ax.grid(axis="y", alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("B  3-year event risk", fontsize=11)
    fig.tight_layout()
    save(fig, "SuppFigure_S1")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    RES.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    log("=" * 72)
    log("Osteosarcoma AutoScore-Survival pipeline")
    log(f"input   : {INPUT}")
    log(f"pathology: {HUVOS}")
    log(f"repeats  : {CV_REPEATS}   bootstrap: {N_BOOT}")
    log("=" * 72)

    # ---------------- 1. cohorts ------------------------------------------ #
    seer, ext = load_cohorts()
    log(f"\n[1] cohorts: SEER {len(seer)} ({int(seer['event'].sum())} deaths), "
        f"external {len(ext)} ({int(ext['event'].sum())} deaths)")
    log(f"    SEER 3-year KM risk {km_risk(seer['time_months'], seer['event'])*100:.1f}%"
        f" | external {km_risk(ext['time_months'], ext['event'])*100:.1f}%")

    tr_idx, va_idx = train_test_split(
        np.arange(len(seer)), test_size=0.30, random_state=SEED,
        stratify=seer["event"])
    train = seer.iloc[tr_idx].reset_index(drop=True)
    valid = seer.iloc[va_idx].reset_index(drop=True)
    log(f"    training {len(train)} | validation {len(valid)}")

    # ---------------- 2. imputation --------------------------------------- #
    imp = MissForest().fit(train)
    train_i = imp.transform(train)
    valid_i = imp.transform(valid)
    seer_i = imp.transform(seer)
    ext_i = imp.transform(ext)
    # delay quintiles must come from the training set and be applied everywhere
    q = train_i["treatment_delay_days"].quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
    dbin = np.concatenate([[-np.inf], q, [np.inf]])
    for frame in (train_i, valid_i, seer_i, ext_i):
        frame["delay_q"] = pd.cut(frame["treatment_delay_days"], bins=dbin,
                                  labels=False)
    need = train[["grade", "tumor_size_mm", "lymph_node", "lung_mets"]].isna().any(axis=1)
    log(f"[2] imputation fitted on training set; "
        f"{need.mean()*100:.1f}% of training patients affected")

    # ---------------- 3. ranking ------------------------------------------ #
    Xtr, _ = design(train_i, CANDIDATES)
    Xva, _ = design(valid_i, CANDIDATES)
    Xva = Xva.reindex(columns=Xtr.columns, fill_value=0.0)
    T_tr = train_i["time_months"].to_numpy(float)
    E_tr = train_i["event"].to_numpy(int)
    y_tr = Surv.from_arrays(event=E_tr.astype(bool), time=T_tr)

    rsf = RandomSurvivalForest(n_estimators=500, min_samples_split=10,
                               min_samples_leaf=5, max_features="sqrt",
                               random_state=SEED, n_jobs=1).fit(
        Xtr.to_numpy(float), y_tr)
    base = concordance_index_censored(
        valid_i["event"].astype(bool), valid_i["time_months"].to_numpy(float),
        rsf.predict(Xva.to_numpy(float)))[0]
    rng = np.random.default_rng(SEED)
    Xv = Xva.to_numpy(float)
    imp_rows = []
    for name in CANDIDATES:
        idx = [Xtr.columns.get_loc(c) for c in design(train_i, [name])[1][name]]
        drops = []
        for _ in range(20):
            Xp = Xv.copy()
            perm = rng.permutation(Xp.shape[0])
            Xp[:, idx] = Xp[perm][:, idx]
            drops.append(base - concordance_index_censored(
                valid_i["event"].astype(bool),
                valid_i["time_months"].to_numpy(float), rsf.predict(Xp))[0])
        imp_rows.append({"variable": name, "label": VAR_LABEL[name],
                         "importance": float(np.mean(drops)),
                         "sd": float(np.std(drops))})
    rank = pd.DataFrame(imp_rows).sort_values("importance", ascending=False)
    rank["rank"] = np.arange(1, len(rank) + 1)
    rank = rank.reset_index(drop=True)
    log("\n[3] permutation importance (random survival forest, validation set)")
    for r in rank.itertuples():
        log(f"    {r.rank:2d}. {r.label:35s} {r.importance:+.4f} (SD {r.sd:.4f})")
    order = rank["variable"].tolist()

    # ---------------- 4. parsimony curve ---------------------------------- #
    log("\n[4] parsimony curve (repeated 5-fold CV inside the training set)")
    curve_rows = []
    for k in range(1, len(order) + 1):
        feats = order[:k]
        Xk, _ = design(train_i, feats)
        mu, sd, _ = cv_cindex(Xk.to_numpy(float), T_tr, E_tr)
        curve_rows.append({"k": k, "variable": VAR_LABEL[order[k - 1]],
                           "c": mu, "sd": sd})
        log(f"    k={k:2d} {VAR_LABEL[order[k-1]]:35s} {mu:.4f} (SD {sd:.4f})")
    curve = pd.DataFrame(curve_rows)
    inc = curve["c"].diff()
    curve["increment"] = inc
    stop = None
    for i in range(1, len(curve)):
        if curve["increment"].iloc[i] < 0.01:
            stop = int(curve["k"].iloc[i - 1])
            break
    stop = stop or len(curve)
    retained = order[:stop]
    log(f"    first increment below 0.010 at k={stop+1}; "
        f"retained {stop} variables: {', '.join(VAR_LABEL[v] for v in retained)}")
    curve.to_csv(RES / "parsimony_curve.csv", index=False, encoding="utf-8-sig")

    # ---------------- 5. score -------------------------------------------- #
    log("\n[5] integer score")
    table_all, cox_full, cols_full = fit_points(seer_i, retained)
    table_tr, _, _ = fit_points(train_i, retained)
    seer_s = score(seer_i, table_all, retained)
    ext_s = score(ext_i, table_all, retained)
    train_s = score(train_i, table_tr, retained)
    valid_s = score(valid_i, table_tr, retained)
    max_points = int(table_all.groupby("feature")["points"].max().sum())
    log(table_all[["variable", "category", "beta", "points"]]
        .to_string(index=False))
    log(f"    maximum {max_points} points; "
        f"external range {ext_s.min()}-{ext_s.max()}")
    table_all.to_csv(RES / "table2_score_raw.csv", index=False, encoding="utf-8-sig")

    m_int = CoxPHSurvivalAnalysis(alpha=0.0).fit(
        train_s.reshape(-1, 1).astype(float), y_tr)
    y_se = Surv.from_arrays(event=seer_i["event"].astype(bool),
                            time=seer_i["time_months"].astype(float))
    m_ext = CoxPHSurvivalAnalysis(alpha=0.0).fit(
        seer_s.reshape(-1, 1).astype(float), y_se)

    vals = np.arange(train_s.min(), train_s.max() + 1)
    sf = m_int.predict_survival_function(vals.reshape(-1, 1).astype(float))
    risk_by_score = np.array([1 - f(HORIZON) for f in sf])
    lo_cut = int(vals[risk_by_score < 0.10].max())
    hi_cut = int(vals[risk_by_score >= 0.40].min())
    log(f"    cut-offs: low <={lo_cut}, intermediate {lo_cut+1}-{hi_cut-1}, "
        f"high >={hi_cut}")

    # ---------------- 6. evaluation --------------------------------------- #
    p_int = predict_risk(m_int, valid_s)
    p_ext = predict_risk(m_ext, ext_s)
    T_va = valid_i["time_months"].to_numpy(float)
    E_va = valid_i["event"].to_numpy(int)
    T_ex = ext_i["time_months"].to_numpy(float)
    E_ex = ext_i["event"].to_numpy(int)
    tmax_tr = float(train_i["time_months"].max())
    tmax_se = float(seer_i["time_months"].max())

    log("\n[6] performance with bootstrap 95% CI")
    internal = metrics(p_int, T_va, E_va, y_tr, tmax_tr)
    internal_ci = bootstrap_metrics(p_int, T_va, E_va, y_tr, tmax_tr)
    external = metrics(p_ext, T_ex, E_ex, y_se, tmax_se)
    external_ci = bootstrap_metrics(p_ext, T_ex, E_ex, y_se, tmax_se)
    for name, pt, ci in (("internal", internal, internal_ci),
                         ("external", external, external_ci)):
        log(f"    {name}: C {pt['c_index']:.3f} "
            f"({ci['c_index'][0]:.3f}-{ci['c_index'][1]:.3f}) | "
            f"AUC {pt['auc']:.3f} ({ci['auc'][0]:.3f}-{ci['auc'][1]:.3f}) | "
            f"O/E {pt['oe_ratio']:.3f} ({ci['oe_ratio'][0]:.3f}-{ci['oe_ratio'][1]:.3f})")
        log(f"                 slope {pt['slope']:.3f} "
            f"({ci['slope'][0]:.3f}-{ci['slope'][1]:.3f}) | "
            f"Brier {pt['brier']:.3f} ({ci['brier'][0]:.3f}-{ci['brier'][1]:.3f})")

    # ---------------- 7. bands, deciles, DCA ------------------------------ #
    band = np.where(ext_s <= lo_cut, "Low",
                    np.where(ext_s >= hi_cut, "High", "Intermediate"))
    band_rows, km_bands = [], {}
    for label in ("Low", "Intermediate", "High"):
        m = band == label
        obs = km_risk(T_ex[m], E_ex[m])
        band_rows.append({"band": label, "n": int(m.sum()),
                          "percent": round(100 * m.mean(), 1),
                          "predicted": round(100 * p_ext[m].mean(), 1),
                          "observed": round(100 * obs, 1),
                          "oe": round(obs / p_ext[m].mean(), 2)})
        km_bands[label] = (T_ex[m], E_ex[m], int(m.sum()))
    bands = pd.DataFrame(band_rows)
    log("\n[7] risk bands (external)")
    log(bands.to_string(index=False))
    bands.to_csv(RES / "bands.csv", index=False, encoding="utf-8-sig")

    rngd = np.random.default_rng(SEED)
    order_risk = np.lexsort((rngd.random(len(p_ext)), p_ext))
    dec_idx = np.empty(len(p_ext), int)
    dec_idx[order_risk] = np.floor(np.arange(len(p_ext)) * 10 / len(p_ext)).astype(int)
    dec_rows = []
    for b in range(10):
        m = dec_idx == b
        boot = []
        rb = np.random.default_rng(SEED + b)
        for _ in range(300):
            i = rb.integers(0, int(m.sum()), int(m.sum()))
            if E_ex[m][i].sum() >= 2:
                boot.append(km_risk(T_ex[m][i], E_ex[m][i]))
        dec_rows.append({"decile": b + 1, "n": int(m.sum()),
                         "pred": float(p_ext[m].mean()),
                         "obs": km_risk(T_ex[m], E_ex[m]),
                         "lo": float(np.percentile(boot, 2.5)),
                         "hi": float(np.percentile(boot, 97.5))})
    dec = pd.DataFrame(dec_rows)
    dec.to_csv(RES / "deciles.csv", index=False, encoding="utf-8-sig")

    thr = np.round(np.arange(0.05, 0.51, 0.05), 2)
    obs_mask = (T_ex > HORIZON) | ((T_ex <= HORIZON) & (E_ex == 1))
    y_bin = ((T_ex <= HORIZON) & (E_ex == 1))[obs_mask]
    p_bin = p_ext[obs_mask]
    n_bin = len(y_bin)
    dca_rows = []
    for t in thr:
        pos = p_bin >= t
        tp = int((pos & y_bin).sum())
        fp = int((pos & ~y_bin).sum())
        dca_rows.append({"threshold": t,
                         "net_benefit": tp / n_bin - fp / n_bin * (t / (1 - t)),
                         "treat_all": y_bin.sum() / n_bin
                         - (n_bin - y_bin.sum()) / n_bin * (t / (1 - t))})
    dca = pd.DataFrame(dca_rows)
    dca.to_csv(RES / "dca.csv", index=False, encoding="utf-8-sig")

    # ---------------- 8. subgroups ---------------------------------------- #
    age = pd.to_numeric(ext_i["age_mid"], errors="coerce").to_numpy(float)
    male = (ext_i["sex"].astype(str) == "Male").to_numpy()
    lung = (ext_i["lung_mets"].astype(str) == "1").to_numpy()
    high = (ext_i["grade"].astype(str) == "1").to_numpy()
    defs = {
        "Male": male, "Female": ~male,
        "Age <20": age < 20, "Age 20-39": (age >= 20) & (age < 40),
        "Age >=40": age >= 40,
        "No lung metastasis": ~lung, "Lung metastasis": lung,
        "Low grade": ~high, "High grade": high,
    }
    rngs = np.random.default_rng(SEED)
    sub_rows = []
    for name, mask in defs.items():
        if mask.sum() < 20 or E_ex[mask].sum() < 10:
            continue
        c = concordance_index_censored(E_ex[mask].astype(bool), T_ex[mask],
                                       p_ext[mask])[0]
        boot = []
        for _ in range(500):
            i = rngs.integers(0, int(mask.sum()), int(mask.sum()))
            if E_ex[mask][i].sum() >= 2:
                boot.append(concordance_index_censored(
                    E_ex[mask][i].astype(bool), T_ex[mask][i],
                    p_ext[mask][i])[0])
        sub_rows.append({"subgroup": name, "n": int(mask.sum()),
                         "deaths": int(E_ex[mask].sum()), "c": float(c),
                         "lo": float(np.percentile(boot, 2.5)),
                         "hi": float(np.percentile(boot, 97.5))})
    sub = pd.DataFrame(sub_rows)
    log("\n[8] subgroups (external)")
    log(sub.round(3).to_string(index=False))
    sub.to_csv(RES / "subgroups.csv", index=False, encoding="utf-8-sig")

    # ---------------- 9. Huvos -------------------------------------------- #
    log("\n[9] histological response")
    huv_doc = None
    if HUVOS.exists():
        huv_doc = pd.read_excel(HUVOS, sheet_name="Sheet1")
    if huv_doc is not None:
        h = huv_doc.copy()
        h["ab"] = pd.to_numeric(h["确诊时年龄"], errors="coerce").map(
            lambda v: (f"{int(v//5)*5:02d}-{int(v//5)*5+4:02d} years"
                       if pd.notna(v) and v >= 5 else
                       ("01-04 years" if pd.notna(v) else None)))
        prof = ["ab", "sex", "size", "site", "grade", "ln", "lung", "om", "ev", "delay"]
        K = pd.DataFrame({
            "ab": h["ab"], "sex": h["性别"].astype(str).str.strip(),
            "size": pd.to_numeric(h["肿瘤大小(mm)"], errors="coerce").round(1),
            "site": pd.to_numeric(h["原发部位四肢骨0中轴骨1"], errors="coerce"),
            "grade": pd.to_numeric(h["肿瘤分级(低级别0高级别1)"], errors="coerce"),
            "ln": pd.to_numeric(h["淋巴结侵犯 否0是1"], errors="coerce"),
            "lung": pd.to_numeric(h["初诊时肺转移"], errors="coerce"),
            "om": pd.to_numeric(h["初诊时其他部位转移"], errors="coerce"),
            "ev": pd.to_numeric(h["是否死亡(0否1是)"], errors="coerce"),
            "delay": pd.to_numeric(h["从诊断到治疗的时间（天）"], errors="coerce").round(2),
            "huvos": pd.to_numeric(h["Huvos分级"], errors="coerce"),
            "t": pd.to_numeric(h["生存期（月）"], errors="coerce"),
        })
        raw_ext = pd.read_excel(INPUT, sheet_name="外部验证")
        Ee = pd.DataFrame({
            "ab": raw_ext["确诊时年龄"].astype(str).str.strip(),
            "sex": raw_ext["性别"].astype(str).str.strip(),
            "size": pd.to_numeric(raw_ext["肿瘤大小(mm)"], errors="coerce").round(1),
            "site": pd.to_numeric(raw_ext["原发部位四肢骨0中轴骨1"], errors="coerce"),
            "grade": pd.to_numeric(raw_ext["肿瘤分级(低级别0高级别1)"], errors="coerce"),
            "ln": pd.to_numeric(raw_ext["淋巴结侵犯 否0是1"], errors="coerce"),
            "lung": pd.to_numeric(raw_ext["肺转移"], errors="coerce"),
            "om": pd.to_numeric(raw_ext["其他部位转移"], errors="coerce"),
            "ev": pd.to_numeric(raw_ext["是否死亡(0否1是)"], errors="coerce"),
            "delay": pd.to_numeric(raw_ext["从诊断到治疗的时间（天）"], errors="coerce").round(2),
            "t": pd.to_numeric(raw_ext["生存期（月）"], errors="coerce"),
        })
        Ee["idx"] = np.arange(len(Ee))
        Ee["p"] = p_ext

        kk = Ee[prof].apply(tuple, axis=1)
        kh = K[prof].apply(tuple, axis=1)
        uniq = (set(kk.value_counts()[lambda s: s == 1].index)
                & set(kh.value_counts()[lambda s: s == 1].index))
        first = Ee[kk.isin(uniq)].merge(K[kh.isin(uniq)], on=prof, how="inner",
                                        suffixes=("", "_h"))
        surv = first[first["ev"] == 0]
        offset = float(np.median(surv["t"] - surv["t_h"])) if len(surv) else 0.0
        Ee["tc"] = np.where(Ee["ev"] == 1, Ee["t"], Ee["t"] - offset).round(2)
        K["tc"] = K["t"].round(2)
        prof_t = prof + ["tc"]
        kk2 = Ee[prof_t].apply(tuple, axis=1)
        kh2 = K[prof_t].apply(tuple, axis=1)
        uniq2 = (set(kk2.value_counts()[lambda s: s == 1].index)
                 & set(kh2.value_counts()[lambda s: s == 1].index))
        linked = Ee[kk2.isin(uniq2)].merge(K[kh2.isin(uniq2)], on=prof_t,
                                           how="inner", suffixes=("", "_h"))
        linked = linked.drop_duplicates(subset=["idx"])

        # survivors whose follow-up was extended cannot be matched on exact time;
        # accept them when their profile is unique among the unused records
        used = set(linked["idx"])
        pool = K[K["huvos"].notna()].copy()
        pool["used"] = False
        extra = []
        for _, row in Ee[~Ee["idx"].isin(used) & (Ee["ev"] == 0)].iterrows():
            cand = pool[(pool["ab"] == row["ab"]) & (pool["sex"] == row["sex"])
                        & (pool["size"] == row["size"]) & (pool["site"] == row["site"])
                        & (pool["grade"] == row["grade"]) & (pool["ln"] == row["ln"])
                        & (pool["lung"] == row["lung"]) & (pool["om"] == row["om"])
                        & (pool["ev"] == 0) & (~pool["used"])
                        & (pool["t"] <= row["t"] + 1e-6)]
            if len(cand) == 1:
                pool.loc[cand.index, "used"] = True
                extra.append({"idx": row["idx"], "t": row["t"], "ev": row["ev"],
                              "p": row["p"], "huvos": float(cand["huvos"].iloc[0])})
        if extra:
            linked = pd.concat([linked, pd.DataFrame(extra)], ignore_index=True)
        d = linked[linked["huvos"].notna()].copy()
        log(f"    survival-time offset for survivors: {offset:.4f} months")
        log(f"    linked {len(d)} patients with a Huvos grade "
            f"({int(d['ev'].sum())} deaths)")

        T_h = d["t"].to_numpy(float)
        E_h = d["ev"].to_numpy(int)
        H_h = d["huvos"].to_numpy(float)
        R_h = d["p"].to_numpy(float)
        poor, good = H_h <= 2, H_h >= 3

        def boot_risk(mask, n=N_BOOT):
            rng_h = np.random.default_rng(SEED)
            vals = []
            for _ in range(n):
                i = rng_h.integers(0, int(mask.sum()), int(mask.sum()))
                if E_h[mask][i].sum() >= 3:
                    vals.append(km_risk(T_h[mask][i], E_h[mask][i]))
            return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

        def cv_rep(X, reps=200):
            rng_c = np.random.default_rng(SEED)
            vals = []
            for _ in range(reps):
                skf = StratifiedKFold(CV_INNER, shuffle=True,
                                      random_state=int(rng_c.integers(1e6)))
                pred = np.zeros(len(T_h))
                for a, b in skf.split(X, E_h):
                    mm = CoxPHSurvivalAnalysis(alpha=0.0).fit(
                        X[a], Surv.from_arrays(event=E_h[a].astype(bool),
                                               time=T_h[a]))
                    pred[b] = mm.predict(X[b])
                vals.append(concordance_index_censored(
                    E_h.astype(bool), T_h, pred)[0])
            return np.asarray(vals)

        c0 = cv_rep(R_h.reshape(-1, 1))
        c1 = cv_rep(np.column_stack([R_h, H_h]))
        c2 = cv_rep(np.column_stack([R_h, good.astype(float)]))
        lo_p, hi_p = boot_risk(poor)
        lo_g, hi_g = boot_risk(good)
        mb = CoxPHSurvivalAnalysis(alpha=0.0).fit(
            np.column_stack([R_h, good.astype(float)]),
            Surv.from_arrays(event=E_h.astype(bool), time=T_h))
        hr = float(np.exp(mb.coef_[1]))
        rng_hr = np.random.default_rng(SEED)
        hr_boot = []
        for _ in range(N_BOOT):
            i = rng_hr.integers(0, len(T_h), len(T_h))
            if E_h[i].sum() < 20:
                continue
            try:
                mm = CoxPHSurvivalAnalysis(alpha=0.0).fit(
                    np.column_stack([R_h, good.astype(float)])[i],
                    Surv.from_arrays(event=E_h[i].astype(bool), time=T_h[i]))
                hr_boot.append(float(np.exp(mm.coef_[1])))
            except Exception:
                pass
        huv = {"n": int(len(d)), "deaths": int(E_h.sum()),
               "poor_n": int(poor.sum()), "poor_deaths": int(E_h[poor].sum()),
               "poor_risk": km_risk(T_h[poor], E_h[poor]) * 100,
               "poor_lo": lo_p * 100, "poor_hi": hi_p * 100,
               "good_n": int(good.sum()), "good_deaths": int(E_h[good].sum()),
               "good_risk": km_risk(T_h[good], E_h[good]) * 100,
               "good_lo": lo_g * 100, "good_hi": hi_g * 100,
               "c0": float(c0.mean()), "c0_lo": float(np.percentile(c0, 2.5)),
               "c0_hi": float(np.percentile(c0, 97.5)),
               "c1": float(c1.mean()), "c1_lo": float(np.percentile(c1, 2.5)),
               "c1_hi": float(np.percentile(c1, 97.5)),
               "c2": float(c2.mean()), "c2_lo": float(np.percentile(c2, 2.5)),
               "c2_hi": float(np.percentile(c2, 97.5)),
               "d1": float((c1 - c0).mean()), "d1_lo": float(np.percentile(c1 - c0, 2.5)),
               "d1_hi": float(np.percentile(c1 - c0, 97.5)),
               "d2": float((c2 - c0).mean()), "d2_lo": float(np.percentile(c2 - c0, 2.5)),
               "d2_hi": float(np.percentile(c2 - c0, 97.5)),
               "hr": hr, "hr_lo": float(np.percentile(hr_boot, 2.5)),
               "hr_hi": float(np.percentile(hr_boot, 97.5)),
               "poor": poor, "good": good, "t": T_h, "e": E_h}
        log(f"    poor response {huv['poor_risk']:.1f}% "
            f"({huv['poor_lo']:.1f}-{huv['poor_hi']:.1f}) | good "
            f"{huv['good_risk']:.1f}% ({huv['good_lo']:.1f}-{huv['good_hi']:.1f})")
        log(f"    C-index {huv['c0']:.3f} -> ordinal {huv['c1']:.3f} "
            f"(+{huv['d1']:.3f}) -> binary {huv['c2']:.3f} (+{huv['d2']:.3f})")
        log(f"    adjusted HR for good response {huv['hr']:.2f} "
            f"({huv['hr_lo']:.2f}-{huv['hr_hi']:.2f})")
        figure_huvos(huv)
        pd.DataFrame([{k: v for k, v in huv.items()
                       if not isinstance(v, (np.ndarray,))}]).to_csv(
            RES / "huvos.csv", index=False, encoding="utf-8-sig")
    else:
        huv = None
        log("    pathology workbook not found; Huvos analysis skipped")

    # ---------------- 10. baseline table ---------------------------------- #
    log("\n[10] baseline table")
    allc = pd.concat([seer, ext], ignore_index=True)
    cohorts = [("ALL", allc), ("Training", train), ("Validation", valid),
               ("External", ext)]
    rows = []

    def add(label, fn):
        rows.append({"Characteristic": label,
                     **{name: fn(df) for name, df in cohorts}})

    def pct_count(df, col, val):
        n = int((df[col].astype(str) == val).sum())
        return f"{n} ({100*n/len(df):.1f})"

    def med(df, col):
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        return f"{s.median():.1f} ({s.quantile(.25):.1f}-{s.quantile(.75):.1f})"

    add("Age, y, median (IQR)", lambda d: med(d, "age_mid"))
    add("Sex, male", lambda d: pct_count(d, "sex", "Male"))
    add("Primary site, appendicular", lambda d: pct_count(d, "primary_site", "0"))
    add("Primary site, axial", lambda d: pct_count(d, "primary_site", "1"))
    for code in ("9180/3", "9184/3", "9187/3", "9192/3", "9193/3", "9194/3"):
        add(f"Histology {code}", lambda d, c=code: pct_count(d, "histology", c))
    add("Grade, low", lambda d: pct_count(d, "grade", "0"))
    add("Grade, high", lambda d: pct_count(d, "grade", "1"))
    add("Grade, missing", lambda d: f"{int(d['grade'].isna().sum())} "
                                    f"({100*d['grade'].isna().mean():.1f})")
    add("Tumour size, mm, median (IQR)", lambda d: med(d, "tumor_size_mm"))
    add("Lung metastasis", lambda d: pct_count(d, "lung_mets", "1"))
    add("Lymph-node metastasis", lambda d: pct_count(d, "lymph_node", "1"))
    add("Other-site metastasis", lambda d: pct_count(d, "other_mets", "1"))
    add("Perioperative chemotherapy", lambda d: pct_count(d, "chemo_seq", "3"))
    add("Postoperative radiotherapy", lambda d: pct_count(d, "radio_seq", "2"))
    add("Diagnosis-to-treatment interval, d", lambda d: med(d, "treatment_delay_days"))
    add("Deaths", lambda d: f"{int(d['event'].sum())} "
                            f"({100*d['event'].mean():.1f})")
    add("Observed follow-up, mo, median (IQR)", lambda d: med(d, "time_months"))
    t1 = pd.DataFrame(rows)
    t1.to_csv(RES / "table1_baseline.csv", index=False, encoding="utf-8-sig")

    # ---------------- 11. performance table ------------------------------- #
    t3 = pd.DataFrame({
        "metric": ["C-index", "3-year AUC", "Calibration slope", "O/E ratio",
                   "Brier score"],
        "internal": [
            f"{internal['c_index']:.3f} ({internal_ci['c_index'][0]:.3f}-"
            f"{internal_ci['c_index'][1]:.3f})",
            f"{internal['auc']:.3f} ({internal_ci['auc'][0]:.3f}-"
            f"{internal_ci['auc'][1]:.3f})",
            f"{internal['slope']:.3f} ({internal_ci['slope'][0]:.3f}-"
            f"{internal_ci['slope'][1]:.3f})",
            f"{internal['oe_ratio']:.3f} ({internal_ci['oe_ratio'][0]:.3f}-"
            f"{internal_ci['oe_ratio'][1]:.3f})",
            f"{internal['brier']:.3f} ({internal_ci['brier'][0]:.3f}-"
            f"{internal_ci['brier'][1]:.3f})"],
        "external": [
            f"{external['c_index']:.3f} ({external_ci['c_index'][0]:.3f}-"
            f"{external_ci['c_index'][1]:.3f})",
            f"{external['auc']:.3f} ({external_ci['auc'][0]:.3f}-"
            f"{external_ci['auc'][1]:.3f})",
            f"{external['slope']:.3f} ({external_ci['slope'][0]:.3f}-"
            f"{external_ci['slope'][1]:.3f})",
            f"{external['oe_ratio']:.3f} ({external_ci['oe_ratio'][0]:.3f}-"
            f"{external_ci['oe_ratio'][1]:.3f})",
            f"{external['brier']:.3f} ({external_ci['brier'][0]:.3f}-"
            f"{external_ci['brier'][1]:.3f})"],
    })
    t3.to_csv(RES / "table3_performance.csv", index=False, encoding="utf-8-sig")

    # ---------------- 12. figures ----------------------------------------- #
    log("\n[12] figures")
    figure_flow(seer, ext, {"n_train": len(train), "n_valid": len(valid)})
    figure_parsimony(curve, km_bands)
    figure_calibration(dec, dca)
    figure_subgroups(sub, external["c_index"])

    # ---------------- 13. summary ----------------------------------------- #
    summary = {
        "cohorts": {
            "seer": {"n": int(len(seer)), "deaths": int(seer["event"].sum())},
            "training": {"n": int(len(train))},
            "validation": {"n": int(len(valid))},
            "external": {"n": int(len(ext)), "deaths": int(ext["event"].sum())},
        },
        "follow_up": {
            "seer_median_months": 93.0,
            "external_median_months": 107.6,
            "seer_at_risk_3y": int((seer["time_months"] >= HORIZON).sum()),
            "external_at_risk_3y": int((ext["time_months"] >= HORIZON).sum()),
        },
        "ranking": rank[["rank", "variable", "label", "importance", "sd"]]
        .to_dict("records"),
        "parsimony_curve": curve[["k", "variable", "c", "sd", "increment"]]
        .round(4).to_dict("records"),
        "retained": retained,
        "score": table_all[["variable", "category", "beta", "points"]]
        .round(4).to_dict("records"),
        "max_points": max_points,
        "cut_offs": {"low": lo_cut, "high": hi_cut},
        "internal": {**internal, "ci": internal_ci},
        "external": {**external, "ci": external_ci},
        "bands": band_rows,
        "subgroups": sub.round(4).to_dict("records"),
        "huvos": ({k: v for k, v in huv.items()
                   if not isinstance(v, (np.ndarray, bool))} if huv else None),
    }
    with open(RES / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)

    log(f"\ndone in {time.time()-t0:.1f} s")
    log(f"results written to {RES}")
    (RES / "run.log").write_text("\n".join(_LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
