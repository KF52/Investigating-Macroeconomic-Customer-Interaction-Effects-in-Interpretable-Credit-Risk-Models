"""
Synthetic data-generating process (DGP) for the interpretable UK credit-risk study.

WHY THIS FILE EXISTS
With real lending data nobody knows the true answer, so there's no way to check if
a model found a genuine macro-by-micro interaction. This file writes the answer
itself. Five interactions between the economy and borrower characteristics get
planted at known strengths, loans get generated whose defaults obey those rules,
and each interpretable model is graded against the exact truth.

TWO ARMS, ONE PANEL
mode="planted" (default) plants five delta_* interactions directly as product terms
in the hazard. This is the instrument, it proves the recovery method works and
gives a guaranteed, gradeable signal.

mode="emergent" plants no product term at all. Instead the macro economy feeds into
each borrower's affordability quarter by quarter: CPI raises expenditure, Bank Rate
reprices variable debt, unemployment triggers income shocks, GDP moves cyclical
income, and savings drain once residual income turns negative. Default is a convex
function of that squeeze, so the same five interactions emerge on their own instead
of being coded in as a rule. pipeline_utils.emergent_truth_fn measures this
numerically since there's no fixed constant to read off here.

Both arms share the same origination process, macro data, competing risks and
target construction, so the two panels stay comparable.

WHAT COMES OUT
One row per active loan per observation quarter, 2001Q2 to 2025Q4. A loan is booked
in one quarter then followed every quarter after, it can stay current, go into
arrears, default, prepay, or mature. Borrower characteristics are fixed at
origination, the eight macro series move with the real UK economy, so an
interaction can genuinely be tested as conditions change around a fixed borrower.

HOW DEFAULT IS GENERATED
The target is drawn straight from the planted hazard rather than through an
adverse-roll-then-accumulate chain. That keeps the feature macro and the macro that
generated the event in the same quarter, no timing offset, and it means the oracle
in pipeline_utils.make_truth_fn is exact: rebuild the hazard, apply the Vasicek
formula, and a calibration check against the realised outcome comes out with slope
near 1 and a tiny ECE. Arrears states are tracked for realism but kept separate
from default, a different roll populates them and never itself causes a default.

2 MODELLING CHOICES WORTH KNOWING
The main target is default_next_quarter, one quarter ahead, matching the
discrete-time hazard the DGP actually runs on. default_next_12m is also produced
as a secondary check. Both targets come from the same simulated loans so they're
directly comparable. N_ORIG_PER_QUARTER is 600 rather than 500, since a quarterly
target has roughly four times fewer positive events than a 12-month one, and the
extra volume keeps the weaker interactions estimable.

HOW THE FILE IS LAID OUT
Reads top to bottom in build order, grouped into three kinds of evidence.

A. MICRO / BORROWER DATA: who the borrowers are and what their loans look like.
Category definitions and probabilities follow FCA CONC 5.2A and Product Sales Data
rules, income is anchored to ONS ASHE, region shares to the ONS regional labour
market release, and the FCA Financial Lives distress purposes are informed by the
2024 survey (though their exact probabilities are synthetic). Zopa and Lloyds
product pages give UK product proxies. German Credit, GMSC and LendingClub are
schema references only, none of their numbers get imported. Financial helpers turn
gross income into take-home pay and price the loan. simulate_loan_statics draws one
cohort. The affordability screen only books a loan if post-loan residual income is
positive, which is what makes this a booked-loan panel rather than an application
pool. The z-score reference is fixed once on a large accepted cohort. Feature
engineering builds the affordability ratios in one place so they can't drift
between stages.

B. MACRO DATA: the real UK economy. Every series is a named, citable official
statistic, not an invented path: ONS unemployment (MGSX), ONS CPI (D7G7), Bank of
England Bank Rate (IUQABEDR), ONS GDP growth (IHYQ), ONS wage growth (A2FA), BIS
household debt-service ratio, Bank of England consumer credit growth (LPMB4TC),
and ONS saving ratio (NRJS). build_macro_frame z-scores each series against its own
2001-2025 history, builds the composite stress index M (a weighted blend of
unemployment, CPI and Bank Rate), and winsorises the most volatile series so one
extreme quarter can't dominate the hazard.

C. GROUND TRUTH: the answer the models get graded against. TRUE_PARAMS holds every
hazard coefficient, the base rate, borrower main effects, the five planted
interactions, two decoy macro main effects with no interaction attached, and the
competing-risk and Vasicek settings. These are researcher-set by design, not
estimates from real UK default data. generate_panel runs the discrete-time survival
loop: books each cohort, rolls active loans forward a quarter under that quarter's
macro row, applies the hazard and the shared Vasicek factor, resolves competing
risks, and derives both targets from the realised default timing.

A FEW SCHEMA NOTES
Cohorts are booked 2001Q2 to 2024Q4, with 2025Q1-2025Q4 as follow-up only so every
loan gets a full forward window. age_at_origination is stored for audit only,
modelling uses age_band since exact age is a protected characteristic. All money
fields are simulated in constant 2023/24 GBP, historical inflation only affects
default risk through the macro channels, not by silently revaluing borrower fields.

CALIBRATION PHILOSOPHY
FCA rules and Product Sales Data give the field definitions. ONS and BoE-NMG give
UK marginal calibration (ASHE anchors employed earnings only). Zopa and Lloyds give
product and performance proxies. German Credit, GMSC and LendingClub are schema
references only. So this data is UK regulatory- and evidence-informed, not
proprietary Lloyds data or a fully calibrated industry dataset.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit

# ----------------------------------------------------------------------------
# Location of the official CSV exports
# ----------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
def _resolve_macro_dir():
    # Locate the eight macro CSVs. Preference order: an explicit CREDIT_MACRO_DIR override, then the
    # macro_data folder bundled inside this project (the self-contained default, so the project runs
    # out of the box after unzipping), then a sibling dataset inspect folder one level up. The first
    # location that actually holds the data wins.
    env = os.environ.get("CREDIT_MACRO_DIR")
    candidates = ([env] if env else []) + [
        os.path.join(_HERE, "macro_data"),
        os.path.normpath(os.path.join(_HERE, "..", "dataset inspect")),
    ]
    for cand in candidates:
        if cand and os.path.exists(os.path.join(cand, "ONS_umemployment.csv")):
            return cand
    # Fall back to the bundled location so a missing-data error points at the expected place.
    return os.path.join(_HERE, "macro_data")
DATA_DIR = _resolve_macro_dir()

# Timeline: originate 2001Q2-2024Q4 (95 quarters), then 2025Q1-2025Q4 are
# FOLLOW-UP ONLY (no new cohorts) so that loans observed up to 2024Q4 get a
# complete forward-12m window. All 8 macro series cover this full range; wage
# growth (A2FA) begins 2001 Q1 with a single month, so origination starts 2001Q2.
# 99 quarters (95 orig + 4 follow-up)
QUARTERS = pd.period_range("2001Q2", "2025Q4", freq="Q")
N_QUARTERS = len(QUARTERS)
# last origination quarter
ORIG_END = pd.Period("2024Q4", freq="Q")
# last observation (follow-up) quarter
OBS_END = pd.Period("2025Q4", freq="Q")


# ----------------------------------------------------------------------------
# Real-data loaders (ONS / BoE / BIS)
#
# Each official CSV export has its own layout, so there is one small loader per
# source. They all end up returning a clean pandas Series indexed by quarter, so
# load_macro_series can line them up side by side. Using the real published series
# (rather than a made-up macro path) is what ties the study to genuine UK cycles.
# ----------------------------------------------------------------------------

def _read_ons_two_col(path):
    # The ONS "download as CSV" exports are messy two-column files: a text label
    # in column one (a year, a quarter, or a month) and the value in column two.
    # Read everything as strings first and skip malformed lines; the callers below
    # decide which rows they actually want.
    return pd.read_csv(path, header=None, names=["k", "v"], dtype=str,
                       on_bad_lines="skip", engine="python")


def load_ons_quarterly(path):
    # Keep only the rows whose label looks like "2008 Q3" (an ONS quarterly row),
    # convert that label to a pandas quarterly Period, and coerce the value to a
    # number. The same file also holds annual and monthly rows, which this drops.
    raw = _read_ons_two_col(path)
    q = raw[raw["k"].str.match(r"^\d{4} Q[1-4]$", na=False)].copy()
    q["quarter"] = pd.PeriodIndex(q["k"].str.replace(" ", "", regex=False), freq="Q")
    q["v"] = pd.to_numeric(q["v"], errors="coerce")
    return q.set_index("quarter")["v"].sort_index()


def load_ons_monthly_to_quarterly(path):
    # Regular pay growth is published monthly ("2008 JAN"). Keep the monthly rows,
    # then average the three months in each quarter to get a quarterly figure that
    # matches the rest of the frame.
    raw = _read_ons_two_col(path)
    m = raw[raw["k"].str.match(r"^\d{4} [A-Z]{3}$", na=False)].copy()
    m["month"] = pd.PeriodIndex(pd.to_datetime(m["k"], format="%Y %b"), freq="M")
    m["v"] = pd.to_numeric(m["v"], errors="coerce")
    mq = m.set_index("month")["v"].groupby(lambda p: p.asfreq("Q")).mean()
    mq.index = pd.PeriodIndex(mq.index, freq="Q")
    return mq.sort_index()


def load_bank_rate_quarterly(path):
    # The Bank Rate only changes on meeting dates, so the raw file is a sparse list
    # of "date, rate" step changes. Expand it to a daily series by carrying the last
    # rate forward, then average each quarter. This gives the average rate a
    # borrower actually faced in a quarter, not just the end-of-quarter level.
    br = pd.read_csv(path, dtype=str)
    br.columns = ["date", "rate"]
    br["date"] = pd.to_datetime(br["date"], format="%d %b %y")
    br["rate"] = pd.to_numeric(br["rate"], errors="coerce")
    br = br.dropna().sort_values("date").set_index("date")["rate"]
    daily = br.reindex(pd.date_range("2000-01-01", "2026-12-31", freq="D")).ffill()
    q = daily.resample("QE").mean()
    q.index = pd.PeriodIndex(q.index, freq="Q")
    return q.sort_index()


def load_bis_dsr(path):
    # BIS household debt-service-ratio export has a few header rows and long column
    # names. Skip the header rows, trim each column name at the first colon, and
    # read the standard TIME_PERIOD / OBS_VALUE columns.
    dsr = pd.read_csv(path, skiprows=3)
    dsr.columns = [c.split(":")[0] for c in dsr.columns]
    dsr["quarter"] = pd.PeriodIndex(pd.to_datetime(dsr["TIME_PERIOD"]), freq="Q")
    return dsr.set_index("quarter")["OBS_VALUE"].astype(float).sort_index()


def load_credit_growth_quarterly(path):
    # Consumer credit growth arrives as dated observations. Take the first two
    # columns (date and value), then average to quarterly like the Bank Rate.
    cc = pd.read_csv(path, dtype=str).iloc[:, :2]
    cc.columns = ["date", "v"]
    cc["date"] = pd.to_datetime(cc["date"], format="%d %b %y")
    cc["v"] = pd.to_numeric(cc["v"], errors="coerce")
    cc = cc.dropna().sort_values("date").set_index("date")["v"]
    q = cc.resample("QE").mean()
    q.index = pd.PeriodIndex(q.index, freq="Q")
    return q.sort_index()


def load_macro_series(data_dir=DATA_DIR, quarters=QUARTERS):
    # Load all eight series with the right loader for each file, then reindex them
    # onto the shared 2001Q2-2025Q4 quarterly grid so they align row for row.
    p = lambda fn: os.path.join(data_dir, fn)
    series = {
        "unemployment":  load_ons_quarterly(p("ONS_umemployment.csv")),
        "gdp_growth":    load_ons_quarterly(p("ONS_GDP.csv")),
        "cpi_inflation": load_ons_quarterly(p("ONS_CPI_inflation.csv")),
        "bank_rate":     load_bank_rate_quarterly(p("BOE_bank_rate.csv")),
        "wage_growth":   load_ons_monthly_to_quarterly(p("ONS_regular_pay_growth.csv")),
        "dsr":           load_bis_dsr(p("BIS_debt_service_ratio_households.csv")),
        "credit_growth": load_credit_growth_quarterly(p("BOE_credit_growth.csv")),
        "saving_ratio":  load_ons_quarterly(p("ONS_household_n_NPISH_saving_ratio.csv")),
    }
    m = pd.DataFrame({k: v.reindex(quarters) for k, v in series.items()})
    # A gap in any series would leave a quarter with no macro state, which would
    # silently break the hazard. Fail loudly instead so the data problem is obvious.
    if m.isna().any().any():
        missing = m.columns[m.isna().any()].tolist()
        raise ValueError(f"Missing quarterly coverage {quarters[0]}-{quarters[-1]} for: {missing}")
    m.index.name = "quarter"
    return m.reset_index()


# ----------------------------------------------------------------------------
# Macro frame: turn the raw levels into standardised drivers for the hazard
#
# The hazard is written in terms of z-scores, not raw percentages, so that the
# planted coefficients sit on a comparable scale. M_WEIGHTS builds the composite
# stress index M, and WINSOR caps the most volatile z-scores.
# ----------------------------------------------------------------------------

# Composite stress index M = 0.5*unemployment + 0.25*CPI + 0.25*Bank Rate (all as
# z-scores). Unemployment carries the most weight because it is the clearest single
# summary of a downturn for a consumer-credit book. M is the macro axis that the
# savings-buffer interaction is planted against.
M_WEIGHTS = {"unemployment": 0.5, "cpi_inflation": 0.25, "bank_rate": 0.25}
# Clip winsor threshold: GDP growth and CPI can have one or two very extreme
# quarters (for example the COVID GDP collapse). Capping their z-scores at +/- 3
# standard deviations stops a single outlier quarter from dominating the hazard.
WINSOR = 3.0


def _hist_stats(m):
    # Mean and (population) standard deviation of each raw series over the full
    # 2001-2025 history. These fixed moments are the reference used to z-score the
    # macro state, so the standardisation is stable across every cohort and scenario.
    cols = ["unemployment", "cpi_inflation", "bank_rate", "gdp_growth",
            "wage_growth", "dsr", "credit_growth", "saving_ratio"]
    return {c: (float(m[c].mean()), float(m[c].std(ddof=0))) for c in cols}


def build_macro_frame(macro_raw, hist_stats=None):
    # Add the standardised (z-scored) columns and the composite stress index M to a
    # raw macro frame. Passing hist_stats in (rather than recomputing) lets a
    # hand-authored stress scenario be z-scored against the SAME historical moments
    # as the real data, so a scenario value is comparable to history.
    m = macro_raw.copy()
    h = _hist_stats(m) if hist_stats is None else hist_stats

    def z(col):
        # Standardise one series against its historical mean and standard deviation.
        mu, sd = h[col]
        return (m[col] - mu) / sd

    # z-score each of the eight series. Short names (zU, zC, zR, ...) are reused
    # throughout the hazard, so they are defined once here.
    m["zU"] = z("unemployment")
    m["zC"] = z("cpi_inflation")
    m["zR"] = z("bank_rate")
    m["zGDP_raw"] = z("gdp_growth")
    m["zWage"] = z("wage_growth")
    m["zDSR"] = z("dsr")
    m["zCredit"] = z("credit_growth")
    m["zSav"] = z("saving_ratio")

    # Winsorised copies of the two most volatile series. The hazard uses zGDP and
    # zCw (capped) rather than the raw z-scores, so extreme quarters cannot blow up
    # the interaction terms.
    m["zGDP"] = m["zGDP_raw"].clip(-WINSOR, WINSOR)
    m["zCw"] = m["zC"].clip(-WINSOR, WINSOR)

    # The composite stress index used by the savings-buffer interaction.
    m["M"] = (M_WEIGHTS["unemployment"] * m["zU"]
              + M_WEIGHTS["cpi_inflation"] * m["zC"]
              + M_WEIGHTS["bank_rate"] * m["zR"])
    return m


# Build the historical macro frame once at import time and reuse it everywhere.
# If the CSV exports are missing (for example on a machine without the data folder)
# fall back to placeholders so the module still imports; generate_panel will then
# raise a clear error only if it is actually asked to run without a macro frame.
try:
    MACRO_RAW = load_macro_series()
    HIST_STATS = _hist_stats(MACRO_RAW)
    HIST_MACRO = build_macro_frame(MACRO_RAW, HIST_STATS)
    # used to price APR at average conditions
    _MEAN_BANK_RATE = HIST_STATS["bank_rate"][0]
except (FileNotFoundError, ValueError) as _e:
    MACRO_RAW = HIST_STATS = HIST_MACRO = None
    _MEAN_BANK_RATE = 2.0
    _LOAD_ERROR = _e


# ----------------------------------------------------------------------------
# Categorical definitions (UK-oriented)
#
# Each categorical field is defined by its list of levels plus a matching list (or
# dict) of marginal probabilities. simulate_loan_statics draws from these to build
# a plausible UK borrower mix. The numbers are UK-informed but researcher-set, so
# they are documented here rather than hidden inside the sampling code.
# ----------------------------------------------------------------------------

# Employment status and its baseline population share. EMP_BASE roughly reflects a
# UK working-age mix; simulate_loan_statics later tilts it with unemployment so the
# downturn cohorts contain more unemployed and benefit-reliant borrowers.
EMP_STATUSES = ["full_time", "part_time", "temporary", "self_employed",
                "unemployed", "benefits", "retired", "student", "other"]
EMP_BASE = {"full_time": 0.50, "part_time": 0.13, "temporary": 0.05,
            "self_employed": 0.12, "unemployed": 0.03, "benefits": 0.03,
            "retired": 0.08, "student": 0.04, "other": 0.02}

# Gross annual income by employment status: (median GBP, lognormal sigma).
# ONS ASHE anchors EMPLOYED earnings only; the rest use separate distributions.
INCOME_PARAMS = {
    # ASHE full-time median ~ GBP 35k
    "full_time":     (35000, 0.50),
    "part_time":     (16000, 0.55),
    "temporary":     (24000, 0.55),
    # dispersed, NOT ASHE
    "self_employed": (24000, 0.75),
    "unemployed":    (8500, 0.35),
    "benefits":      (10000, 0.40),
    "retired":       (19000, 0.55),
    "student":       (11000, 0.45),
    "other":         (16000, 0.60),
}

UK_REGIONS = ["NE", "NW", "YH", "EM", "WM", "EE", "LON", "SE", "SW", "SCT", "WAL", "NI"]
UK_REGION_PROBS = [0.04, 0.11, 0.08, 0.07, 0.09, 0.09, 0.13, 0.14, 0.08, 0.08, 0.05, 0.04]

HOUSING_STATUSES = ["own_outright", "mortgage", "private_rent", "social_rent", "family_other"]
HOUSING_PROBS = [0.22, 0.30, 0.26, 0.14, 0.08]
HOUSING_COST_FRAC = {"own_outright": 0.04, "mortgage": 0.16, "private_rent": 0.28,
                     "social_rent": 0.16, "family_other": 0.07}

# FCA-FLS-informed purposes added: essential_living_costs, unexpected_emergency,
# housing_payment (distress borrowing). Probabilities are synthetic, NOT copied
# from FLS (whose question allows multiple answers); the other shares are scaled
# down to make room.
LOAN_PURPOSES = ["debt_consolidation", "vehicle", "home_improvement",
                 "wedding_event", "holiday", "education", "other_unknown",
                 "essential_living_costs", "unexpected_emergency", "housing_payment"]
LOAN_PURPOSE_PROBS = [0.30, 0.18, 0.16, 0.04, 0.07, 0.04, 0.09, 0.05, 0.04, 0.03]

LOAN_TERMS = [12, 24, 36, 48, 60, 72, 84]
LOAN_TERM_PROBS = [0.08, 0.15, 0.28, 0.12, 0.22, 0.10, 0.05]

ADVERSE_RECORDS = ["none", "CCJ", "recent_default", "IVA_DRO", "bankruptcy"]
ADVERSE_PROBS = [0.85, 0.06, 0.05, 0.025, 0.015]

# 65+ split per FCA FLS
AGE_BANDS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]

# Performance-status codes used inside the simulation. The loop tracks each loan
# with one of these integer codes, and PERF_LABELS maps them to readable strings in
# the stored panel. A loan in a TERMINAL code leaves the panel (default, prepaid or
# closed); a DEFAULT code is the actual credit-loss event the targets are built on.
PERF_LABELS = {0: "Current", 1: "1-29 DPD", 2: "30-59 DPD", 3: "60-89 DPD",
               4: "90+ DPD", 5: "default", 6: "prepaid", 7: "closed"}
# loan leaves the panel
_TERMINAL_CODES = {4, 5, 6, 7}
# 90+ roll-to-default, and UTP default
_DEFAULT_CODES = {4, 5}

# 600 per quarter to offset the ~4x fewer quarterly events
N_ORIG_PER_QUARTER = 600


# ----------------------------------------------------------------------------
# Ground-truth parameters (quarterly discrete-time hazard)
#
# This dictionary is the answer key for the whole project. Every number here is a
# coefficient in the true default hazard, so when a model later recovers an
# interaction we grade it against exactly these values. The structure is:
#   * base_hazard: the intercept, set so the quarterly default rate is realistic.
#   * beta_*: borrower main effects on the z-scored risk drivers.
#   * housing_beta / purpose_beta / adverse_beta / region_beta: categorical add-ons.
#   * beta_M / beta_gdp / beta_wage / beta_saving_macro: macro main effects. Note
#     that beta_wage and beta_saving_macro are DECOYS: they move the hazard on their
#     own but have NO interaction, so the recovery test has clean negatives to reject.
#   * delta_*: the FIVE planted macro-by-micro interactions. These are the heart of
#     the experiment. Micro side is measured at origination, macro side at the
#     observation quarter, so the interaction is genuinely tested as the economy moves.
#   * season / shock / competing-risk / Vasicek blocks: the realism machinery that
#     shapes when defaults, prepayments and income shocks actually happen.
# ----------------------------------------------------------------------------

TRUE_PARAMS = {
    # direct-default hazard intercept. default_next_quarter is drawn straight from
    # expit(this hazard) via the Vasicek trigger at the observation quarter, so this
    # intercept IS the log-odds of default and the oracle that rebuilds it is exact.
    # calibrated so the quarterly default rate on the Current modelling set stays ~0.6%.
    "base_hazard":       -6.15,
    # separate intercept for the arrears-display roll (Current -> 1-29/30-59/60-89 DPD).
    # this roll is decoupled from default: it only keeps performance_status_t and
    # prior_arrears_count_12m_t realistic, it never creates a default event on its own.
    # set so a few percent of performing loan-quarters show some arrears.
    "base_arrears":      -3.30,
    # borrower risk main effects (on z-scored drivers unless noted)
    "beta_income":       -0.45,
    "beta_dti":           0.45,
    "beta_util":          0.30,
    "beta_pti":           0.20,
    # existing debt-service ratio
    "beta_dsr":           0.20,
    "beta_residual":     -0.12,
    # savings buffer months
    "beta_savings":      -0.20,
    "beta_hist":         -0.15,
    "beta_tenure":       -0.10,
    # per prior 30+ DPD in 24m (capped)
    "beta_delinq24":      0.22,
    "beta_searches":      0.06,
    # income_evidence_type == declared_only
    "beta_unverified":    0.20,
    "beta_self_emp":      0.20,
    # new static main effects (small, realistic; all mirrored in the oracle)
    # per new credit account opened in last 24m
    "beta_new_credit":    0.04,
    # a second household earner adds resilience
    "beta_other_earner": -0.12,
    # relationship/known-customer effect (Lloyds)
    "beta_existing_customer": -0.08,
    # DYNAMIC main effects (evaluated at the observation quarter)
    # currently in a job-loss / reduced-hours shock
    "beta_income_shock":  0.45,
    # per arrears quarter in the trailing 12m (0-4)
    "beta_prior_arrears": 0.25,
    "housing_beta": {"own_outright": 0.0, "mortgage": 0.05, "private_rent": 0.15,
                     "social_rent": 0.20, "family_other": 0.10},
    "purpose_beta": {"debt_consolidation": 0.15, "vehicle": -0.05,
                     "home_improvement": -0.03, "wedding_event": 0.05,
                     "holiday": 0.05, "education": 0.03, "other_unknown": 0.08,
                     "essential_living_costs": 0.20, "unexpected_emergency": 0.18,
                     # distress-borrowing purposes (FCA FLS)
                     "housing_payment": 0.22},
    "adverse_beta": {"none": 0.0, "CCJ": 0.35, "recent_default": 0.55,
                     "IVA_DRO": 0.70, "bankruptcy": 0.90},
    "region_beta": {"NE": 0.08, "NW": 0.04, "YH": 0.05, "EM": 0.02, "WM": 0.03,
                    "EE": -0.01, "LON": -0.06, "SE": -0.05, "SW": -0.02,
                    "SCT": 0.03, "WAL": 0.05, "NI": 0.04},
    # macro main effects (time-varying, at the observation quarter)
    "beta_M":             0.20,
    "beta_gdp":          -0.10,
    # A2FA real pay growth (DECOY: main effect only)
    "beta_wage":         -0.08,
    # NRJS saving ratio (DECOY: main effect only)
    "beta_saving_macro": -0.06,
    # FIVE ground-truth interactions (micro at origination x macro at observation)
    # 1 income x unemployment (MGSX)
    "delta_income_unemp": -0.35,
    # 2 DTI x Bank Rate (IUQABEDR)
    "delta_dti_rate":      0.35,
    # 3 income x CPI (D7G7)
    "delta_income_cpi":   -0.20,
    # 4 self-employed x GDP (IHYQ)
    "delta_selfemp_gdp":  -0.30,
    # 5 savings buffer x macro stress M
    "delta_savings_stress": -0.20,
    # lifecycle shape
    # seasoning hump on months-on-book (peak ~ +0.35 logit near mob 8)
    "beta_season":        0.12,
    # quarters to peak-ish
    "season_scale":       8.0,
    # already-delinquent -> higher hazard (sticky but curable)
    "beta_inarrears":     0.80,
    # income-shock dynamics (employed borrowers only)
    # per-quarter shock hazard at average unemployment
    "shock_base_prob":    0.010,
    # multiplies base by (1 + sens*max(zU,0))
    "shock_unemp_sens":   1.50,
    # per-quarter chance a shocked borrower recovers
    "shock_recover_prob": 0.30,
    # of shocks that are job loss (vs reduced hours)
    "shock_jobloss_frac": 0.45,
    # net income multiplier under job loss
    "shock_mult_jobloss": 0.45,
    # net income multiplier under reduced hours
    "shock_mult_reduced": 0.75,
    # competing risks
    # quarterly prepayment hazard intercept
    "prepay_base":       -3.10,
    # per quarter on book
    "prepay_season":      0.04,
    # lower-risk loans prepay a little more
    "prepay_riskaverse":  0.30,
    # unlikeliness-to-pay (direct default) intercept
    "utp_base":          -6.20,
    # UTP loading per adverse-record severity step
    "utp_adverse":        0.60,
    # Vasicek systematic factor
    "rho_asset":          0.10,
    "theta_macro":        0.50,
    "sigma_eta":          0.45,
    # structural composition sensitivities
    # household DSR scales borrower unsecured balance
    "kappa_dsr_dti":      0.06,
    # consumer-credit growth scales utilisation
    "kappa_credit_util":  0.10,
}


# ----------------------------------------------------------------------------
# EMERGENT-MODE parameters and the structural macro->micro chain
#
# These are used ONLY when generate_panel(mode="emergent"). They are NOT planted
# interaction coefficients: there is no product term here. They are the structural
# sensitivities of the affordability accounting to the macro economy, taken in
# DIRECTION from documented links (CPI raises living costs; Bank Rate reprices
# variable-rate debt; GDP moves income, self-employed most). The macro-by-micro
# INTERACTION is not written down anywhere; it emerges because residual income is a
# function of both the borrower's micro level and the macro state, and the hazard is
# convex in residual income. The emergent "truth" is therefore MEASURED from the
# hazard surface (pipeline_utils.emergent_truth_fn), not read off these numbers.
# ----------------------------------------------------------------------------

EMERGENT_PARAMS = {
    # Intercept for the emergent hazard. Tuned so the emergent quarterly default rate
    # on the modelling set is ~0.6%, matching the planted panel (see Stage 1 smoke
    # check; re-anchor here if the rate drifts).
    "base_hazard_e":      -8.60,
    # --- macro -> micro structural sensitivities (the ONLY place macro meets micro) ---
    # These enter the affordability accounting DETERMINISTICALLY (as expectations),
    # so the one-quarter emergent hazard is a pure function of (origination features,
    # macro state). That is what lets pipeline_utils.emergent_truth_fn MEASURE the
    # emergent interaction by gridding the same surface the generator used. No product
    # (interaction) term appears anywhere; the interaction emerges from the convexity.
    # Expected income drag from higher unemployment, as a fraction of a reference
    # monthly income per unit of relu(zU). Being an ABSOLUTE slice (not proportional to
    # the borrower's own income) it is a larger SHARE of a low income, so the downturn
    # bites lower earners harder (they also face higher job-loss risk). This is the
    # income x unemployment channel.
    "unemp_income_drag":   0.14,
    # reference monthly net income (~GBP 35k gross) that the unemployment drag scales.
    "ref_net_monthly":     2350.0,
    # CPI z-score -> multiplier on essential monthly expenditure (higher prices bite
    # hardest on low-income, high-spend borrowers).
    "cpi_exp_sens":        0.18,
    # Bank Rate z-score -> multiplier on EXISTING (variable-rate) monthly debt
    # service. Personal-loan payments are fixed; revolving/other credit reprices, so
    # a high-DTI borrower faces a bigger absolute payment rise when rates climb.
    "rate_passthrough":    0.30,
    # GDP z-score -> income multiplier, self-employed (strongly cyclical earnings).
    "gdp_income_sens_se":  0.25,
    # GDP z-score -> income multiplier, employed (mildly cyclical).
    "gdp_income_sens_emp": 0.05,
    # Stress drains the savings buffer: effective buffer z falls with relu(M). A
    # drained buffer stops protecting, so buffer and stress interact via the convex
    # buffer term (emergent, not planted).
    "buffer_drain":        0.90,
    # --- hazard response to the (macro-adjusted) affordability state ---
    # linear residual-income effect (z-scored): more slack, lower hazard.
    "beta_resid_lin":     -0.45,
    # CONVEX squeeze: extra hazard as residual income turns negative. This convexity
    # is what turns co-movement of a micro driver and a macro driver into a genuine
    # interaction (non-zero cross-partial), with no product term planted.
    "beta_resid_conv":     0.85,
    # linear (macro-adjusted) savings-buffer effect (z-scored).
    "beta_savbuf_lin":    -0.20,
    # convex buffer squeeze: extra hazard once the effective buffer runs thin.
    "beta_savbuf_conv":    0.55,
}


def emergent_affordability(net_income, self_emp, housing, essential_base,
                           existing_base, sched, savings, zU, zCw, zR, zGDP, M,
                           ep=None, ps=None):
    """Structural macro->micro accounting chain for the emergent DGP mode.

    Given a borrower's affordability money components (in raw GBP), origination liquid
    savings, and the macro state (unemployment, CPI, Bank Rate, GDP z-scores and the
    composite stress index M), return the macro-adjusted monthly residual income,
    total outgoings, the z-scored residual income, and the EFFECTIVE z-scored savings
    buffer. This is the ONLY place the macro economy enters the micro state, and it
    does so DETERMINISTICALLY (as expectations):
      * higher unemployment drags expected income down (job-loss risk), relu(zU);
      * GDP moves income, self-employed most;
      * CPI lifts essential expenditure;
      * Bank Rate reprices the variable-rate debt service;
      * stress (M) drains the protective savings buffer.
    No interaction product is formed here. Because the resulting residual income and
    effective buffer both depend on the borrower's micro level AND the macro state,
    and the hazard (emergent_logit) is CONVEX in them, the macro-by-micro interaction
    emerges. Being a pure, deterministic function of its inputs, the same chain is
    used by the generator AND by pipeline_utils.emergent_truth_fn to MEASURE the
    emergent interaction on a grid."""
    ep = EMERGENT_PARAMS if ep is None else ep
    ps = POP_STATS if ps is None else ps
    net_income = np.asarray(net_income, float)
    reluU = np.maximum(zU, 0.0)
    # income cyclicality (GDP, self-employed most) and expected job-loss drag (zU)
    gdp_sens = np.where(np.asarray(self_emp) > 0,
                        ep["gdp_income_sens_se"], ep["gdp_income_sens_emp"])
    income_eff = (net_income * (1.0 + gdp_sens * zGDP)
                  - ep["unemp_income_drag"] * reluU * ep["ref_net_monthly"])
    # essential spend rises with CPI; variable-rate debt service rises with Bank Rate
    expend = np.asarray(essential_base, float) * (1.0 + ep["cpi_exp_sens"] * zCw)
    existing = np.asarray(existing_base, float) * (1.0 + ep["rate_passthrough"] * zR)
    housing = np.asarray(housing, float); sched = np.asarray(sched, float)
    resid = income_eff - housing - expend - existing - sched
    outgo = housing + expend + existing + sched
    # RELATIVE residual (share of net income): a macro squeeze then bites hardest
    # where the affordability share is already tight (lower earners), giving the
    # emergent interactions their economically expected direction. Denominator is the
    # ORIGINATION net income, so it is fixed across the macro perturbation.
    resid_frac = resid / np.maximum(net_income, 1.0)
    z_resid = np.clip((resid_frac - ps["residfrac_mu"]) / ps["residfrac_sd"], -4, 4)
    savbuf = np.asarray(savings, float) / np.maximum(outgo, 1.0)
    z_savbuf = (savbuf - ps["savbuf_mu"]) / ps["savbuf_sd"]
    # stress drains the buffer: the EFFECTIVE buffer falls with relu(M)
    z_savbuf_eff = z_savbuf - ep["buffer_drain"] * np.maximum(M, 0.0)
    return resid, outgo, z_resid, z_savbuf_eff


def emergent_logit(lin_nonaff, macro_main, z_resid, z_savbuf, season, inarr,
                   shk_now, prior_cnt, ep=None, p=None):
    """The emergent one-quarter default log-odds.

    The affordability signal enters ONLY through the dynamic z_resid and z_savbuf,
    each with a CONVEX squeeze term that steepens as residual income / buffer turn
    negative. That convexity is exactly what makes the macro effect an INTERACTION:
    a given macro-driven fall in residual income raises the hazard more for a
    borrower who already sits low (set by their micro level), so the cross-partial
    of the hazard with respect to (micro driver, macro driver) is non-zero even
    though no product coefficient was written. `lin_nonaff` carries the borrower
    risk factors that do NOT run through affordability (credit history, delinquency,
    utilisation, categoricals), and `macro_main` the direct macro main effects."""
    ep = EMERGENT_PARAMS if ep is None else ep
    p = TRUE_PARAMS if p is None else p
    # softplus(-z_resid): ~0 when residual income is comfortably positive, grows as it
    # goes negative, so the hazard bends upward in a squeeze (the convex channel).
    conv_r = np.log1p(np.exp(np.clip(-np.asarray(z_resid, float), -30, 30)))
    conv_s = np.maximum(-np.asarray(z_savbuf, float), 0.0)
    return (ep["base_hazard_e"] + lin_nonaff + macro_main
            + ep["beta_resid_lin"] * z_resid + ep["beta_resid_conv"] * conv_r
            + ep["beta_savbuf_lin"] * z_savbuf + ep["beta_savbuf_conv"] * conv_s
            + season + inarr
            + p["beta_income_shock"] * shk_now
            + p["beta_prior_arrears"] * prior_cnt)


# When a loan has a bad quarter its days-past-due jumps by one of these step sizes.
# Mixing small and large steps means some loans creep into arrears while others
# lurch straight to 90+ days, which is closer to real arrears behaviour than a
# single fixed step. _DPD_STEP_P are the probabilities of each step.
_DPD_STEPS = np.array([25, 40, 70, 100])
_DPD_STEP_P = np.array([0.30, 0.30, 0.20, 0.20])


# ----------------------------------------------------------------------------
# Helpers: UK net income, risk-based APR, amortisation
#
# These keep the money side of each loan internally consistent, so income, tax,
# payment size and outstanding balance all agree with each other.
# ----------------------------------------------------------------------------

def uk_net_annual_income(gross):
    """Turn gross annual income into take-home pay using simplified 2023/24 UK
    income tax and employee National Insurance. The bands are the real 2023/24
    thresholds (12,570 personal allowance, 50,270 basic-rate limit). This is a
    simplification, not a full tax calculator, but it gives a realistic net-income
    distribution which the affordability ratios then build on."""
    g = np.asarray(gross, dtype=float)
    pa = 12_570.0
    # Income tax by band: 20% basic, 40% higher, 45% additional.
    band_basic = np.maximum(np.clip(g, 0, 50_270) - pa, 0)
    band_higher = np.maximum(np.clip(g, 50_270, 125_140) - 50_270, 0)
    band_add = np.maximum(g - 125_140, 0)
    income_tax = band_basic * 0.20 + band_higher * 0.40 + band_add * 0.45
    # Employee NI: 10% on earnings between the allowance and the basic-rate limit,
    # 2% above it (a simplified two-rate version of the real schedule).
    ni = (np.maximum(np.clip(g, 0, 50_270) - pa, 0) * 0.10
          + np.maximum(g - 50_270, 0) * 0.02)
    return g - income_tax - ni


def price_apr(bank_rate_level, util, delinq24, adverse_sev, searches, hist_months, rng, n, p):
    """Set a risk-based APR (in %). The APR starts from the Bank Rate plus a margin,
    then adds a loading for each risk signal (high utilisation, past delinquency,
    adverse records, many recent searches, a thin credit file), plus a little noise.
    APR is an endogenous, post-offer variable, so downstream stages can test models
    with and without it. Result is clipped to a realistic 3.9% to 39.9% range."""
    base = bank_rate_level + 3.5
    load = (6.0 * np.nan_to_num(util) + 0.6 * delinq24 + 1.2 * adverse_sev
            + 0.15 * searches + 4.0 * (hist_months < 24))
    apr = base + load + rng.normal(0, 1.2, n)
    return np.clip(apr, 3.9, 39.9)


def amortised_monthly_payment(principal, apr_pct, term_months):
    """Standard amortising loan payment: the fixed monthly amount that repays the
    principal plus interest over the term. r is the monthly rate (APR / 12 / 100).
    The r > 0 branch is the usual annuity formula; the fallback handles a zero rate."""
    r = apr_pct / 1200.0
    principal = np.asarray(principal, float)
    term_months = np.asarray(term_months, float)
    with np.errstate(over="ignore", invalid="ignore"):
        factor = np.where(r > 0, r / (1 - (1 + r) ** (-term_months)), 1.0 / term_months)
    return principal * factor


def outstanding_balance(principal, apr_pct, term_months, months_elapsed):
    """Remaining principal after `months_elapsed` scheduled payments. This is the
    standard amortisation balance: grow the principal at the monthly rate and
    subtract the future value of the payments made so far. Used to record a
    shrinking loan balance as each loan ages through the panel."""
    P = np.asarray(principal, float)
    r = np.asarray(apr_pct, float) / 1200.0
    n = np.asarray(term_months, float)
    # cannot pay past the term
    k = np.minimum(np.asarray(months_elapsed, float), n)
    pmt = amortised_monthly_payment(P, apr_pct, term_months)
    with np.errstate(over="ignore", invalid="ignore"):
        fv = np.where(r > 0, (1 + r) ** k, 1.0)
        bal = np.where(r > 0, P * fv - pmt * (fv - 1) / np.where(r > 0, r, 1.0),
                       P * (1 - k / n))
    # never below zero or above the original principal
    return np.clip(bal, 0.0, P)


def _age_to_band(age):
    # Bucket exact age into the seven reporting bands. Modelling uses the band, not
    # the raw age, because exact age is a protected characteristic.
    bins = [17, 24, 34, 44, 54, 64, 74, 200]
    return pd.cut(age, bins=bins, labels=AGE_BANDS).astype(str)


# ----------------------------------------------------------------------------
# Origination: static loan + borrower attributes
# ----------------------------------------------------------------------------

def simulate_loan_statics(n, rng, bank_rate_level=None, zU=0.0, zDSR=0.0,
                          zCredit=0.0, params=None):
    """Draw n loans' static (origination-time) attributes. Employment
    composition tracks zU; the household DSR and consumer-credit growth reshape
    the unsecured-balance and utilisation distributions (structural composition
    channels). Returns a DataFrame of the stored static columns plus a few
    internal helper columns (age, self_emp flag, scheduled payment)."""
    p = TRUE_PARAMS if params is None else params
    if bank_rate_level is None:
        bank_rate_level = _MEAN_BANK_RATE

    # employment composition (cyclical: unemployment/benefits rise with zU)
    probs = np.array([EMP_BASE[s] for s in EMP_STATUSES], dtype=float)
    cyc = probs.copy()
    for i, s in enumerate(EMP_STATUSES):
        if s in ("unemployed", "benefits"):
            cyc[i] *= (1 + 0.7 * zU)
    cyc = np.clip(cyc, 1e-6, None); cyc /= cyc.sum()
    emp = rng.choice(EMP_STATUSES, p=cyc, size=n)
    self_emp = (emp == "self_employed").astype(float)

    # age depends on status (students young, retired old)
    age = rng.normal(42, 13, n)
    age = np.where(emp == "retired", rng.normal(70, 6, n), age)
    age = np.where(emp == "student", rng.normal(22, 3, n), age)
    age = np.clip(age, 18, 85).round(0)

    # income by employment status (ASHE anchors employed only)
    med = np.array([INCOME_PARAMS[s][0] for s in EMP_STATUSES])
    sig = np.array([INCOME_PARAMS[s][1] for s in EMP_STATUSES])
    idx = np.array([EMP_STATUSES.index(s) for s in emp])
    gross = rng.lognormal(mean=np.log(med[idx]), sigma=sig[idx])
    gross_annual_income = np.clip(gross, 5_000, 400_000).round(0)
    net_monthly_income = (uk_net_annual_income(gross_annual_income) / 12.0).round(0)
    gross_monthly = gross_annual_income / 12.0

    region = rng.choice(UK_REGIONS, p=UK_REGION_PROBS, size=n)

    # housing (nudged: retired -> own_outright, student -> family_other)
    housing = rng.choice(HOUSING_STATUSES, p=HOUSING_PROBS, size=n)
    housing = np.where((emp == "retired") & (rng.random(n) < 0.5), "own_outright", housing)
    housing = np.where((emp == "student") & (rng.random(n) < 0.5), "family_other", housing)
    hfrac = np.array([HOUSING_COST_FRAC[h] for h in housing])
    monthly_housing_cost = np.maximum(gross_monthly * hfrac * (0.8 + 0.4 * rng.random(n)),
                                      100.0).round(0)

    financial_dependants = np.minimum(rng.poisson(0.7, n), 6)
    essential_monthly_expenditure = ((600.0 + 300.0 * financial_dependants)
                                     * (gross_annual_income / 35_000.0) ** 0.15
                                     * (0.85 + 0.30 * rng.random(n))).round(0)

    # existing unsecured debt (stock + flow); household DSR reshapes the stock
    dti_ratio = rng.beta(2.0, 6.0, n) * 1.2
    dti_ratio = np.clip(dti_ratio * (1 + p["kappa_dsr_dti"] * zDSR), 0.0, 1.5)
    remaining_unsecured_balance = (dti_ratio * gross_annual_income).round(0)
    service_rate = np.clip(rng.normal(0.035, 0.010, n), 0.010, 0.080)
    existing_monthly_credit_payments = (remaining_unsecured_balance * service_rate).round(0)

    # liquid savings, expressed to give a plausible buffer-months distribution
    buffer_target = np.clip(rng.lognormal(np.log(2.0), 0.9, n)
                            * (gross_annual_income / 35_000.0) ** 0.25, 0.0, 24.0)
    liquid_savings = (buffer_target * (monthly_housing_cost
                      + essential_monthly_expenditure
                      + existing_monthly_credit_payments)).round(0)

    # credit-file
    age_months_adult = np.maximum((age - 18) * 12.0, 6.0)
    credit_history_months = np.minimum(rng.gamma(2.5, 36.0, n), age_months_adult).round(0)
    tenure_cap = np.where(np.isin(emp, ["unemployed", "student", "retired"]),
                          age_months_adult * 0.4, age_months_adult)
    employment_tenure_months = np.minimum(rng.gamma(1.8, 30.0, n), tenure_cap).round(0)

    revolving_utilisation = rng.beta(1.5, 3.5, n) * 1.3
    revolving_utilisation = np.clip(revolving_utilisation * (1 + p["kappa_credit_util"] * zCredit),
                                    0.0, 1.5).round(3)
    no_facility = rng.random(n) < 0.10
    revolving_utilisation = np.where(no_facility, np.nan, revolving_utilisation)

    delinquency_count_24m = np.minimum(rng.poisson(0.30, n), 8)
    recency = np.clip(rng.gamma(2.0, 5.0, n), 1.0, 24.0).round(0)
    months_since_last_delinquency = np.where(delinquency_count_24m > 0, recency, np.nan)

    # public adverse record (worse if prior delinquencies)
    adv = rng.choice(ADVERSE_RECORDS, p=ADVERSE_PROBS, size=n)
    bump = (delinquency_count_24m > 0) & (adv == "none") & (rng.random(n) < 0.25)
    adv = np.where(bump, "CCJ", adv)
    adverse_sev = np.array([ADVERSE_RECORDS.index(a) for a in adv], dtype=float)

    recent_credit_searches_6m = np.minimum(rng.poisson(1.2, n), 15)

    # loan terms
    lti = np.clip(rng.normal(0.25, 0.12, n), 0.02, 0.80)
    loan_amount = np.clip((lti * gross_annual_income), 1_000, 40_000).round(0)
    loan_term_months = rng.choice(LOAN_TERMS, p=LOAN_TERM_PROBS, size=n)
    apr_at_origination = price_apr(bank_rate_level, revolving_utilisation,
                                   delinquency_count_24m, adverse_sev,
                                   recent_credit_searches_6m, credit_history_months,
                                   rng, n, p).round(2)
    loan_purpose = rng.choice(LOAN_PURPOSES, p=LOAN_PURPOSE_PROBS, size=n)
    scheduled_monthly_payment = amortised_monthly_payment(
        loan_amount, apr_at_origination, loan_term_months).round(0)

    # income evidence type (merges old income_source + income_verified)
    def _evidence(e, r):
        if e in ("full_time", "part_time", "temporary"):
            return np.select([r < 0.80, r < 0.92], ["payslip_paye", "bank_transactions"], "declared_only")
        if e == "self_employed":
            return np.select([r < 0.55, r < 0.85], ["tax_return", "bank_transactions"], "declared_only")
        if e == "retired":
            return np.where(r < 0.85, "pension_evidence", "declared_only")
        if e in ("unemployed", "benefits"):
            return np.where(r < 0.80, "benefits_evidence", "declared_only")
        return np.where(r < 0.50, "bank_transactions", "declared_only")
    ru = rng.random(n)
    income_evidence_type = np.array([_evidence(emp[i], ru[i]) for i in range(n)], dtype=object)

    # --- static (origination) fields ---
    open_credit_accounts = np.clip(rng.poisson(3.0, n) + 1
                                   + (credit_history_months / 60.0).astype(int), 1, 25)
    new_credit_accounts_24m = np.minimum(rng.poisson(0.6 + 0.15 * recent_credit_searches_6m), 12)
    # worst prior delinquency band -- EDA / SCHEMA-COMPLETENESS ONLY, not a model feature.
    # It is an almost-deterministic rebinning of delinquency_count_24m (near-perfect
    # information overlap), so models use the richer count instead to avoid redundancy.
    _lvl = np.clip(np.minimum(delinquency_count_24m, 4)
                   + ((adverse_sev >= 2) & (rng.random(n) < 0.5)).astype(int), 0, 4).astype(int)
    _band_map = np.array(["none", "mild_1-29", "moderate_30-59", "severe_60-89", "default_90+"])
    worst_prior_delinquency_band_24m = _band_map[_lvl]
    other_household_earner_flag = (rng.random(n) < 0.45).astype(int)
    # revolving credit limit -- SCHEMA-COMPLETENESS ONLY, a rough AGGREGATE PROXY, not a
    # model feature: it divides TOTAL unsecured balance (not just the revolving balance)
    # by revolving utilisation, so its economic meaning is weak. NaN where no facility.
    with np.errstate(invalid="ignore"):
        revolving_credit_limit = np.where(
            np.isnan(revolving_utilisation), np.nan,
            np.round(remaining_unsecured_balance / np.clip(revolving_utilisation, 0.02, 1.5), 0))
    revolving_credit_limit = np.where(np.isnan(revolving_credit_limit), np.nan,
                                      np.clip(revolving_credit_limit, 500, 100_000))

    """ Justification for existing_customer_flag :
    Existing-customer relationship indicator at loan origination. Inclusion is supported by 
    retail relationship-banking research showing that prior bank relationships provide additional 
    screening and monitoring information and are associated with lower subsequent default risk 
    (Puri, Rocholl & Steffen, 2017; Agarwal et al., 2018).

    FCA Financial Lives 2024, CC16 reports that 35% of recent personal-loan holders who shopped around 
    chose their provider partly because they were a previous/existing customer. This is conditional 
    choice evidence, not the prevalence of existing customers across all booked personal loans.

    Therefore p=0.35 and beta_existing_customer=-0.08 are transparent synthetic DGP assumptions, 
    not UK/Lloyds-calibrated estimates, and should be tested through sensitivity analysis.
    """
    existing_customer_flag = (rng.random(n) < 0.35).astype(int)

    return pd.DataFrame({
        # internal helper (drives age_band/caps; dropped)
        "age": age,
        # STORED for schema completeness / auditing only
        "age_at_origination": age.astype(int),
        # modelling uses the band, not exact age
        "age_band": _age_to_band(age),
        "uk_region": region,
        "employment_status_at_origination": emp,
        "employment_tenure_months": employment_tenure_months,
        "financial_dependants": financial_dependants,
        "income_evidence_type": income_evidence_type,
        "gross_annual_income": gross_annual_income,
        "net_monthly_income": net_monthly_income,
        "housing_status": housing,
        "monthly_housing_cost": monthly_housing_cost,
        "essential_monthly_expenditure": essential_monthly_expenditure,
        "remaining_unsecured_balance": remaining_unsecured_balance,
        "existing_monthly_credit_payments": existing_monthly_credit_payments,
        "liquid_savings": liquid_savings,
        "credit_history_months": credit_history_months,
        "revolving_utilisation": revolving_utilisation,
        "delinquency_count_24m": delinquency_count_24m,
        "months_since_last_delinquency": months_since_last_delinquency,
        "public_adverse_record": adv,
        "recent_credit_searches_6m": recent_credit_searches_6m,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "apr_at_origination": apr_at_origination,
        "loan_purpose": loan_purpose,
        # static fields
        "open_credit_accounts": open_credit_accounts,
        "new_credit_accounts_24m": new_credit_accounts_24m,
        "worst_prior_delinquency_band_24m": worst_prior_delinquency_band_24m,
        "other_household_earner_flag": other_household_earner_flag,
        "revolving_credit_limit": revolving_credit_limit,
        "existing_customer_flag": existing_customer_flag,
        # internal helpers (not stored in the panel)
        "_self_emp": self_emp,
        "_scheduled_payment": scheduled_monthly_payment,
        "_adverse_sev": adverse_sev,
    })


# ----------------------------------------------------------------------------
# Affordability / approval rule: the panel is a BOOKED-loan portfolio, so
# applicants must pass a transparent affordability screen at origination before
# they become loans. The rule is post-loan RESIDUAL INCOME > 0:
#   net monthly income
#     - monthly housing cost - essential expenditure
#     - existing credit payments - new scheduled loan payment  > 0
# FCA CONC 5.2A is principles-based and prescribes no single numeric threshold,
# so "residual income > 0" is a transparent SYNTHETIC research assumption, not an
# industry-calibrated cutoff. Rejected applicants never receive a loan_id, a
# balance, a performance history or a default_next_12m outcome.
# ----------------------------------------------------------------------------

def _residual_income(df):
    """Post-loan monthly residual income at origination (Series-aligned to df)."""
    return (df["net_monthly_income"].to_numpy(float)
            - df["monthly_housing_cost"].to_numpy(float)
            - df["essential_monthly_expenditure"].to_numpy(float)
            - df["existing_monthly_credit_payments"].to_numpy(float)
            - df["_scheduled_payment"].to_numpy(float))


def affordable_mask(df):
    """Boolean mask of applicants passing the affordability rule (residual > 0)."""
    return _residual_income(df) > 0.0


def simulate_accepted_cohort(n_accept, rng, affordability=True, oversample=1.8, **kwargs):
    """Draw candidate applications and keep the first `n_accept` that PASS the
    affordability rule, so every origination quarter books exactly `n_accept`
    loans. Oversamples (default 1.8x) and tops up until enough candidates pass;
    rejected candidates are discarded and never enter the panel. With
    affordability=False this is identical to simulate_loan_statics(n_accept)."""
    if not affordability:
        # Screen switched off: behave exactly like a plain draw (used for comparison).
        return simulate_loan_statics(n_accept, rng, **kwargs)
    kept, n_have = [], 0
    # Draw a bit more than needed up front, because some applicants will be rejected.
    batch = max(int(n_accept * oversample), n_accept)
    while n_have < n_accept:
        cand = simulate_loan_statics(batch, rng, **kwargs)
        # keep only those who pass the screen
        ok = cand[affordable_mask(cand)]
        kept.append(ok); n_have += len(ok)
        # If still short, top up with another batch sized to the remaining shortfall.
        batch = max(int((n_accept - n_have) * (oversample + 0.4)) + 50, 100)
    # Trim to exactly n_accept so every quarter books the same number of loans.
    return pd.concat(kept, ignore_index=True).iloc[:n_accept].reset_index(drop=True)


def reference_pop_stats():
    """Means/SDs for z-scoring the hazard's risk drivers, from a 200k reference
    cohort of ACCEPTED (affordability-passing) applicants at average conditions
    (seed 7, mean Bank Rate, zU=zDSR=zCredit=0), so the z-score reference matches
    the booked population that is actually modelled."""
    ref = simulate_accepted_cohort(200_000, np.random.default_rng(7),
                                   bank_rate_level=_MEAN_BANK_RATE)
    net = ref["net_monthly_income"].to_numpy()
    dti = ref["remaining_unsecured_balance"] / ref["gross_annual_income"]
    pti = ref["_scheduled_payment"] / np.maximum(net, 1.0)
    dsr = ref["existing_monthly_credit_payments"] / np.maximum(net, 1.0)
    # loan-to-income (needed as a candidate micro axis for the randomised planted grid)
    lti = ref["loan_amount"] / ref["gross_annual_income"]
    resid = (net - ref["monthly_housing_cost"] - ref["essential_monthly_expenditure"]
             - ref["existing_monthly_credit_payments"] - ref["_scheduled_payment"])
    outgo = (ref["monthly_housing_cost"] + ref["essential_monthly_expenditure"]
             + ref["existing_monthly_credit_payments"] + ref["_scheduled_payment"])
    # emergent channel: residual income as a SHARE of net income. Using a relative
    # measure (not absolute GBP) is what makes a macro squeeze bite hardest where the
    # affordability share is already tight, i.e. lower-income borrowers, so the
    # emergent macro-by-micro interactions carry the economically expected direction.
    residfrac = resid / np.maximum(net, 1.0)
    savbuf = ref["liquid_savings"] / np.maximum(outgo, 1.0)
    util = ref["revolving_utilisation"].fillna(ref["revolving_utilisation"].mean())

    def ms(x):
        x = np.asarray(x, float)
        return float(np.nanmean(x)), float(np.nanstd(x))

    return {
        "log_inc_mu": float(np.log(ref["gross_annual_income"]).mean()),
        "log_inc_sd": float(np.log(ref["gross_annual_income"]).std()),
        "dti_mu": ms(dti)[0], "dti_sd": ms(dti)[1],
        "lti_mu": ms(lti)[0], "lti_sd": ms(lti)[1],
        "pti_mu": ms(pti)[0], "pti_sd": ms(pti)[1],
        "dsr_mu": ms(dsr)[0], "dsr_sd": ms(dsr)[1],
        "resid_mu": ms(resid)[0], "resid_sd": ms(resid)[1],
        "residfrac_mu": ms(residfrac)[0], "residfrac_sd": ms(residfrac)[1],
        "savbuf_mu": ms(savbuf)[0], "savbuf_sd": ms(savbuf)[1],
        "util_mu": ms(util)[0], "util_sd": ms(util)[1],
        "hist_mu": ms(ref["credit_history_months"])[0], "hist_sd": ms(ref["credit_history_months"])[1],
        "tenure_mu": ms(ref["employment_tenure_months"])[0], "tenure_sd": ms(ref["employment_tenure_months"])[1],
    }


# Compute the z-score reference once at import time. Every hazard evaluation reuses
# it, so the standardisation never shifts between cohorts, quarters or scenarios.
POP_STATS = reference_pop_stats()


def _static_hazard_components(df, pop_stats=None, params=None):
    """Precompute, per loan, the time-invariant part of the hazard logit plus the
    per-loan sensitivities to each macro factor (so the quarter loop is a cheap
    vectorised add). Returns a dict of numpy arrays.

    The idea is to split the hazard into two pieces. `lin_static` is everything that
    does not change from quarter to quarter (the borrower main effects, categoricals
    and base rate). The `coef_*` arrays are each loan's SLOPE against a macro factor,
    which is exactly the planted interaction: for example coef_zU is
    delta_income_unemp * z_income, so multiplying it by that quarter's unemployment
    z-score gives the income-by-unemployment contribution. Precomputing these means
    the per-quarter loop only has to do a handful of multiply-adds per loan."""
    p = TRUE_PARAMS if params is None else params
    ps = POP_STATS if pop_stats is None else pop_stats
    # avoid divide-by-zero
    net = np.maximum(df["net_monthly_income"].to_numpy(float), 1.0)

    # Standardise each risk driver onto the reference z-scale. These match the ratio
    # definitions in feature_engineering, so the oracle and the models see the same
    # quantities. resid is clipped to keep a few extreme applicants from dominating.
    z_inc = (np.log(df["gross_annual_income"].to_numpy(float)) - ps["log_inc_mu"]) / ps["log_inc_sd"]
    dti = df["remaining_unsecured_balance"].to_numpy(float) / df["gross_annual_income"].to_numpy(float)
    z_dti = (dti - ps["dti_mu"]) / ps["dti_sd"]
    pti = df["_scheduled_payment"].to_numpy(float) / net
    z_pti = (pti - ps["pti_mu"]) / ps["pti_sd"]
    dsr = df["existing_monthly_credit_payments"].to_numpy(float) / net
    z_dsr = (dsr - ps["dsr_mu"]) / ps["dsr_sd"]
    resid = (net - df["monthly_housing_cost"].to_numpy(float)
             - df["essential_monthly_expenditure"].to_numpy(float)
             - df["existing_monthly_credit_payments"].to_numpy(float)
             - df["_scheduled_payment"].to_numpy(float))
    z_resid = np.clip((resid - ps["resid_mu"]) / ps["resid_sd"], -4, 4)
    outgo = (df["monthly_housing_cost"].to_numpy(float)
             + df["essential_monthly_expenditure"].to_numpy(float)
             + df["existing_monthly_credit_payments"].to_numpy(float)
             + df["_scheduled_payment"].to_numpy(float))
    savbuf = df["liquid_savings"].to_numpy(float) / np.maximum(outgo, 1.0)
    z_sav = (savbuf - ps["savbuf_mu"]) / ps["savbuf_sd"]
    util = df["revolving_utilisation"].to_numpy(float)
    util = np.where(np.isnan(util), ps["util_mu"], util)
    z_util = (util - ps["util_mu"]) / ps["util_sd"]
    z_hist = (df["credit_history_months"].to_numpy(float) - ps["hist_mu"]) / ps["hist_sd"]
    z_tenure = (df["employment_tenure_months"].to_numpy(float) - ps["tenure_mu"]) / ps["tenure_sd"]
    # loan-to-income, a candidate micro axis for the randomised planted grid
    lti = df["loan_amount"].to_numpy(float) / df["gross_annual_income"].to_numpy(float)
    z_lti = (lti - ps.get("lti_mu", 0.0)) / ps.get("lti_sd", 1.0)

    self_emp = df["_self_emp"].to_numpy(float)
    unverified = (df["income_evidence_type"].to_numpy() == "declared_only").astype(float)
    housing_adj = df["housing_status"].map(p["housing_beta"]).to_numpy(float)
    purpose_adj = df["loan_purpose"].map(p["purpose_beta"]).to_numpy(float)
    adverse_adj = df["public_adverse_record"].map(p["adverse_beta"]).to_numpy(float)
    region_adj = df["uk_region"].map(p["region_beta"]).to_numpy(float)

    lin_static = (p["base_hazard"]
                  + p["beta_income"] * z_inc
                  + p["beta_dti"] * z_dti
                  + p["beta_util"] * z_util
                  + p["beta_pti"] * z_pti
                  + p["beta_dsr"] * z_dsr
                  + p["beta_residual"] * z_resid
                  + p["beta_savings"] * z_sav
                  + p["beta_hist"] * z_hist
                  + p["beta_tenure"] * z_tenure
                  + p["beta_delinq24"] * df["delinquency_count_24m"].to_numpy(float)
                  + p["beta_searches"] * df["recent_credit_searches_6m"].to_numpy(float)
                  + p["beta_unverified"] * unverified
                  + p["beta_self_emp"] * self_emp
                  + p["beta_new_credit"] * df["new_credit_accounts_24m"].to_numpy(float)
                  + p["beta_other_earner"] * df["other_household_earner_flag"].to_numpy(float)
                  + p["beta_existing_customer"] * df["existing_customer_flag"].to_numpy(float)
                  + housing_adj + purpose_adj + adverse_adj + region_adj)

    # lin_static is the fixed part of the hazard. The coef_* arrays are the loan's
    # sensitivity to each macro factor, i.e. the planted interaction slope waiting to
    # be multiplied by that quarter's macro z-score inside the loop. z_risk is the
    # borrower's risk above the base rate, used to make safer loans prepay a bit more.
    return {
        "lin_static": lin_static,
        # income x unemployment
        "coef_zU": p["delta_income_unemp"] * z_inc,
        # DTI x Bank Rate
        "coef_zR": p["delta_dti_rate"] * z_dti,
        # income x CPI
        "coef_zCw": p["delta_income_cpi"] * z_inc,
        # self-employed x GDP
        "coef_zGDP": p["delta_selfemp_gdp"] * self_emp,
        # savings buffer x stress
        "coef_M": p["delta_savings_stress"] * z_sav,
        # centred risk index for prepay
        "z_risk": lin_static - p["base_hazard"],
        # per-loan micro z-values for the 7 candidate axes, so the
        # randomised-hidden-planted experiment can form ARBITRARY micro x macro products
        "micro_z": {"z_income": z_inc, "z_dti": z_dti, "z_util": z_util, "z_pti": z_pti,
                    "z_savbuf": z_sav, "z_lti": z_lti, "emp_self_employed": self_emp},
    }


def _emergent_static_components(df, pop_stats=None, params=None):
    """Per-loan pieces the EMERGENT hazard needs (mode="emergent" only).

    Two kinds of thing. First, the affordability MONEY components in raw GBP
    (housing, essential expenditure, existing debt service, scheduled payment,
    starting savings), kept raw so the quarter loop can reprice them with that
    quarter's macro. Second, `lin_nonaff`: the borrower risk main effects that do
    NOT flow through affordability (utilisation, credit history, tenure, delinquency,
    searches, unverified income, self-employed, the static flags, and the four
    categoricals). The affordability z-terms (income, DTI, PTI, DSR, residual,
    savings) are deliberately ABSENT here: in emergent mode they act only through the
    dynamic residual income, which is what lets the interaction emerge rather than be
    planted. Mirrors the driver definitions in _static_hazard_components exactly."""
    p = TRUE_PARAMS if params is None else params
    ps = POP_STATS if pop_stats is None else pop_stats
    util = df["revolving_utilisation"].to_numpy(float)
    util = np.where(np.isnan(util), ps["util_mu"], util)
    z_util = (util - ps["util_mu"]) / ps["util_sd"]
    z_hist = (df["credit_history_months"].to_numpy(float) - ps["hist_mu"]) / ps["hist_sd"]
    z_tenure = (df["employment_tenure_months"].to_numpy(float) - ps["tenure_mu"]) / ps["tenure_sd"]
    self_emp = df["_self_emp"].to_numpy(float)
    unverified = (df["income_evidence_type"].to_numpy() == "declared_only").astype(float)
    housing_adj = df["housing_status"].map(p["housing_beta"]).to_numpy(float)
    purpose_adj = df["loan_purpose"].map(p["purpose_beta"]).to_numpy(float)
    adverse_adj = df["public_adverse_record"].map(p["adverse_beta"]).to_numpy(float)
    region_adj = df["uk_region"].map(p["region_beta"]).to_numpy(float)

    # Borrower risk NOT running through affordability. No base_hazard here (the
    # emergent intercept lives in EMERGENT_PARAMS), and no income/DTI/PTI/DSR/
    # residual/savings main effects (those are the emergent dynamic channel).
    lin_nonaff = (p["beta_util"] * z_util
                  + p["beta_hist"] * z_hist
                  + p["beta_tenure"] * z_tenure
                  + p["beta_delinq24"] * df["delinquency_count_24m"].to_numpy(float)
                  + p["beta_searches"] * df["recent_credit_searches_6m"].to_numpy(float)
                  + p["beta_unverified"] * unverified
                  + p["beta_self_emp"] * self_emp
                  + p["beta_new_credit"] * df["new_credit_accounts_24m"].to_numpy(float)
                  + p["beta_other_earner"] * df["other_household_earner_flag"].to_numpy(float)
                  + p["beta_existing_customer"] * df["existing_customer_flag"].to_numpy(float)
                  + housing_adj + purpose_adj + adverse_adj + region_adj)

    return {
        "lin_nonaff": lin_nonaff,
        "self_emp": self_emp,
        "housing": df["monthly_housing_cost"].to_numpy(float),
        "essential_base": df["essential_monthly_expenditure"].to_numpy(float),
        "existing_base": df["existing_monthly_credit_payments"].to_numpy(float),
        "sched": df["_scheduled_payment"].to_numpy(float),
        "savings0": df["liquid_savings"].to_numpy(float),
    }


# ----------------------------------------------------------------------------
# Randomised hidden planted-interaction benchmark
#
# The planted arm is a METHODOLOGICAL benchmark (not a claim about real UK
# coefficients). One hidden set of five active pairs is chosen by a fixed seed and
# reused across all simulation seeds, so recovery frequency, sign and magnitude are
# directly comparable. Three signal strengths (null/moderate/strong) are applied as
# amplitudes on the standardised bilinear product z_micro * z_macro (the micro/macro
# z-scores are ~unit SD, so the amplitude IS the coefficient, matching the exact
# oracle). No Vasicek in the primary design, so the oracle is exactly expit(hazard).
# ----------------------------------------------------------------------------

# public 35-pair candidate axes (must match pipeline_utils MICRO_SHORT / MACRO_SHORT)
PLANTED_MICRO_AXES = ["z_income", "z_dti", "z_util", "z_pti", "z_savbuf", "z_lti", "emp_self_employed"]
PLANTED_MACRO_AXES = ["z_unemp", "z_cpi", "z_rate", "z_gdp", "z_wage"]
# strength ladder: null (control), moderate (headline), strong (positive control)
PLANTED_STRENGTHS = {"null": 0.0, "weak": 0.10, "moderate": 0.20, "strong": 0.35}


def select_hidden_planted_set(seed=20240824, n_active=5):
    """Constrained seeded random choice of the hidden active interaction set, FIXED
    across simulation seeds. Returns a list of {micro, macro, sign}. Signs are random.
    Constrained so the active set spans at least three distinct micro and three distinct
    macro axes (avoids a degenerate all-same-axis selection). The result is the private
    truth; the public registry (pipeline_utils) exposes only the 35 pair names."""
    rng = np.random.default_rng(seed)
    all_pairs = [(mi, ma) for mi in PLANTED_MICRO_AXES for ma in PLANTED_MACRO_AXES]
    chosen = None
    for _ in range(2000):
        idx = rng.choice(len(all_pairs), size=n_active, replace=False)
        cand = [all_pairs[i] for i in idx]
        if len({c[0] for c in cand}) >= 3 and len({c[1] for c in cand}) >= 3:
            chosen = cand
            break
    if chosen is None:
        chosen = [all_pairs[i] for i in idx]
    signs = rng.choice([-1.0, 1.0], size=n_active)
    return [{"micro": mi, "macro": ma, "sign": float(s)} for (mi, ma), s in zip(chosen, signs)]


def make_interaction_config(hidden_set, strength):
    """Turn a hidden set + a named strength into the per-pair delta list generate_panel
    consumes: delta = sign * amplitude on the standardised bilinear product."""
    amp = PLANTED_STRENGTHS[strength] if isinstance(strength, str) else float(strength)
    return [{"micro": h["micro"], "macro": h["macro"], "delta": h["sign"] * amp} for h in hidden_set]


# ----------------------------------------------------------------------------
# Panel generation: discrete-time survival over the real macro path
# ----------------------------------------------------------------------------

def generate_panel(macro_df=None, params=None, seed=42,
                   n_orig_per_quarter=N_ORIG_PER_QUARTER, orig_end=None,
                   affordability=True, mode="planted",
                   interaction_config=None, use_vasicek=True, paired_seed=None, crn_orig=None):
    """Simulate the full loan-level panel. A fresh cohort is booked each
    ORIGINATION quarter; all active loans are then rolled forward one quarter,
    defaulting (90+ DPD or UTP), prepaying, or maturing under the observation
    quarter's macro row.

    mode="planted" (default) is the direct planted hazard: the five delta_* interactions
    are injected as products and the oracle is exact. mode="emergent" instead wires
    macro into the affordability accounting each quarter (via emergent_affordability)
    and draws default from a convex hazard in residual income (via emergent_logit),
    so the same five interactions EMERGE with no product term. Everything else
    (origination, Vasicek trigger, competing risks, targets) is shared, and the
    planted branch is the direct planted hazard, so the two panels are directly comparable.

    Origination cutoff: quarters after `orig_end` are FOLLOW-UP ONLY (no new
    cohorts), so loans observed near the end still get a full forward-12m window.
    For the historical frame (macro_df=None) this defaults to ORIG_END (2024Q4),
    leaving 2025Q1-2025Q4 as pure follow-up. For a caller-supplied macro frame
    (e.g. the Stage 4 scenarios) it defaults to originating in EVERY quarter."""
    p = TRUE_PARAMS if params is None else params
    m = HIST_MACRO if macro_df is None else macro_df
    if m is None:
        raise RuntimeError(f"Macro frame unavailable: {globals().get('_LOAD_ERROR')}")
    rng = np.random.default_rng(seed)
    nq = len(m)
    rho, theta, s_eta = p["rho_asset"], p["theta_macro"], p["sigma_eta"]

    # origination cutoff (see docstring): historical frame stops originating at
    # ORIG_END; a supplied scenario frame originates in every quarter by default.
    if orig_end is None and macro_df is None:
        orig_end = ORIG_END
    if orig_end is not None:
        n_orig = int((m["quarter"] <= pd.Period(str(orig_end), freq="Q")).sum())
    else:
        n_orig = nq

    # 1. originate each ORIGINATION-quarter cohort up front (each quarter books
    #    exactly n_orig_per_quarter loans that PASS the affordability screen),
    #    concatenate into one static table
    statics, orig_q = [], []
    for t in range(n_orig):
        row = m.iloc[t]
        # true CRN: with crn_orig set, every cohort is originated at a FIXED reference
        # macro, so the borrower population is identical across paired scenarios (baseline
        # vs severe). Only the quarter-by-quarter ROLL then uses the scenario macro, so a
        # PD difference between scenarios is purely the macro effect, not composition.
        _o = crn_orig if crn_orig is not None else {"bank_rate": row["bank_rate"], "zU": row["zU"],
                                                    "zDSR": row["zDSR"], "zCredit": row["zCredit"]}
        cohort = simulate_accepted_cohort(n_orig_per_quarter, rng, affordability=affordability,
                                          bank_rate_level=float(_o["bank_rate"]),
                                          zU=float(_o["zU"]), zDSR=float(_o["zDSR"]),
                                          zCredit=float(_o["zCredit"]), params=p)
        statics.append(cohort)
        orig_q.append(np.full(len(cohort), t))
    static_df = pd.concat(statics, ignore_index=True)
    orig_q = np.concatenate(orig_q)
    N = len(static_df)
    comp = _static_hazard_components(static_df, POP_STATS, p)

    # emergent-mode setup. `ecomp` holds the raw affordability components plus the
    # non-affordability borrower risk. The emergent one-quarter hazard is deterministic
    # in the macro state (savings-buffer drain is expressed via the composite index M,
    # not path-tracked), so the same surface both generates data and yields the truth.
    if mode not in ("planted", "emergent"):
        raise ValueError(f"mode must be 'planted' or 'emergent', got {mode!r}")
    ep = EMERGENT_PARAMS
    ecomp = _emergent_static_components(static_df, POP_STATS, p) if mode == "emergent" else None

    amount = static_df["loan_amount"].to_numpy(float)
    apr = static_df["apr_at_origination"].to_numpy(float)
    term = static_df["loan_term_months"].to_numpy(float)

    # 2. per-loan mutable state
    # 0 active, 1 terminated
    status = np.zeros(N, dtype=np.int8)
    dpd = np.zeros(N, dtype=float)
    default_q = np.full(N, -1, dtype=np.int32)
    term_q = np.full(N, -1, dtype=np.int32)

    # dynamic state (income shocks + trailing-arrears memory)
    emp_orig = static_df["employment_status_at_origination"].to_numpy()
    net_orig = static_df["net_monthly_income"].to_numpy(float)
    is_employable = np.isin(emp_orig, ["full_time", "part_time", "temporary", "self_employed"])
    # current employment status (object)
    emp_t = emp_orig.copy()
    # current net monthly income
    net_t = net_orig.copy()
    # currently in an income shock
    shocked = np.zeros(N, dtype=bool)
    # trailing 4-quarter arrears indicators
    arr_ring = np.zeros((N, 4), dtype=np.float32)

    rows_loan, rows_t, rows_mob, rows_out, rows_dpd, rows_perf = [], [], [], [], [], []
    rows_shock, rows_empt, rows_nett, rows_prior = [], [], [], []

    # Roll the whole book forward one quarter at a time. Every loan that is alive in
    # quarter t is exposed to that quarter's macro state, so the same borrower meets
    # different economic conditions over its life. That is what makes an interaction
    # testable at all.
    for t in range(nq):
        row = m.iloc[t]
        # This quarter's macro state, as z-scores (winsorised where relevant).
        zU, zR, zCw = float(row["zU"]), float(row["zR"]), float(row["zCw"])
        zGDP, zWage, zSav, Mt = float(row["zGDP"]), float(row["zWage"]), float(row["zSav"]), float(row["M"])
        # Macro main effects, shared by every loan this quarter. Wage growth and the
        # saving ratio enter here only (they are the two decoys, main effect no interaction).
        macro_main = (p["beta_M"] * Mt + p["beta_gdp"] * zGDP
                      + p["beta_wage"] * zWage + p["beta_saving_macro"] * zSav)
        # Vasicek systematic factor: one shared shock per quarter that pushes every
        # loan the same way, so defaults cluster in bad quarters instead of being
        # independent. theta ties it to the stress index, sigma_eta adds noise.
        # shared systematic factor
        Z_t = -(theta * Mt + rng.normal(0, s_eta))

        # --- every loan alive this quarter (PATH A: booked this quarter or earlier) ---
        # Unified processing. A loan is at risk in every quarter it is alive, INCLUDING
        # its booking quarter, so there is one hazard mechanism for every loan-quarter.
        active = np.where((orig_q <= t) & (status == 0))[0]
        if active.size:
            # quarters on book (0 in the booking quarter)
            mob = (t - orig_q[active]).astype(float)
            matured = mob * 3.0 >= term[active]
            act = active[~matured]
            mat = active[matured]

            # mature loans close this quarter (competing exit; not a default). They are
            # emitted as "closed" and carry no target (perf 7 is terminal below).
            if mat.size:
                status[mat] = 1; term_q[mat] = t
                rows_loan.append(mat); rows_t.append(np.full(mat.size, t))
                rows_mob.append((t - orig_q[mat]).astype(float))
                rows_out.append(np.zeros(mat.size)); rows_dpd.append(dpd[mat].copy())
                # closed
                rows_perf.append(np.full(mat.size, 7, dtype=np.int8))
                rows_shock.append(shocked[mat].astype(np.int8))
                rows_empt.append(emp_t[mat].copy()); rows_nett.append(net_t[mat].copy())
                rows_prior.append(arr_ring[mat].sum(axis=1))

            if act.size:
                mob_a = (t - orig_q[act]).astype(float)
                season = p["beta_season"] * mob_a * np.exp(-mob_a / p["season_scale"])
                # arrears at the START of this quarter (entry state); drives the hazard
                # and the stored performance_status_t (which is now the ENTRY state, i.e.
                # the state the loan was in when this at-risk quarter began).
                entry_dpd = dpd[act].copy()
                inarr = p["beta_inarrears"] * (entry_dpd > 0)

                # trailing-arrears count (from the ring, BEFORE writing quarter t)
                prior_cnt = arr_ring[act].sum(axis=1)
                # income-shock dynamics on employable borrowers
                cur_shk = shocked[act]
                rec = cur_shk & (rng.random(act.size) < p["shock_recover_prob"])
                p_sh = p["shock_base_prob"] * (1.0 + p["shock_unemp_sens"] * max(zU, 0.0))
                new_sh = (~cur_shk) & is_employable[act] & (rng.random(act.size) < p_sh)
                # recoveries restore status/income
                idx_rec = act[rec]
                shocked[idx_rec] = False
                emp_t[idx_rec] = emp_orig[idx_rec]; net_t[idx_rec] = net_orig[idx_rec]
                # new shocks: job loss vs reduced hours
                idx_new = act[new_sh]
                if idx_new.size:
                    jobloss = rng.random(idx_new.size) < p["shock_jobloss_frac"]
                    shocked[idx_new] = True
                    emp_t[idx_new] = np.where(jobloss, "unemployed", "part_time")
                    net_t[idx_new] = net_orig[idx_new] * np.where(
                        jobloss, p["shock_mult_jobloss"], p["shock_mult_reduced"])
                shk_now = shocked[act].astype(float)

                # THE DIRECT DEFAULT HAZARD (PATH A). This is the complete one-quarter
                # default log-odds: the fixed per-loan part (base_hazard + borrower main
                # effects + categoricals), the macro main effects, the FIVE planted
                # interactions (the coef_* lines, evaluated at THIS quarter's macro),
                # seasoning, entry arrears, the income shock and the arrears memory.
                # The stored default_next_quarter target is drawn DIRECTLY from this, so
                # this equation IS the default probability and the oracle that rebuilds
                # it is exact. Because it uses THIS quarter's macro (the same macro stored
                # on the row), the feature macro and the event macro are aligned.
                if interaction_config is not None:
                    # Planted-interaction experiment: inject the hidden active pairs as
                    # standardised bilinear products at this condition's amplitude. No
                    # Vasicek by default, so the exact oracle is expit(logit_d).
                    _macro_vals = {"z_unemp": zU, "z_cpi": zCw, "z_rate": zR, "z_gdp": zGDP, "z_wage": zWage}
                    ix_sum = np.zeros(act.size)
                    for cfg in interaction_config:
                        ix_sum = ix_sum + cfg["delta"] * comp["micro_z"][cfg["micro"]][act] * _macro_vals[cfg["macro"]]
                    logit_d = (comp["lin_static"][act] + macro_main + ix_sum + season + inarr
                               + p["beta_income_shock"] * shk_now
                               + p["beta_prior_arrears"] * prior_cnt)
                    risk_e = comp["z_risk"][act]
                elif mode == "emergent":
                    # EMERGENT: macro flows into the affordability accounting for
                    # each active loan (unemployment->income drag, GDP->income,
                    # CPI->spend, Bank Rate->debt service, stress->buffer drain), then
                    # the hazard is a CONVEX function of the resulting residual income
                    # and effective savings buffer. No product term is added: the
                    # macro-by-micro interaction emerges from that convexity. Income
                    # uses ORIGINATION net income so the surface stays deterministic and
                    # the truth is extractable; the stochastic shock enters only as the
                    # income_shock_t main effect below.
                    resid_a, outgo_a, z_resid_a, z_savbuf_a = emergent_affordability(
                        net_orig[act], ecomp["self_emp"][act], ecomp["housing"][act],
                        ecomp["essential_base"][act], ecomp["existing_base"][act],
                        ecomp["sched"][act], ecomp["savings0"][act],
                        zU, zCw, zR, zGDP, Mt, ep, POP_STATS)
                    logit_d = emergent_logit(
                        ecomp["lin_nonaff"][act], macro_main, z_resid_a, z_savbuf_a,
                        season, inarr, shk_now, prior_cnt, ep, p)
                    # risk index used by the competing-risk / arrears-display realism
                    risk_e = (ecomp["lin_nonaff"][act]
                              + ep["beta_resid_lin"] * z_resid_a
                              + ep["beta_savbuf_lin"] * z_savbuf_a)
                else:
                    # THE DIRECT DEFAULT HAZARD (PATH A, planted).
                    logit_d = (comp["lin_static"][act] + macro_main
                               + comp["coef_zU"][act] * zU + comp["coef_zR"][act] * zR
                               + comp["coef_zCw"][act] * zCw + comp["coef_zGDP"][act] * zGDP
                               + comp["coef_M"][act] * Mt + season + inarr
                               + p["beta_income_shock"] * shk_now
                               + p["beta_prior_arrears"] * prior_cnt)
                    risk_e = comp["z_risk"][act]
                # default probability this quarter
                d = expit(logit_d)
                if use_vasicek:
                    # Vasicek trigger: threshold a latent asset value A that mixes the shared
                    # quarterly factor Z_t (weight sqrt(rho)) with an idiosyncratic draw
                    # (weight sqrt(1-rho)), so defaults still correlate within a quarter
                    # (Basel single-factor structure). is_default IS the target event.
                    thr = stats.norm.ppf(np.clip(d, 1e-9, 1 - 1e-9))
                    eps = rng.normal(0, 1, act.size)
                    A = np.sqrt(rho) * Z_t + np.sqrt(1 - rho) * eps
                    is_default = A < thr
                else:
                    # Direct Bernoulli draw -> exact oracle = expit(logit_d).
                    # With paired_seed, the default uniform for each loan-quarter is keyed by
                    # (paired_seed, t) and indexed by GLOBAL loan id, so it is IDENTICAL across
                    # strength conditions (common random numbers at the loan-quarter level).
                    if paired_seed is not None:
                        u_t = np.random.default_rng((int(paired_seed), int(t))).random(N)[act]
                    else:
                        u_t = rng.random(act.size)
                    is_default = u_t < d

                # Prepayment: a competing exit for non-defaulting loans.
                prepay_logit = (p["prepay_base"] + p["prepay_season"] * mob_a
                                - p["prepay_riskaverse"] * risk_e)
                do_prepay = (rng.random(act.size) < expit(prepay_logit)) & ~is_default

                # ARREARS DISPLAY (realism only; decoupled from default). Non-defaulting,
                # non-prepaying loans may roll into or out of arrears. This populates
                # performance_status_t (1-29/30-59/60-89 DPD) and prior_arrears_count_12m_t
                # but NEVER creates a default (capped below 90). A separate, higher
                # intercept (base_arrears) makes arrears more common than default.
                surv = ~is_default & ~do_prepay
                risk_index = risk_e
                arr_p = expit(p["base_arrears"] + risk_index + macro_main + season)
                adverse_disp = (rng.random(act.size) < arr_p) & surv
                steps = rng.choice(_DPD_STEPS, p=_DPD_STEP_P, size=act.size)
                new_dpd = entry_dpd.copy()
                # roll deeper into arrears, or cure on a clean quarter
                new_dpd = np.where(adverse_disp, new_dpd + steps, new_dpd)
                new_dpd = np.where(surv & ~adverse_disp, 0.0, new_dpd)
                # cap the DISPLAY strictly below 90 (90+ = default, which is the direct
                # draw's job, never the arrears roll's)
                new_dpd = np.clip(new_dpd, 0.0, 89.0)

                # performance_status_t = the ENTRY state of this at-risk quarter, from
                # entry_dpd. A loan Current at entry (0 DPD) is in the modelling set even
                # if it defaults THIS quarter, which is exactly the one-step hazard we want.
                perf = np.zeros(act.size, dtype=np.int8)
                perf = np.where((entry_dpd >= 1) & (entry_dpd <= 29), 1, perf)
                perf = np.where((entry_dpd >= 30) & (entry_dpd <= 59), 2, perf)
                perf = np.where((entry_dpd >= 60) & (entry_dpd <= 89), 3, perf).astype(np.int8)

                out_bal = outstanding_balance(amount[act], apr[act], term[act], mob_a * 3.0)
                out_bal = np.where(do_prepay, 0.0, out_bal)

                # terminations this quarter: default or prepay (maturity handled above)
                term_here = is_default | do_prepay
                dpd[act] = new_dpd
                fin = act[term_here]
                status[fin] = 1; term_q[fin] = t
                # record the quarter of default (used to build the aligned targets below)
                default_q[act[is_default]] = t

                rows_loan.append(act); rows_t.append(np.full(act.size, t))
                rows_mob.append(mob_a); rows_out.append(out_bal)
                rows_dpd.append(entry_dpd.copy()); rows_perf.append(perf)
                rows_shock.append(shocked[act].astype(np.int8))
                rows_empt.append(emp_t[act].copy()); rows_nett.append(net_t[act].copy())
                rows_prior.append(prior_cnt)
                # write this quarter's arrears indicator into the trailing ring (survivors)
                arr_ring[act, t % 4] = (new_dpd > 0).astype(np.float32)

    # loans still active at the end survived (right-censored). did_terminate
    # distinguishes a genuine in-sample resolution (default/prepay/mature) from
    # this end-of-panel bookkeeping value, so the target logic below does not
    # mistake "still active when the simulation stopped" for "known outcome".
    did_terminate = (status == 1).copy()
    surv = np.where(status == 0)[0]
    term_q[surv] = nq - 1

    # 3. assemble the panel
    loan_ix = np.concatenate(rows_loan)
    t_ix = np.concatenate(rows_t)
    mob = np.concatenate(rows_mob)
    out = np.concatenate(rows_out)
    dpd_col = np.concatenate(rows_dpd)
    perf = np.concatenate(rows_perf).astype(np.int8)
    shock_col = np.concatenate(rows_shock)
    empt_col = np.concatenate(rows_empt)
    nett_col = np.concatenate(rows_nett)
    prior_col = np.concatenate(rows_prior)

    panel = static_df.drop(columns=["age", "_self_emp", "_scheduled_payment", "_adverse_sev"]).iloc[loan_ix].reset_index(drop=True)
    panel.insert(0, "loan_id", ["L%07d" % i for i in loan_ix])
    panel.insert(1, "origination_quarter", m["quarter"].to_numpy()[orig_q[loan_ix]])
    panel.insert(2, "observation_quarter", m["quarter"].to_numpy()[t_ix])
    panel["outstanding_principal"] = out.round(0)
    panel["performance_status_t"] = pd.Series(perf).map(PERF_LABELS).to_numpy()
    # dynamic (observation-quarter) fields
    panel["employment_status_t"] = empt_col
    panel["income_shock_t"] = shock_col.astype(int)
    panel["net_monthly_income_t"] = np.round(nett_col, 0)
    panel["prior_arrears_count_12m_t"] = prior_col.astype(int)

    # quarter the loan defaulted (-1 if it never did)
    dq = default_q[loan_ix]
    # quarter the loan left the panel (default, prepay or maturity)
    tq = term_q[loan_ix]
    # True only if it genuinely resolved in-sample
    dt = did_terminate[loan_ix]
    # matured/closed rows carry no target (the loan ran its full term, not at risk here)
    terminal = (perf == 7)

    # 4b. PRIMARY target (PATH A): the ALIGNED one-quarter default hazard. For a
    # loan that is at risk in quarter t (performing at its start), did it default DURING
    # quarter t? That default was drawn directly from the planted hazard evaluated at
    # THIS quarter's macro, so the feature macro on the row and the event macro are the
    # same quarter. From the lender's decision point at the start of t this is still a
    # one-quarter-ahead default. Because every at-risk quarter's outcome is simulated,
    # there is no right-censoring here: the label is 1 (defaulted this quarter) or 0.
    within_q = (dq == t_ix)
    target_q = np.where(within_q, 1.0, 0.0)
    # matured rows: no prediction
    target_q = np.where(terminal, np.nan, target_q)
    panel["default_next_quarter"] = target_q

    # 4. SECONDARY target: default within this quarter or the next three, window
    # [t, t+3] (four quarters). Same aligned convention as the primary target but over
    # a four-quarter horizon. Right-censoring still applies here: if the four-quarter
    # window runs past the panel end and the loan had not already resolved, we cannot
    # tell, so the label is NaN (dropped, not imputed as a survival).
    within = (dq != -1) & (dq >= t_ix) & (dq <= t_ix + 3)
    # a 0 is trustworthy only if the whole window is inside the panel, or the loan
    # actually resolved by t+3 (so we truly saw it not default in the remaining window)
    fully_observed = (t_ix + 3 <= nq - 1) | (dt & (tq <= t_ix + 3))
    target = np.where(within, 1.0, np.where(fully_observed, 0.0, np.nan))
    # matured rows: no prediction
    target = np.where(terminal, np.nan, target)
    panel["default_next_12m"] = target

    # 5. attach macro series at the observation quarter
    macro_cols = ["unemployment", "cpi_inflation", "bank_rate", "gdp_growth",
                  "wage_growth", "dsr", "credit_growth", "saving_ratio"]
    macro_at_t = m[macro_cols].to_numpy()[t_ix]
    for j, c in enumerate(macro_cols):
        panel[c] = macro_at_t[:, j]

    return panel


# The three lists below declare the panel schema. STORED_MICRO_COLUMNS are the raw
# borrower and loan fields written to the panel. ENGINEERED_COLUMNS are the ratios
# and lifecycle features derived from those raw fields by feature_engineering (kept
# separate so a ratio is defined in exactly one place and cannot drift). MACRO_COLUMNS
# are the eight macro series appended at the observation quarter. Stage 1 checks the
# generated panel against these lists so any missing or extra column is caught early.
STORED_MICRO_COLUMNS = [
    "loan_id", "origination_quarter", "observation_quarter", "age_at_origination", "age_band", "uk_region",
    "employment_status_at_origination", "employment_status_t", "employment_tenure_months",
    "financial_dependants", "other_household_earner_flag", "existing_customer_flag",
    "income_evidence_type", "gross_annual_income", "net_monthly_income", "net_monthly_income_t",
    "income_shock_t", "housing_status", "monthly_housing_cost", "essential_monthly_expenditure",
    "remaining_unsecured_balance", "existing_monthly_credit_payments", "liquid_savings",
    "credit_history_months", "open_credit_accounts", "new_credit_accounts_24m",
    "revolving_utilisation", "revolving_credit_limit", "delinquency_count_24m",
    "months_since_last_delinquency", "worst_prior_delinquency_band_24m",
    "prior_arrears_count_12m_t", "public_adverse_record", "recent_credit_searches_6m",
    "loan_amount", "loan_term_months", "apr_at_origination", "loan_purpose",
    "outstanding_principal", "performance_status_t",
    # PRIMARY target + 12m sensitivity target
    "default_next_quarter", "default_next_12m",
]
# columns produced by feature_engineering() (not stored raw twice in a model)
ENGINEERED_COLUMNS = [
    "quarters_on_book", "months_on_book", "scheduled_monthly_payment", "log_income",
    "debt_to_income", "existing_dsr", "payment_to_income", "residual_income",
    "savings_buffer_months", "loan_to_income", "util_missing",
]
MACRO_COLUMNS = ["unemployment", "cpi_inflation", "bank_rate", "gdp_growth",
                 "wage_growth", "dsr", "credit_growth", "saving_ratio"]


# ----------------------------------------------------------------------------
# Feature engineering: affordability ratios + lifecycle + regime
# ----------------------------------------------------------------------------

def feature_engineering(df):
    """Compute the affordability ratios and lifecycle features from the stored raw
    components. Origination-based by design (the hybrid dynamics keep the modelling
    features on origination values; the dynamic _t fields stay separate). Idempotent;
    safe to call on any raw panel frame.

        DTI               = existing unsecured balance / gross income
        existing DSR       = existing monthly payments / net income
        PTI                = scheduled loan payment / net income
        residual income    = net income - all monthly outgoings
        savings buffer     = liquid savings / total monthly outgoings
        LTI                = loan amount / gross income
        time on book       = observation quarter - origination quarter
    """
    out = df.copy()
    oq = out["origination_quarter"].astype("period[Q]")
    ob = out["observation_quarter"].astype("period[Q]")
    out["quarters_on_book"] = (ob - oq).apply(lambda x: x.n)
    # quarters x 3
    out["months_on_book"] = out["quarters_on_book"] * 3
    out["scheduled_monthly_payment"] = amortised_monthly_payment(
        out["loan_amount"], out["apr_at_origination"], out["loan_term_months"]).round(0)
    out["log_income"] = np.log(out["gross_annual_income"])
    nmi = out["net_monthly_income"].clip(lower=1)
    out["debt_to_income"] = out["remaining_unsecured_balance"] / out["gross_annual_income"]
    out["existing_dsr"] = out["existing_monthly_credit_payments"] / nmi
    out["payment_to_income"] = out["scheduled_monthly_payment"] / nmi
    out["residual_income"] = (out["net_monthly_income"] - out["monthly_housing_cost"]
                              - out["essential_monthly_expenditure"]
                              - out["existing_monthly_credit_payments"]
                              - out["scheduled_monthly_payment"])
    _outgo = (out["monthly_housing_cost"] + out["essential_monthly_expenditure"]
              + out["existing_monthly_credit_payments"]
              + out["scheduled_monthly_payment"]).clip(lower=1)
    out["savings_buffer_months"] = out["liquid_savings"] / _outgo
    out["loan_to_income"] = out["loan_amount"] / out["gross_annual_income"]
    # keep no-facility flag; utilisation itself is imputed later in preprocessing
    out["util_missing"] = out["revolving_utilisation"].isna().astype(int)
    return out


def add_regime(df, hist_macro=None):
    """Label each row calm / normal / stressed using terciles of the stress index M
    at its observation quarter. The bottom third of quarters (lowest M) are calm, the
    top third are stressed, the middle third are normal. This is a project-defined
    split for the regime-stability analysis, not an official Bank of England
    classification, and it lets Stage 4 and Stage 5 compare model behaviour across
    genuinely calmer and genuinely tougher parts of the real UK cycle."""
    m = HIST_MACRO if hist_macro is None else hist_macro
    out = df.copy()
    qM = m.set_index("quarter")["M"]; qM.index = qM.index.astype(str)
    # the two tercile cut points on M
    t1, t2 = qM.quantile(1 / 3), qM.quantile(2 / 3)
    regime_of_q = pd.cut(qM, [-np.inf, t1, t2, np.inf], labels=["calm", "normal", "stressed"])
    oqs = out["observation_quarter"].astype("period[Q]").astype(str)
    out["regime"] = oqs.map(regime_of_q).astype("object")
    return out


# Running this file directly writes the ground-truth JSON. Downstream stages load it
# to rebuild the exact oracle (the true hazard) and to grade each model's recovered
# interactions against the planted coefficients. It stores the true parameters, the
# z-score reference, the macro moments, and the settings needed to reproduce the run.
if __name__ == "__main__":
    import json
    with open("ground_truth_params.json", "w") as f:
        json.dump({**{k: v for k, v in TRUE_PARAMS.items()},
                   "pop_stats": POP_STATS,
                   "hist_stats": {k: {"mu": v[0], "sd": v[1]} for k, v in HIST_STATS.items()},
                   "winsor_sd": WINSOR, "m_weights": M_WEIGHTS,
                   # the emergent-mode structural sensitivities, so downstream
                   # stages can rebuild the emergent hazard and measure its truth.
                   "emergent_params": EMERGENT_PARAMS,
                   "n_orig_per_quarter": N_ORIG_PER_QUARTER, "seed": 42}, f, indent=2)
    print("Wrote ground_truth_params.json")
