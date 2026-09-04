"""
pipeline_utils.py
Shared helper functions used by every stage notebook.

WHAT THIS FILE IS FOR
The DGP file builds the data. This file holds the shared tools the five stage
notebooks all need, so those tools are written once and behave the same way across
stages, the interaction column names and the oracle formula end up identical in
Stage 2, Stage 3 and Stage 4 because they all come from here rather than being
copy-pasted around.

Four groups of helpers live here, in order:

  1. NAMING AND FEATURE BUILDING (MICRO_SHORT, MACRO_SHORT, ix_col, add_derived,
     Z_SOURCES, the categorical level lists, apply_preprocessing). Turns a raw panel
     into the model-ready feature matrix: adds the affordability ratios (by calling
     the DGP's own feature_engineering, so a ratio is defined in one place only),
     z-scores the drivers using frozen training moments, one-hot encodes the
     categoricals, and builds the interaction product columns.

  2. PREDICTION WRAPPERS (make_logreg_fn, make_gam_fn, make_ebm_fn, make_truth_fn).
     Wraps a fitted model, or the known truth, in a function that takes a feature
     frame and returns the log-odds. Giving every model the same call signature is
     what lets the interaction-quantification code treat them interchangeably.
     make_truth_fn rebuilds the exact DGP hazard from the ground-truth JSON, so
     there's always a real answer to compare against.

  3. INTERACTION QUANTIFICATION (single_feature_pds, friedman_h2_and_bilinear,
     pure_interaction_surface). The model-agnostic machinery measuring how much of
     a model's behaviour is a genuine interaction between two features, using
     partial dependence, Friedman's H2 statistic, and an equivalent bilinear
     coefficient so any model can be scored on the same scale.

  4. METRICS AND IO (gini, expected_calibration_error, load_json, save_json). Small
     shared scoring and file helpers.

TWO DESIGN CHOICES WORTH KNOWING
The modelling set is current-state rows only, with an observed target. On those
rows the DGP's arrears term is identically zero, so the models answer a clean
behavioural-scoring question about performing loans. The five planted interactions
are always evaluated on origination micro values. The savings-buffer interaction
was planted on the composite stress index M, a weighted blend of unemployment, CPI
and Bank Rate (weights 0.5, 0.25, 0.25), so it correctly carries signal across all
three macro axes.
"""

import json
import numpy as np
import pandas as pd
from scipy.special import logit as _logit_fn
from scipy.special import expit as _expit, ndtr as _Phi, ndtri as _Phi_inv

# small floor used to keep logits and divisions numerically safe
EPS = 1e-9

# ----------------------------------------------------------------------------
# Experiment protocol constants + the temporal split
# ----------------------------------------------------------------------------
# five independent simulation seeds (replications from the UNCHANGED generator) and
# three signal conditions (null control / moderate headline / strong positive control)
REDUCED_SEEDS = [1, 2, 3, 4, 5]
REDUCED_STRENGTHS = ["null", "weak", "moderate", "strong"]
# observation-time block boundaries for the loan-disjoint temporal split
SPLIT_DEV_END = "2016Q4"
SPLIT_VAL_END = "2019Q4"


def time_blocked_loan_disjoint_split(df, dev_end=SPLIT_DEV_END, val_end=SPLIT_VAL_END):
    """Partition modelling rows into development / validation / confirmation so the split
    is BOTH loan-disjoint AND observation-time-blocked (Stage 2 objective 2.3/2.4).

    Each loan is assigned wholly by its ORIGINATION quarter block (dev: <= dev_end;
    val: (dev_end, val_end]; conf: > val_end), and each loan's rows are then CENSORED to
    that block's observation window, so no loan contributes rows to more than one block
    and each block is a genuine later time period. Returns a Series aligned to df.index
    with values in {'dev','val','conf'} or NaN for censored rows (to be dropped)."""
    oq = df["origination_quarter"].astype("period[Q]")
    ob = df["observation_quarter"].astype("period[Q]")
    dev_end_p = pd.Period(dev_end, "Q"); val_end_p = pd.Period(val_end, "Q")
    part = pd.Series(np.nan, index=df.index, dtype=object)
    part[(oq <= dev_end_p) & (ob <= dev_end_p)] = "dev"
    part[(oq > dev_end_p) & (oq <= val_end_p) & (ob > dev_end_p) & (ob <= val_end_p)] = "val"
    part[(oq > val_end_p) & (ob > val_end_p)] = "conf"
    return part

# Short names for the interaction axes. These map the long feature names to tidy
# stems so an interaction column has one canonical name everywhere. For example
# z_income crossed with z_unemp always becomes "ix_income_unemp", in every stage.
MICRO_SHORT = {"z_income": "income", "z_dti": "dti", "z_util": "util",
               "z_pti": "pti", "z_savbuf": "savbuf", "z_lti": "lti",
               "emp_self_employed": "selfemp"}
MACRO_SHORT = {"z_unemp": "unemp", "z_cpi": "cpi", "z_rate": "rate",
               "z_gdp": "gdp", "z_wage": "wage"}


def ix_col(micro_z, macro_z):
    # Build the canonical interaction-column name from a micro and a macro axis.
    return f"ix_{MICRO_SHORT[micro_z]}_{MACRO_SHORT[macro_z]}"


# ----------------------------------------------------------------------------
# Derived affordability features (recomputed from raw stored components)
# ----------------------------------------------------------------------------

def _amort(P, apr_pct, term_months):
    r = np.asarray(apr_pct, float) / 1200.0
    P = np.asarray(P, float); n = np.asarray(term_months, float)
    with np.errstate(over="ignore", invalid="ignore"):
        fac = np.where(r > 0, r / (1 - (1 + r) ** (-n)), 1.0 / n)
    return P * fac


def add_derived(df):
    """Add the engineered columns by delegating to the DGP's own feature
    engineering section (credit_dgp.feature_engineering), so the ratio
    definitions live in one place. Idempotent; safe on any raw panel frame."""
    from credit_dgp import feature_engineering
    return feature_engineering(df)


# z-score sources (engineered z-name -> derived/raw column)
Z_SOURCES = {
    "z_income": "log_income", "z_dti": "debt_to_income", "z_dsr": "existing_dsr",
    "z_pti": "payment_to_income", "z_residual": "residual_income",
    "z_savbuf": "savings_buffer_months", "z_lti": "loan_to_income",
    "z_util": "revolving_utilisation", "z_hist": "credit_history_months",
    "z_tenure": "employment_tenure_months", "z_mob": "quarters_on_book",
    "z_searches": "recent_credit_searches_6m", "z_dependants": "financial_dependants",
    "z_unemp": "unemployment", "z_cpi": "cpi_inflation", "z_rate": "bank_rate",
    "z_gdp": "gdp_growth", "z_wage": "wage_growth", "z_dsr_m": "dsr",
    "z_credit_m": "credit_growth", "z_sav_m": "saving_ratio",
}

EMP_DUMMIES = ["part_time", "temporary", "self_employed", "unemployed",
               # full_time base
               "benefits", "retired", "student", "other"]
HOUSING_LEVELS = ["own_outright", "mortgage", "private_rent", "social_rent", "family_other"]
HOUSING_BASELINE = "own_outright"
EVIDENCE_LEVELS = ["payslip_paye", "bank_transactions", "tax_return",
                   "pension_evidence", "benefits_evidence", "declared_only"]
EVIDENCE_BASELINE = "payslip_paye"
PURPOSE_LEVELS = ["debt_consolidation", "vehicle", "home_improvement",
                  "wedding_event", "holiday", "education", "other_unknown",
                  "essential_living_costs", "unexpected_emergency", "housing_payment"]
PURPOSE_BASELINE = "debt_consolidation"
ADVERSE_LEVELS = ["none", "CCJ", "recent_default", "IVA_DRO", "bankruptcy"]
ADVERSE_BASELINE = "none"
REGION_LEVELS = ["NE", "NW", "YH", "EM", "WM", "EE", "LON", "SE", "SW", "SCT", "WAL", "NI"]
REGION_BASELINE = "NE"
AGE_BANDS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]
AGE_BASELINE = "35-44"


def apply_preprocessing(df, meta):
    """Turn a raw panel frame into the exact model-ready feature matrix, using the
    FROZEN training moments stored in meta. Stage 1 does this once to build the
    train/test files; Stage 4 reuses this same function on freshly simulated stress
    scenarios, so a scenario is preprocessed identically to the training data.

    The steps are: add the affordability ratios, impute missing utilisation with the
    training mean, z-score every driver against the training mean and standard
    deviation, rebuild the composite stress index, one-hot the categoricals against
    the same baselines, and finally build the interaction product columns."""
    # affordability ratios from the DGP
    out = add_derived(df)
    # Current arrears state at the START of the at-risk quarter: 1 if the loan entered the
    # quarter already in 1-89 DPD, else 0. This mirrors the DGP's beta_inarrears hazard term,
    # which lifts the default odds for a loan that begins the quarter in arrears. It is read
    # straight off the stored entry-state performance_status_t, so no re-derivation is needed.
    out["in_arrears_t"] = out["performance_status_t"].isin(
        ["1-29 DPD", "30-59 DPD", "60-89 DPD"]).astype(int)
    # frozen per-feature mean and sd
    sc = meta["scaler"]
    # training-mean utilisation
    util_impute = meta["util_impute"]
    # Fill missing utilisation (no revolving facility) with the training mean, so the
    # z-score below is well defined. The separate util_missing flag keeps that info.
    out["revolving_utilisation"] = out["revolving_utilisation"].fillna(util_impute)

    # Standardise each driver using the FROZEN training moments (never this frame's
    # own moments), so scenario and test data land on the same scale as training.
    for zc, src in Z_SOURCES.items():
        out[zc] = (out[src] - sc[zc]["mu"]) / sc[zc]["sd"]

    # composite stress index on the standardised scale (matches DGP M weights)
    out["z_M"] = 0.5 * out["z_unemp"] + 0.25 * out["z_cpi"] + 0.25 * out["z_rate"]

    # One-hot each categorical, dropping the baseline level (reference coding) so the
    # dummies are not collinear. Baselines match the DGP definitions.
    for lvl in EMP_DUMMIES:
        out[f"emp_{lvl}"] = (out["employment_status_at_origination"] == lvl).astype(int)
    for lvl in HOUSING_LEVELS:
        if lvl != HOUSING_BASELINE:
            out[f"housing_{lvl}"] = (out["housing_status"] == lvl).astype(int)
    for lvl in EVIDENCE_LEVELS:
        if lvl != EVIDENCE_BASELINE:
            out[f"evid_{lvl}"] = (out["income_evidence_type"] == lvl).astype(int)
    for lvl in PURPOSE_LEVELS:
        if lvl != PURPOSE_BASELINE:
            out[f"purpose_{lvl}"] = (out["loan_purpose"] == lvl).astype(int)
    for lvl in ADVERSE_LEVELS:
        if lvl != ADVERSE_BASELINE:
            out[f"adverse_{lvl}"] = (out["public_adverse_record"] == lvl).astype(int)
    for lvl in REGION_LEVELS:
        if lvl != REGION_BASELINE:
            out[f"region_{lvl}"] = (out["uk_region"] == lvl).astype(int)
    for lvl in AGE_BANDS:
        if lvl != AGE_BASELINE:
            out[f"age_{lvl}"] = (out["age_band"] == lvl).astype(int)

    # Build each interaction column as the simple product of its micro and macro
    # z-scores. The logistic model uses these directly; the H2 machinery uses them
    # as candidates to test.
    for pair in meta["candidate_pairs"]:
        out[pair["col"]] = out[pair["micro"]].values * out[pair["macro"]].values
    return out


# ----------------------------------------------------------------------------
# Common-scale prediction functions (all return log-odds)
#
# Every model is wrapped so it can be called the same way: pass a feature frame,
# get back the log-odds. That shared signature is what lets the interaction code in
# Stage 3 and Stage 4 loop over the models without special-casing each one.
# ----------------------------------------------------------------------------

def _to_logit(p):
    # Convert a probability to log-odds, clipped away from 0 and 1 so it stays finite.
    return _logit_fn(np.clip(p, EPS, 1 - EPS))


def make_logreg_fn(sm_result, meta):
    # Wrap the fitted logistic model. It rebuilds the interaction columns then applies
    # the fitted coefficients by hand (a plain dot product), which returns log-odds.
    base = meta["base_features"]; pairs = meta["candidate_pairs"]

    def f(X):
        D = X[base].astype(float).copy()
        for pr in pairs:
            D[pr["col"]] = X[pr["micro"]].values * X[pr["macro"]].values
        # intercept column to match sm params
        D.insert(0, "const", 1.0)
        return np.asarray(D @ sm_result.params)
    return f


def make_gam_fn(gam, meta):
    # Wrap the fitted GAM. It predicts a probability, which we convert back to log-odds
    # so it is on the same scale as the other wrapped models.
    base = meta["base_features"]

    def f(X):
        return _to_logit(gam.predict_proba(X[base].astype(float).values))
    return f


def make_ebm_fn(ebm, meta):
    # Wrap the fitted EBM. Same idea as the GAM wrapper: take the positive-class
    # probability and convert to log-odds.
    base = meta["base_features"]

    def f(X):
        return _to_logit(ebm.predict_proba(X[base].astype(float).values)[:, 1])
    return f


def make_truth_fn(gt, meta):
    """Rebuild the oracle: the exact true default probability the DGP used, as a
    function of the engineered features. Since the true coefficients are known (loaded
    from the ground-truth JSON, gt), the real probability for any row can be computed
    and every fitted model graded against it.

    default_next_quarter is drawn directly from the planted hazard at the observation
    quarter via the Vasicek trigger, so this rebuilds the latent hazard log-odds
    (logit_d) then applies the closed-form Vasicek marginalisation to get the exact
    log-odds of default. A calibration regression of the realised outcome on this
    oracle comes out with slope ~1, intercept ~0, and a tiny ECE.

    One scaling subtlety: model features are z-scored against the training moments,
    but the DGP wrote its hazard against its own reference moments (POP_STATS for
    micro, HIST_STATS for macro). Each feature gets converted back to a raw value
    then re-standardised onto the DGP scale before the true coefficients apply.

    This is the Vasicek-path oracle. make_planted_experiment_oracle_fn and
    make_emergent_truth_fn draw straight from Bernoulli(expit(logit)) instead and
    carry no Vasicek marginalisation."""
    sc = meta["scaler"]; ps = gt["pop_stats"]; hs = gt["hist_stats"]
    # winsor cap, matching the DGP
    W = gt.get("winsor_sd", 3.0)

    def rescale(X, zc, mu_dgp, sd_dgp):
        # Undo the training-moment z-score to get the raw value, then re-standardise
        # onto the DGP's own reference scale so the true coefficients line up.
        raw = X[zc] * sc[zc]["sd"] + sc[zc]["mu"]
        return (raw - mu_dgp) / sd_dgp

    housing_map = [("mortgage", "mortgage"), ("private_rent", "private_rent"),
                   ("social_rent", "social_rent"), ("family_other", "family_other")]
    purpose_map = [(l, l) for l in PURPOSE_LEVELS if l != PURPOSE_BASELINE]
    adverse_map = [(l, l) for l in ADVERSE_LEVELS if l != ADVERSE_BASELINE]
    region_nonbase = [r for r in REGION_LEVELS if r != REGION_BASELINE]

    def f(X):
        z_inc = rescale(X, "z_income", ps["log_inc_mu"], ps["log_inc_sd"])
        z_dti = rescale(X, "z_dti", ps["dti_mu"], ps["dti_sd"])
        z_dsr = rescale(X, "z_dsr", ps["dsr_mu"], ps["dsr_sd"])
        z_pti = rescale(X, "z_pti", ps["pti_mu"], ps["pti_sd"])
        z_res = np.clip(rescale(X, "z_residual", ps["resid_mu"], ps["resid_sd"]), -4, 4)
        z_sav = rescale(X, "z_savbuf", ps["savbuf_mu"], ps["savbuf_sd"])
        z_util = rescale(X, "z_util", ps["util_mu"], ps["util_sd"])
        z_hist = rescale(X, "z_hist", ps["hist_mu"], ps["hist_sd"])
        z_ten = rescale(X, "z_tenure", ps["tenure_mu"], ps["tenure_sd"])
        # macro to DGP z-scale
        zU = rescale(X, "z_unemp", hs["unemployment"]["mu"], hs["unemployment"]["sd"])
        zC = rescale(X, "z_cpi", hs["cpi_inflation"]["mu"], hs["cpi_inflation"]["sd"])
        zR = rescale(X, "z_rate", hs["bank_rate"]["mu"], hs["bank_rate"]["sd"])
        zGDP = np.clip(rescale(X, "z_gdp", hs["gdp_growth"]["mu"], hs["gdp_growth"]["sd"]), -W, W)
        zWage = rescale(X, "z_wage", hs["wage_growth"]["mu"], hs["wage_growth"]["sd"])
        zSav = rescale(X, "z_sav_m", hs["saving_ratio"]["mu"], hs["saving_ratio"]["sd"])
        zCw = np.clip(zC, -W, W)
        M = 0.5 * zU + 0.25 * zC + 0.25 * zR

        self_emp = X["emp_self_employed"].values
        unverified = X["evid_declared_only"].values
        # quarters
        mob = X["z_mob"] * sc["z_mob"]["sd"] + sc["z_mob"]["mu"]
        season = gt["beta_season"] * mob * np.exp(-mob / gt["season_scale"])

        housing_adj = sum(gt["housing_beta"][b] * X[f"housing_{c}"] for c, b in housing_map)
        purpose_adj = (gt["purpose_beta"][PURPOSE_BASELINE]
                       + sum((gt["purpose_beta"][b] - gt["purpose_beta"][PURPOSE_BASELINE])
                             * X[f"purpose_{c}"] for c, b in purpose_map))
        adverse_adj = sum(gt["adverse_beta"][b] * X[f"adverse_{c}"] for c, b in adverse_map)
        base_r = gt["region_beta"][REGION_BASELINE]
        region_adj = base_r + sum((gt["region_beta"][r] - base_r) * X[f"region_{r}"]
                                  for r in region_nonbase)

        # Reassemble the exact DGP LATENT default hazard (logit_d): base rate, borrower
        # main effects, categoricals, macro main effects, seasoning, and the FIVE planted
        # interactions on the last five lines. The default event is drawn from this via
        # the Vasicek trigger, so these five lines are the ground truth being recovered.
        logit_d = np.asarray(
            gt["base_hazard"]
            + gt["beta_income"] * z_inc + gt["beta_dti"] * z_dti
            + gt["beta_util"] * z_util + gt["beta_pti"] * z_pti
            + gt["beta_dsr"] * z_dsr + gt["beta_residual"] * z_res
            + gt["beta_savings"] * z_sav + gt["beta_hist"] * z_hist
            + gt["beta_tenure"] * z_ten
            + gt["beta_delinq24"] * X["delinquency_count_24m"]
            + gt["beta_searches"] * X["recent_credit_searches_6m"]
            + gt["beta_unverified"] * unverified
            + gt["beta_self_emp"] * self_emp
            # new main effects (static + dynamic), mirroring the generator
            + gt["beta_new_credit"] * X["new_credit_accounts_24m"]
            + gt["beta_other_earner"] * X["other_household_earner_flag"]
            + gt["beta_existing_customer"] * X["existing_customer_flag"]
            + gt["beta_income_shock"] * X["income_shock_t"]
            + gt["beta_prior_arrears"] * X["prior_arrears_count_12m_t"]
            + gt["beta_inarrears"] * X["in_arrears_t"]
            + housing_adj + purpose_adj + adverse_adj + region_adj
            + gt["beta_M"] * M + gt["beta_gdp"] * zGDP
            + gt["beta_wage"] * zWage + gt["beta_saving_macro"] * zSav
            + season
            # 1 income x unemployment
            + gt["delta_income_unemp"] * z_inc * zU
            # 2 DTI x Bank Rate
            + gt["delta_dti_rate"] * z_dti * zR
            # 3 income x CPI
            + gt["delta_income_cpi"] * z_inc * zCw
            # 4 self-employed x GDP
            + gt["delta_selfemp_gdp"] * self_emp * zGDP
            # 5 savings buffer x stress
            + gt["delta_savings_stress"] * z_sav * M)
        # PATH A EXACT ORACLE. The realised default is A < Phi^-1(expit(logit_d)) with a
        # latent asset A = sqrt(rho)*Z + sqrt(1-rho)*eps, Z = -(theta*M + N(0,sigma)).
        # Marginalising the unobserved systematic shock gives the EXACT borrower-conditional
        # default probability in closed form: P = Phi( (Phi^-1(d) + sqrt(rho)*theta*M) / s ),
        # with s = sqrt(rho*sigma^2 + (1-rho)). Returning logit(P) makes the oracle the true
        # log-odds of the target, so a calibration regression of the outcome on it has slope
        # ~1 and intercept ~0 (the acceptance test), and the small 1/s scaling of the planted
        # deltas is what the Stage 2 attenuation slope reports.
        rho = gt.get("rho_asset", 0.10); theta = gt.get("theta_macro", 0.50); sig = gt.get("sigma_eta", 0.45)
        d = np.clip(_expit(logit_d), 1e-9, 1 - 1e-9)
        s_A = float(np.sqrt(rho * sig * sig + (1.0 - rho)))
        thr = _Phi_inv(d)
        P = _Phi((thr + np.sqrt(rho) * theta * np.asarray(M, float)) / s_A)
        return _logit_fn(np.clip(P, 1e-9, 1 - 1e-9))
    return f


# ----------------------------------------------------------------------------
# EMERGENT truth: the economically grounded target for interaction recovery
#
# make_truth_fn above rebuilds the PLANTED hazard (the exact instrument). The two
# functions here rebuild the EMERGENT hazard and MEASURE the interaction it contains.
# No product is planted: the macro economy enters the affordability accounting
# deterministically (mirroring credit_dgp.emergent_affordability) and the
# hazard is convex in the resulting residual income and effective savings buffer, so
# the macro-by-micro interaction emerges. Because the emergent truth is not an exact
# constant, emergent_truth_table summarises each pair's coefficient-equivalent with a
# bootstrap confidence interval, using the SAME H2/bilinear ruler applied to the
# models, so model recovery and the emergent target sit on one scale.
# ----------------------------------------------------------------------------

# Reconstruction constant (see make_emergent_truth_fn): the DGP mean monthly service
# rate on unsecured balance, used to rebuild existing debt service from DTI. Sets the
# emergent magnitude scale, not its sign or shape.
_EMERGENT_SERVICE_RATE = 0.035


def make_emergent_truth_fn(gt, meta):
    """Rebuild the EMERGENT one-quarter default log-odds as a function of the model
    features, so the SAME H2/bilinear machinery that scores the models can MEASURE the
    emergent interaction on the identical ruler. Unlike make_truth_fn (which injects
    the five planted delta_* products), no product is planted here: the macro economy
    enters the affordability accounting deterministically and the hazard is convex in
    the resulting residual income and effective savings buffer, so the interaction
    EMERGES. This is the ECONOMIC recovery target; its magnitude is summarised with a
    bootstrap CI by emergent_truth_table rather than read off a constant.

    Faithfulness. Affordability money components are reconstructed from the features on
    the DGP scale (net income from z_income; existing debt service tied to z_dti;
    scheduled payment from z_pti; origination residual from z_residual), so perturbing
    a micro axis flows through the same chain the generator used. Essential expenditure
    (the CPI channel) is reconstructed with the DGP's own sublinear income formula, so
    its share of income falls as income rises; housing takes the remainder of non-debt
    outgoings. The Vasicek marginalisation is applied exactly as in make_truth_fn."""
    from credit_dgp import uk_net_annual_income
    ep = gt["emergent_params"]
    sc = meta["scaler"]; ps = gt["pop_stats"]; hs = gt["hist_stats"]
    W = gt.get("winsor_sd", 3.0)

    def raw_of(X, zc):
        # undo the training-moment z-score to recover the raw feature value
        return X[zc] * sc[zc]["sd"] + sc[zc]["mu"]

    def rescale(X, zc, mu_dgp, sd_dgp):
        # raw value, then re-standardise onto the DGP's own reference scale
        return (raw_of(X, zc) - mu_dgp) / sd_dgp

    housing_map = [("mortgage", "mortgage"), ("private_rent", "private_rent"),
                   ("social_rent", "social_rent"), ("family_other", "family_other")]
    purpose_map = [(l, l) for l in PURPOSE_LEVELS if l != PURPOSE_BASELINE]
    adverse_map = [(l, l) for l in ADVERSE_LEVELS if l != ADVERSE_BASELINE]
    region_nonbase = [r for r in REGION_LEVELS if r != REGION_BASELINE]

    def f(X):
        # --- macro state on the DGP scale ---
        zU = np.asarray(rescale(X, "z_unemp", hs["unemployment"]["mu"], hs["unemployment"]["sd"]), float)
        zC = np.asarray(rescale(X, "z_cpi", hs["cpi_inflation"]["mu"], hs["cpi_inflation"]["sd"]), float)
        zR = np.asarray(rescale(X, "z_rate", hs["bank_rate"]["mu"], hs["bank_rate"]["sd"]), float)
        zGDP = np.clip(np.asarray(rescale(X, "z_gdp", hs["gdp_growth"]["mu"], hs["gdp_growth"]["sd"]), float), -W, W)
        zWage = np.asarray(rescale(X, "z_wage", hs["wage_growth"]["mu"], hs["wage_growth"]["sd"]), float)
        zSav = np.asarray(rescale(X, "z_sav_m", hs["saving_ratio"]["mu"], hs["saving_ratio"]["sd"]), float)
        zCw = np.clip(zC, -W, W)
        M = 0.5 * zU + 0.25 * zC + 0.25 * zR

        # --- reconstruct raw affordability components from the features ---
        gross = np.exp(np.asarray(raw_of(X, "z_income"), float))            # annual gross
        net = np.asarray(uk_net_annual_income(gross), float) / 12.0         # monthly net
        dti = np.asarray(raw_of(X, "z_dti"), float)                          # balance / gross
        existing = np.maximum(dti, 0.0) * gross * _EMERGENT_SERVICE_RATE     # monthly debt service
        pti = np.asarray(raw_of(X, "z_pti"), float)
        sched = np.maximum(pti, 0.0) * net                                   # scheduled payment
        resid_static = np.asarray(raw_of(X, "z_residual"), float)           # origination residual
        # non-debt outgoings from the residual identity, split into essential + housing.
        # Essential is reconstructed with the DGP's own SUBLINEAR income formula, so its
        # SHARE of income falls as income rises (this is the income x CPI mechanism);
        # housing takes the remainder. The split does not change resid at macro=0.
        nondebt = np.maximum(net - resid_static - existing - sched, 0.0)
        dep = np.asarray(X["financial_dependants"], float)
        essential = (600.0 + 300.0 * dep) * (gross / 35000.0) ** 0.15
        essential = np.minimum(essential, nondebt)
        housing = np.maximum(nondebt - essential, 0.0)

        # --- the deterministic macro->micro chain (mirrors emergent_affordability) ---
        self_emp = np.asarray(X["emp_self_employed"], float)
        gdp_sens = np.where(self_emp > 0, ep["gdp_income_sens_se"], ep["gdp_income_sens_emp"])
        income_eff = (net * (1.0 + gdp_sens * zGDP)
                      - ep["unemp_income_drag"] * np.maximum(zU, 0.0) * ep["ref_net_monthly"])
        expend = essential * (1.0 + ep["cpi_exp_sens"] * zCw)
        existing_eff = existing * (1.0 + ep["rate_passthrough"] * zR)
        resid_dyn = income_eff - housing - expend - existing_eff - sched
        # RELATIVE residual (share of net income), matching emergent_affordability, so
        # the macro squeeze bites hardest where the affordability share is tight.
        z_resid_dyn = np.clip((resid_dyn / np.maximum(net, 1.0) - ps["residfrac_mu"]) / ps["residfrac_sd"], -4, 4)
        z_sav = rescale(X, "z_savbuf", ps["savbuf_mu"], ps["savbuf_sd"])
        z_savbuf_eff = np.asarray(z_sav, float) - ep["buffer_drain"] * np.maximum(M, 0.0)

        # --- non-affordability borrower risk (same betas as the generator) ---
        z_util = rescale(X, "z_util", ps["util_mu"], ps["util_sd"])
        z_hist = rescale(X, "z_hist", ps["hist_mu"], ps["hist_sd"])
        z_ten = rescale(X, "z_tenure", ps["tenure_mu"], ps["tenure_sd"])
        unverified = X["evid_declared_only"].values
        housing_adj = sum(gt["housing_beta"][b] * X[f"housing_{c}"] for c, b in housing_map)
        purpose_adj = (gt["purpose_beta"][PURPOSE_BASELINE]
                       + sum((gt["purpose_beta"][b] - gt["purpose_beta"][PURPOSE_BASELINE])
                             * X[f"purpose_{c}"] for c, b in purpose_map))
        adverse_adj = sum(gt["adverse_beta"][b] * X[f"adverse_{c}"] for c, b in adverse_map)
        base_r = gt["region_beta"][REGION_BASELINE]
        region_adj = base_r + sum((gt["region_beta"][r] - base_r) * X[f"region_{r}"]
                                  for r in region_nonbase)
        lin_nonaff = (gt["beta_util"] * z_util + gt["beta_hist"] * z_hist
                      + gt["beta_tenure"] * z_ten
                      + gt["beta_delinq24"] * X["delinquency_count_24m"]
                      + gt["beta_searches"] * X["recent_credit_searches_6m"]
                      + gt["beta_unverified"] * unverified
                      + gt["beta_self_emp"] * self_emp
                      + gt["beta_new_credit"] * X["new_credit_accounts_24m"]
                      + gt["beta_other_earner"] * X["other_household_earner_flag"]
                      + gt["beta_existing_customer"] * X["existing_customer_flag"]
                      + housing_adj + purpose_adj + adverse_adj + region_adj)

        macro_main = (gt["beta_M"] * M + gt["beta_gdp"] * zGDP
                      + gt["beta_wage"] * zWage + gt["beta_saving_macro"] * zSav)
        mob = raw_of(X, "z_mob")
        season = gt["beta_season"] * mob * np.exp(-mob / gt["season_scale"])

        conv_r = np.log1p(np.exp(np.clip(-z_resid_dyn, -30, 30)))
        conv_s = np.maximum(-z_savbuf_eff, 0.0)
        logit_d = np.asarray(
            ep["base_hazard_e"] + lin_nonaff + macro_main
            + ep["beta_resid_lin"] * z_resid_dyn + ep["beta_resid_conv"] * conv_r
            + ep["beta_savbuf_lin"] * z_savbuf_eff + ep["beta_savbuf_conv"] * conv_s
            + gt["beta_income_shock"] * X["income_shock_t"]
            + gt["beta_prior_arrears"] * X["prior_arrears_count_12m_t"]
            + gt["beta_inarrears"] * X["in_arrears_t"]
            + season, float)

        # The emergent panels are generated with use_vasicek=False, so each default is a
        # direct Bernoulli draw from expit(logit_d). The exact borrower-conditional oracle
        # log-odds is therefore logit_d itself, with NO Vasicek marginalisation. Applying the
        # marginalisation here would fold the macro stress index M back into the target
        # nonlinearly and distort the very emergent interaction the ruler is meant to read.
        return logit_d
    return f


def emergent_truth_table(f_truth, X_bg_full, candidate_pairs, n_boot=15,
                         bg_size=150, seed=0):
    """Measure the emergent interaction per candidate pair, with a bootstrap CI.

    For each bootstrap resample of the background, run the SAME H2/bilinear ruler used
    on the fitted models (single_feature_pds + friedman_h2_and_bilinear). Returns a
    DataFrame indexed by pair with the mean coefficient-equivalent (the emergent
    'truth' magnitude), its 95% CI, sign, and mean H2. The CI is the honest statement
    that the emergent truth is estimated, not exact."""
    rng = np.random.default_rng(seed)
    axes = sorted({p["micro"] for p in candidate_pairs} | {p["macro"] for p in candidate_pairs})
    boot = {p["col"]: {"b": [], "h2": []} for p in candidate_pairs}
    n = len(X_bg_full)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=min(bg_size, n))
        Xb = X_bg_full.iloc[idx].reset_index(drop=True)
        pd1 = single_feature_pds(f_truth, Xb, axes)
        for p in candidate_pairs:
            h, b = friedman_h2_and_bilinear(f_truth, Xb, p["micro"], p["macro"], pd1)
            boot[p["col"]]["b"].append(b); boot[p["col"]]["h2"].append(h)
    rows = []
    for p in candidate_pairs:
        b = np.array(boot[p["col"]]["b"]); h = np.array(boot[p["col"]]["h2"])
        rows.append({"pair": p["col"], "micro": p["micro"], "macro": p["macro"],
                     "is_true": bool(p.get("is_true", False)),
                     "b_emergent": float(b.mean()),
                     "b_lo": float(np.quantile(b, 0.025)),
                     "b_hi": float(np.quantile(b, 0.975)),
                     "sign": int(np.sign(b.mean())),
                     "H2_emergent": float(h.mean())})
    return pd.DataFrame(rows).set_index("pair")


# ----------------------------------------------------------------------------
# Exact oracle for the randomised-hidden-planted experiment,
# and the truth-free public candidate registry.
# ----------------------------------------------------------------------------

def public_candidate_registry():
    """The public 35-pair candidate grid: pair names only, with NO truth labels, signs
    or coefficients. This is what the modelling/discovery stage is allowed to see."""
    return [{"micro": mi, "macro": ma, "col": ix_col(mi, ma)}
            for mi in MICRO_SHORT for ma in MACRO_SHORT]


def make_planted_experiment_oracle_fn(gt, meta, interaction_config):
    """Exact oracle for a randomised-hidden-planted experiment panel (direct-Bernoulli,
    no Vasicek). Rebuilds the DGP hazard log-odds from the model features: borrower main
    effects + categoricals + macro main effects + seasoning + the ARBITRARY active
    interaction products (delta * micro_z * macro_z) in interaction_config. Because the
    experiment DGP draws default directly from Bernoulli(expit(logit)), the exact oracle
    PD log-odds IS this logit (no Vasicek marginalisation). Feed this to the common
    H2/bilinear ruler to obtain the oracle's OWN b-hat per pair, which is the correct
    recovery target (not the raw planted delta)."""
    sc = meta["scaler"]; ps = gt["pop_stats"]; hs = gt["hist_stats"]; W = gt.get("winsor_sd", 3.0)

    def rescale(X, zc, mu, sd):
        return (X[zc] * sc[zc]["sd"] + sc[zc]["mu"] - mu) / sd

    housing_map = [("mortgage", "mortgage"), ("private_rent", "private_rent"),
                   ("social_rent", "social_rent"), ("family_other", "family_other")]
    purpose_map = [(l, l) for l in PURPOSE_LEVELS if l != PURPOSE_BASELINE]
    adverse_map = [(l, l) for l in ADVERSE_LEVELS if l != ADVERSE_BASELINE]
    region_nonbase = [r for r in REGION_LEVELS if r != REGION_BASELINE]

    def f(X):
        z_inc = rescale(X, "z_income", ps["log_inc_mu"], ps["log_inc_sd"])
        z_dti = rescale(X, "z_dti", ps["dti_mu"], ps["dti_sd"])
        z_dsr = rescale(X, "z_dsr", ps["dsr_mu"], ps["dsr_sd"])
        z_pti = rescale(X, "z_pti", ps["pti_mu"], ps["pti_sd"])
        z_res = np.clip(rescale(X, "z_residual", ps["resid_mu"], ps["resid_sd"]), -4, 4)
        z_sav = rescale(X, "z_savbuf", ps["savbuf_mu"], ps["savbuf_sd"])
        z_util = rescale(X, "z_util", ps["util_mu"], ps["util_sd"])
        z_hist = rescale(X, "z_hist", ps["hist_mu"], ps["hist_sd"])
        z_ten = rescale(X, "z_tenure", ps["tenure_mu"], ps["tenure_sd"])
        z_lti = rescale(X, "z_lti", ps.get("lti_mu", 0.0), ps.get("lti_sd", 1.0))
        zU = rescale(X, "z_unemp", hs["unemployment"]["mu"], hs["unemployment"]["sd"])
        zC = rescale(X, "z_cpi", hs["cpi_inflation"]["mu"], hs["cpi_inflation"]["sd"])
        zR = rescale(X, "z_rate", hs["bank_rate"]["mu"], hs["bank_rate"]["sd"])
        zGDP = np.clip(rescale(X, "z_gdp", hs["gdp_growth"]["mu"], hs["gdp_growth"]["sd"]), -W, W)
        zWage = rescale(X, "z_wage", hs["wage_growth"]["mu"], hs["wage_growth"]["sd"])
        zSav = rescale(X, "z_sav_m", hs["saving_ratio"]["mu"], hs["saving_ratio"]["sd"])
        zCw = np.clip(zC, -W, W); M = 0.5 * zU + 0.25 * zC + 0.25 * zR
        self_emp = X["emp_self_employed"].values; unverified = X["evid_declared_only"].values
        mob = X["z_mob"] * sc["z_mob"]["sd"] + sc["z_mob"]["mu"]
        season = gt["beta_season"] * mob * np.exp(-mob / gt["season_scale"])
        housing_adj = sum(gt["housing_beta"][b] * X[f"housing_{c}"] for c, b in housing_map)
        purpose_adj = (gt["purpose_beta"][PURPOSE_BASELINE]
                       + sum((gt["purpose_beta"][b] - gt["purpose_beta"][PURPOSE_BASELINE])
                             * X[f"purpose_{c}"] for c, b in purpose_map))
        adverse_adj = sum(gt["adverse_beta"][b] * X[f"adverse_{c}"] for c, b in adverse_map)
        base_r = gt["region_beta"][REGION_BASELINE]
        region_adj = base_r + sum((gt["region_beta"][r] - base_r) * X[f"region_{r}"] for r in region_nonbase)
        micro = {"z_income": z_inc, "z_dti": z_dti, "z_util": z_util, "z_pti": z_pti,
                 "z_savbuf": z_sav, "z_lti": z_lti, "emp_self_employed": self_emp}
        macro = {"z_unemp": zU, "z_cpi": zCw, "z_rate": zR, "z_gdp": zGDP, "z_wage": zWage}
        logit = np.asarray(
            gt["base_hazard"]
            + gt["beta_income"] * z_inc + gt["beta_dti"] * z_dti + gt["beta_util"] * z_util
            + gt["beta_pti"] * z_pti + gt["beta_dsr"] * z_dsr + gt["beta_residual"] * z_res
            + gt["beta_savings"] * z_sav + gt["beta_hist"] * z_hist + gt["beta_tenure"] * z_ten
            + gt["beta_delinq24"] * X["delinquency_count_24m"]
            + gt["beta_searches"] * X["recent_credit_searches_6m"]
            + gt["beta_unverified"] * unverified + gt["beta_self_emp"] * self_emp
            + gt["beta_new_credit"] * X["new_credit_accounts_24m"]
            + gt["beta_other_earner"] * X["other_household_earner_flag"]
            + gt["beta_existing_customer"] * X["existing_customer_flag"]
            + gt["beta_income_shock"] * X["income_shock_t"]
            + gt["beta_prior_arrears"] * X["prior_arrears_count_12m_t"]
            + gt["beta_inarrears"] * X["in_arrears_t"]
            + housing_adj + purpose_adj + adverse_adj + region_adj
            + gt["beta_M"] * M + gt["beta_gdp"] * zGDP + gt["beta_wage"] * zWage + gt["beta_saving_macro"] * zSav
            + season, float)
        for cfg in interaction_config:
            logit = logit + cfg["delta"] * np.asarray(micro[cfg["micro"]], float) * np.asarray(macro[cfg["macro"]], float)
        return logit  # exact planted-experiment oracle log-odds (no Vasicek marginalisation)
    return f


# ----------------------------------------------------------------------------
# Friedman H-statistic + equivalent bilinear coefficient
#
# This is the model-agnostic ruler for interactions. The idea: build the two-feature
# partial dependence, subtract off what each feature does on its own, and whatever is
# left is the pure interaction. H2 reports the share of the joint effect that is
# interaction; the bilinear coefficient turns that surface into a single comparable
# number. Because it only needs a predict function, it works on logistic, GAM, EBM
# and the oracle alike, which is what makes cross-model comparison fair.
# ----------------------------------------------------------------------------

def _partial_dependence_at_sample(f, X_bg, cols, values_per_row):
    # Average partial dependence over a background sample. For each requested value it
    # overwrites the given columns, scores the whole background, and averages, which
    # marginalises out every other feature.
    n = len(X_bg)
    stacked = pd.concat([X_bg] * n, ignore_index=True)
    for c, v in zip(cols, values_per_row.T):
        stacked[c] = np.repeat(v, n)
    preds = f(stacked)
    return preds.reshape(n, n).mean(axis=1)


def single_feature_pds(f, X_bg, features):
    # One-feature partial dependence for each feature, centred to mean zero. These are
    # the "each feature on its own" baselines that the interaction is measured against.
    out = {}
    for c in features:
        pd1 = _partial_dependence_at_sample(f, X_bg, [c], X_bg[[c]].values)
        out[c] = pd1 - pd1.mean()
    return out


def friedman_h2_and_bilinear(f, X_bg, col_j, col_k, pd_singles, pred_var=None,
                             floor_frac=0.01):
    # Two-feature partial dependence, then remove each feature's solo effect. The
    # residual is the pure interaction. H2 is its share of the joint variation; the
    # bilinear coefficient is the slope of a * b fitted to that residual.
    pd_jk = _partial_dependence_at_sample(f, X_bg, [col_j, col_k],
                                          X_bg[[col_j, col_k]].values)
    pd_jk = pd_jk - pd_jk.mean()
    resid = pd_jk - pd_singles[col_j] - pd_singles[col_k]
    denom = np.sum(pd_jk ** 2)
    # Stabilise H2 under the null. For an inert pair both numerator and denominator
    # collapse toward zero and the raw ratio blows up to implausible values (this was
    # the Logistic null-threshold instability). Require the pair's joint variation to
    # be at least `floor_frac` of the model's total prediction variation
    # (n * Var(centred log-odds), passed in as pred_var) before the ratio is trusted;
    # otherwise the pair contributes negligibly and H2 is floored toward 0. Passing
    # pred_var=None preserves the original unfloored behaviour.
    if pred_var is not None and pred_var > 0:
        denom = max(denom, floor_frac * len(pd_jk) * pred_var)
    # H2 = interaction variance / total joint variance (0 means no interaction).
    h2 = float(np.sum(resid ** 2) / denom) if denom > 0 else 0.0
    # Fit intercept + a + b + a*b to the residual by least squares; the a*b slope is
    # the equivalent bilinear coefficient, directly comparable to the planted delta.
    a, b = X_bg[col_j].values, X_bg[col_k].values
    D = np.column_stack([np.ones_like(a), a, b, a * b])
    beta, *_ = np.linalg.lstsq(D, resid, rcond=None)
    return h2, float(beta[3])


def pure_interaction_surface(f, X_bg, col_j, col_k, grid_j, grid_k):
    # Build the interaction as a 2-D surface over a grid of the two features, for
    # plotting. Same logic as friedman_h2_and_bilinear, but evaluated on a regular
    # grid and weighted by how common each feature value actually is, so the surface
    # is centred and comparable to the true interaction surface.
    n = len(X_bg); gj, gk = len(grid_j), len(grid_k)
    stacked = pd.concat([X_bg] * (gj * gk), ignore_index=True)
    aa, bb = np.meshgrid(grid_j, grid_k, indexing="ij")
    stacked[col_j] = np.repeat(aa.ravel(), n)
    stacked[col_k] = np.repeat(bb.ravel(), n)
    pd_jk = f(stacked).reshape(gj * gk, n).mean(axis=1).reshape(gj, gk)

    def pd1(col, grid):
        st = pd.concat([X_bg] * len(grid), ignore_index=True)
        st[col] = np.repeat(np.asarray(grid), n)
        return f(st).reshape(len(grid), n).mean(axis=1)

    pd_j, pd_k = pd1(col_j, grid_j), pd1(col_k, grid_k)

    def marg_w(col, grid):
        edges = np.concatenate([[-np.inf], (np.asarray(grid[:-1]) + np.asarray(grid[1:])) / 2, [np.inf]])
        w = np.histogram(X_bg[col].values, bins=edges)[0].astype(float)
        return w / w.sum()

    wj, wk = marg_w(col_j, grid_j), marg_w(col_k, grid_k)
    Wm = np.outer(wj, wk)
    pd_j_c = pd_j - np.sum(pd_j * wj)
    pd_k_c = pd_k - np.sum(pd_k * wk)
    surf = pd_jk - pd_j_c[:, None] - pd_k_c[None, :]
    surf = surf - np.sum(surf * Wm)
    return surf


# ----------------------------------------------------------------------------
# Metrics and small IO helpers
# ----------------------------------------------------------------------------

def gini(auc):
    # Gini coefficient, the usual credit-scoring rescaling of AUC (0 = random).
    return 2 * auc - 1


def expected_calibration_error(y_true, p_hat, n_bins=10):
    # How far predicted probabilities sit from realised default rates, on average.
    # Sort predictions into equal-count bins, and in each bin compare the mean
    # prediction with the actual default rate. The weighted average of those gaps is
    # the ECE. Lower is better; this is the metric that exposes over/under-prediction.
    y_true, p_hat = np.asarray(y_true), np.asarray(p_hat)
    qs = np.quantile(p_hat, np.linspace(0, 1, n_bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    idx = np.searchsorted(qs, p_hat, side="right") - 1
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(y_true[m].mean() - p_hat[m].mean())
    return float(ece)


def load_json(path):
    # Read a JSON file (ground truth, preprocessing meta, or a stage result).
    with open(path) as fh:
        return json.load(fh)


def save_json(obj, path):
    # Write a JSON file. default=float lets numpy numbers serialise cleanly.
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=float)
