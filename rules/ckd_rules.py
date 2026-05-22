# rules/ckd_rules.py
from rules.ckd_classifier import (
    classify_ckd_stage,
    classify_acr_category,
    is_ckd_confirmed
)

# ── Drug name lists for medication gap checks ────────────────────────
RAAS_INHIBITORS = [
    "ramipril", "lisinopril", "enalapril", "perindopril",
    "captopril", "fosinopril",             # ACE inhibitors
    "losartan", "valsartan", "candesartan",
    "irbesartan", "telmisartan",           # ARBs
    "sacubitril"                           # ARNI component
]

SGLT2_INHIBITORS = [
    "dapagliflozin", "empagliflozin",
    "canagliflozin", "sotagliflozin"
]


def _safe_float(value, default=None):
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_test_date(label: str, date_value) -> str:
    if not date_value:
        return ""
    normalized = str(date_value).strip()

    # Databricks dates often arrive as YYYYMMDD; render as YYYY/MM/DD.
    if len(normalized) == 8 and normalized.isdigit():
        normalized = f"{normalized[:4]}/{normalized[4:6]}/{normalized[6:]}"

    return f" ({label} date: {normalized})"


def run_ckd_rules(patient: dict) -> tuple[str, str, list[str]]:
    flags = []

    # ── Extract values ───────────────────────────────────────────────
    egfr       = _safe_float(patient.get("egfr"))
    potassium  = _safe_float(patient.get("potassium"))
    urine_acr  = _safe_float(patient.get("urine_acr"))
    systolic   = _safe_float(patient.get("systolic_bp"))
    meds       = [m.lower() for m in patient.get("medications", [])]
    icd_codes  = [c.lower() for c in patient.get("icd_codes", [])]

    has_diabetes = any("diabetes" in c for c in icd_codes)

    egfr_date = _format_test_date("eGFR", patient.get("egfr_date"))
    acr_date = _format_test_date("ACR", patient.get("acr_date"))
    potassium_date = _format_test_date("potassium", patient.get("potassium_date"))
    bp_date = _format_test_date("BP", patient.get("bp_date"))

    # ── Rule 0 — CKD confirmation gate ───────────────────────────────
    if not is_ckd_confirmed(patient):
        return "Not CKD", "Unknown", []

    # ── Stage + ACR ──────────────────────────────────────────────────
    stage = classify_ckd_stage(egfr) if egfr is not None else "Unknown"
    acr_cat = classify_acr_category(urine_acr) if urine_acr is not None else "Unknown"

    # ── Rule 1 — Blood Pressure (refined) ────────────────────────────
    if systolic is not None:
        if systolic >= 140:
            flags.append(
                f"Uncontrolled BP (SBP {systolic:.0f}) — requires treatment{bp_date}"
            )
        elif systolic >= 130:
            flags.append(
                f"Suboptimal BP (SBP {systolic:.0f}) — consider intensification{bp_date}"
            )

    # ── Rule 2 — ACEi / ARB (fixed) ──────────────────────────────────
    has_raas = any(drug in meds for drug in RAAS_INHIBITORS)
    if urine_acr is not None:
        if urine_acr >= 30 and not has_raas:
            flags.append(
                "Missing ACEi/ARB — strong indication "
                f"(ACR {urine_acr:.0f} mg/g, {acr_cat}){acr_date}"
            )
        elif (
            3 <= urine_acr < 30 and
            not has_raas and
            (has_diabetes or (systolic is not None and systolic >= 130))
        ):
            flags.append(
                "Consider ACEi/ARB — mild albuminuria with risk factors "
                f"(ACR {urine_acr:.0f}){acr_date}"
            )

    # ── Rule 3 — SGLT2 inhibitor (updated) ───────────────────────────
    has_sglt2 = any(drug in meds for drug in SGLT2_INHIBITORS)
    if (
        egfr is not None and egfr >= 20 and
        urine_acr is not None and urine_acr >= 30 and
        not has_sglt2
    ):
        flags.append(
            "Missing SGLT2 inhibitor — indicated per KDIGO 2024 "
            f"(eGFR {egfr:.0f}, ACR {urine_acr:.0f}){egfr_date}{acr_date}"
        )

    # ── Rule 4 — Hyperkalaemia (context-aware) ───────────────────────
    if potassium is not None and potassium > 5.5:
        if has_raas:
            flags.append(
                f"Hyperkalaemia (K+ {potassium:.1f}) — review/hold ACEi/ARB{potassium_date}"
            )
        else:
            flags.append(
                f"Hyperkalaemia (K+ {potassium:.1f}) — caution starting RAAS inhibitors{potassium_date}"
            )

    # ── Rule 5 — Advanced CKD medication review ──────────────────────
    if stage in {"Stage 4", "Stage 5"} and egfr is not None:
        flags.append(
            f"{stage} CKD (eGFR {egfr:.0f}){egfr_date} — "
            "review medications for renal dosing and nephrotoxicity"
        )

    # ── Rule 6 — Nephrology referral (expanded) ──────────────────────
    if egfr is not None and (
        egfr < 30 or
        (urine_acr is not None and urine_acr >= 300)
    ):
        flags.append(
            f"High-risk CKD — consider nephrology referral "
            f"(eGFR {egfr:.0f}, ACR {urine_acr:.0f}){egfr_date}{acr_date}"
        )

    return stage, acr_cat, flags