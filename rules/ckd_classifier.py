def classify_ckd_stage(egfr: float) -> str:
    """
    Classify CKD stage from eGFR value using KDIGO thresholds.
    """
    if egfr >= 90:
        return "Stage 1"
    if egfr >= 60:
        return "Stage 2"
    if egfr >= 45:
        return "Stage 3a"
    if egfr >= 30:
        return "Stage 3b"
    if egfr >= 15:
        return "Stage 4"
    return "Stage 5"


def classify_acr_category(urine_acr: float) -> str:
    """
    Classify albuminuria category from urine ACR (mg/g).
    """
    if urine_acr < 3:
        return "A1"
    if urine_acr < 30:
        return "A2"
    return "A3"


def is_ckd_confirmed(icd_codes: list[str]) -> bool:
    """
    Check if CKD is confirmed via ICD-10 codes.
    """
    ckd_icd_prefixes = ("N18.1", "N18.2", "N18.3", "N18.4", "N18.5", "N18.6", "N18.9")
    return any(code.startswith(prefix) for code in icd_codes for prefix in ckd_icd_prefixes)
# rules/ckd_classifier.py

def classify_ckd_stage(egfr: float) -> str:
    """
    Rule 0 — Classify CKD stage from eGFR value.
    Uses KDIGO 2024 thresholds.
    """
    if egfr >= 90:   return "Stage 1"
    elif egfr >= 60: return "Stage 2"
    elif egfr >= 45: return "Stage 3a"
    elif egfr >= 30: return "Stage 3b"
    elif egfr >= 15: return "Stage 4"
    else:            return "Stage 5"


def classify_acr_category(urine_acr: float) -> str:
    """
    Classify albuminuria category from urine ACR (mg/g).
    A1 = normal, A2 = moderately increased, A3 = severely increased.
    """
    if urine_acr < 30:    return "A1"
    elif urine_acr < 300: return "A2"
    else:                return "A3"


def is_ckd_confirmed(icd_codes: list) -> bool:
    """
    Check if CKD is confirmed via ICD-10 codes.
    N18.1–N18.6 covers all CKD stages.
    """
    CKD_ICD_PREFIXES = ("N18.1","N18.2","N18.3",
                        "N18.4","N18.5","N18.6","N18.9")
    return any(
        code.startswith(p)
        for code in icd_codes
        for p in CKD_ICD_PREFIXES
    )