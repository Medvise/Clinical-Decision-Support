# rules/patient_summary.py
import argparse
from datetime import date
import json
import re
import sys

from rules.ckd_rules import run_ckd_rules
from rules.disease_inference import (
    infer_patient_comorbidity_tags,
    infer_patient_disease_tags,
    primary_disease_tag,
)
from rules.utils import parse_date, safe_float


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return []
        if "," in cleaned:
            return [v.strip() for v in cleaned.split(",") if v.strip()]
        return [cleaned]
    return [str(value).strip()]


def _parse_bp(bp_value: str) -> tuple[float | None, float | None]:
    if not bp_value or not isinstance(bp_value, str):
        return None, None
    cleaned = bp_value.strip().lower()
    if cleaned in {"unknown", "not known", "na", "n/a"}:
        return None, None

    # Handles common formats such as:
    # - "120/80", "120 / 80", "120/80 mmHg"
    # - "120-80", "120 - 80"
    # - "120 over 80"
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:/|-|\bover\b)\s*(\d+(?:\.\d+)?)",
        cleaned,
    )
    if not match:
        return None, None

    systolic = safe_float(match.group(1))
    diastolic = safe_float(match.group(2))
    return systolic, diastolic


def _is_present(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    if isinstance(value, (list, tuple, dict)) and len(value) == 0:
        return False
    return True


def _pick_date(raw_patient: dict, *keys: str):
    for key in keys:
        if _is_present(raw_patient.get(key)):
            return raw_patient.get(key)
    return None


def _drop_null_values(data: dict) -> dict:
    return {k: v for k, v in data.items() if _is_present(v)}


CLINICAL_OBSERVATION_SPECS = [
    ("Urine ACR (albumin/creat ratio)", "acr_result", "acr_unit", "acr_performed_date", "acr_date"),
    ("Creatinine", "creatinine_result", "creatinine_unit", "creatinine_performed_date", "creatinine_date"),
    ("eGFR", "egfr_result", "egfr_unit", "egfr_performed_date", "egfr_date"),
    ("Potassium", "potassium_result", "potassium_unit", "potassium_performed_date", "potassium_date"),
    ("Aldosterone", "aldosterone_result", "aldosterone_unit", "aldosterone_performed_date", "aldosterone_date"),
    ("Renin", "renin_result", "renin_unit", "renin_performed_date", "renin_date"),
    ("Aldosterone/Renin ratio", "aldosterone_renin_ratio_result", "aldosterone_renin_ratio_unit", "aldosterone_renin_ratio_performed_date", "aldosterone_renin_ratio_date"),
    ("Fasting glucose", "fasting_glucose_result", "fasting_glucose_unit", "fasting_glucose_performed_date", "fasting_glucose_date"),
    ("Random glucose", "random_glucose_result", "random_glucose_unit", "random_glucose_performed_date", "random_glucose_date"),
    ("HbA1c", "hba1c_result", "hba1c_unit", "hba1c_performed_date", "hba1c_date"),
    ("Total cholesterol", "total_cholesterol_result", "total_cholesterol_unit", "total_cholesterol_performed_date", "total_cholesterol_date"),
    ("LDL", "ldl_result", "ldl_unit", "ldl_performed_date", "ldl_date"),
    ("HDL", "hdl_result", "hdl_unit", "hdl_performed_date", "hdl_date"),
    ("Triglycerides", "triglycerides_result", "triglycerides_unit", "triglycerides_performed_date", "triglycerides_date"),
    ("Troponin", "troponin_result", "troponin_unit", "troponin_performed_date", "troponin_date"),
    ("BNP", "bnp_result", "bnp_unit", "bnp_performed_date", "bnp_date"),
    ("NT-proBNP", "nt_pro_bnp_result", "nt_pro_bnp_unit", "nt_pro_bnp_performed_date", "nt_pro_bnp_date"),
    ("CRP", "crp_result", "crp_unit", "crp_performed_date", "crp_date"),
    ("hs-CRP", "hs_crp_result", "hs_crp_unit", "hs_crp_performed_date", "hs_crp_date"),
    ("Hemoglobin", "hemoglobin_result", "hemoglobin_unit", "hemoglobin_performed_date", "hemoglobin_date"),
    ("Sodium", "sodium_result", "sodium_unit", "sodium_performed_date", "sodium_date"),
    ("TSH", "tsh_result", "tsh_unit", "tsh_performed_date", "tsh_date"),
    ("Blood pressure", "bp_result", "bp_unit", "bp_date"),
    ("Oxygen saturation", "spo2_result", "spo2_unit", "spo2_date"),
    ("Respiratory rate", "rr_result", "rr_unit", "rr_date"),
    ("Height", "height_result", "height_unit", "height_date"),
    ("Weight", "weight_result", "weight_unit", "weight_date"),
    ("BMI", "bmi_result", "bmi_unit", "bmi_date"),
]


def _build_clinical_observations(raw_patient: dict) -> list[dict]:
    observations: list[dict] = []

    for label, result_key, unit_key, *date_keys in CLINICAL_OBSERVATION_SPECS:
        result = raw_patient.get(result_key)
        if not _is_present(result):
            continue
        obs = {
            "label": label,
            "value": result,
        }
        unit = raw_patient.get(unit_key)
        if _is_present(unit):
            obs["unit"] = unit
        date_value = _pick_date(raw_patient, *date_keys)
        if _is_present(date_value):
            obs["date"] = date_value
        observations.append(obs)

    if _is_present(raw_patient.get("latest_medication_dosage")):
        med_obs = {
            "label": "Latest medication",
            "value": raw_patient.get("latest_medication_dosage"),
        }
        if _is_present(raw_patient.get("latest_medication_date")):
            med_obs["date"] = raw_patient.get("latest_medication_date")
        observations.append(med_obs)

    if _is_present(raw_patient.get("latest_diag_code")):
        cond_obs = {
            "label": "Latest diagnosis (ICD)",
            "value": raw_patient.get("latest_diag_code"),
        }
        if _is_present(raw_patient.get("latest_condition_date")):
            cond_obs["date"] = raw_patient.get("latest_condition_date")
        observations.append(cond_obs)

    return observations


def _calculate_age(dob_value) -> int | None:
    dob = parse_date(dob_value)
    if dob is None:
        return None

    today = date.today()
    age = today.year - dob.year
    if (today.month, today.day) < (dob.month, dob.day):
        age -= 1

    return age if age >= 0 else None


def _normalize_patient(raw_patient: dict) -> dict:
    systolic_bp = raw_patient.get("systolic_bp")
    diastolic_bp = raw_patient.get("diastolic_bp")
    if systolic_bp is None or diastolic_bp is None:
        parsed_systolic, parsed_diastolic = _parse_bp(raw_patient.get("bp_result"))
        if systolic_bp is None:
            systolic_bp = parsed_systolic
        if diastolic_bp is None:
            diastolic_bp = parsed_diastolic

    meds = raw_patient.get("medications")
    if meds is None:
        meds = _as_list(raw_patient.get("latest_medication_dosage"))
    else:
        meds = _as_list(meds)

    icd_codes = raw_patient.get("icd_codes")
    if icd_codes is None:
        icd_codes = _as_list(raw_patient.get("latest_diag_code"))
    else:
        icd_codes = _as_list(icd_codes)

    return {
        "patient_id": raw_patient.get("patient_id") or raw_patient.get("uniqueempi"),
        "egfr": safe_float(raw_patient.get("egfr", raw_patient.get("egfr_result"))),
        "creatinine": safe_float(raw_patient.get("creatinine", raw_patient.get("creatinine_result"))),
        "potassium": safe_float(raw_patient.get("potassium", raw_patient.get("potassium_result"))),
        "urine_acr": safe_float(raw_patient.get("urine_acr", raw_patient.get("acr_result"))),
        "systolic_bp": safe_float(systolic_bp),
        "diastolic_bp": safe_float(diastolic_bp),
        "egfr_date": raw_patient.get("egfr_date") or raw_patient.get("egfr_performed_date"),
        "creatinine_date": raw_patient.get("creatinine_date") or raw_patient.get("creatinine_performed_date"),
        "potassium_date": raw_patient.get("potassium_date") or raw_patient.get("potassium_performed_date"),
        "acr_date": raw_patient.get("acr_date") or raw_patient.get("acr_performed_date"),
        "bp_date": raw_patient.get("bp_date"),
        "medications": meds,
        "icd_codes": icd_codes,
    }


def build_patient_summary(patient: dict) -> dict:
    """
    Assembles enriched patient summary (labs, disease tags, CKD rules when applicable)
    for LLM prompts and retrieval tagging.
    """
    normalized_patient = _normalize_patient(patient)
    stage, acr_cat, flags = run_ckd_rules(normalized_patient)

    key_values = _drop_null_values({
        "egfr": normalized_patient.get("egfr"),
        "creatinine": normalized_patient.get("creatinine"),
        "potassium": normalized_patient.get("potassium"),
        "urine_acr": normalized_patient.get("urine_acr"),
        "systolic_bp": normalized_patient.get("systolic_bp"),
        "diastolic_bp": normalized_patient.get("diastolic_bp"),
    })
    test_dates = _drop_null_values({
        "acr_order_date": patient.get("acr_date"),
        "acr_performed_date": patient.get("acr_performed_date"),
        "creatinine_order_date": patient.get("creatinine_date"),
        "creatinine_performed_date": patient.get("creatinine_performed_date"),
        "egfr_order_date": patient.get("egfr_date"),
        "egfr_performed_date": patient.get("egfr_performed_date"),
        "potassium_order_date": patient.get("potassium_date"),
        "potassium_performed_date": patient.get("potassium_performed_date"),
        "bp_date": patient.get("bp_date"),
        "latest_condition_date": patient.get("latest_condition_date"),
        "latest_medication_date": patient.get("latest_medication_date"),
    })
    demographics = _drop_null_values({
        "uniqueempi": patient.get("uniqueempi"),
        "first_name": patient.get("memfirstname"),
        "last_name": patient.get("memlastname"),
        "gender": patient.get("gender"),
        "dob": patient.get("memdob"),
        "age": _calculate_age(patient.get("memdob")),
        "memberkey": patient.get("memberkey"),
    })

    icd_codes = normalized_patient.get("icd_codes", [])
    disease_tags = infer_patient_disease_tags(icd_codes, key_values, flags)
    comorbidity_tags = infer_patient_comorbidity_tags(icd_codes)

    return {
        "patient_id": normalized_patient.get("patient_id"),
        "demographics": demographics,
        "key_values": key_values,
        "clinical_observations": _build_clinical_observations(patient),
        "test_dates": test_dates,
        "medications": normalized_patient.get("medications", []),
        "icd_codes": icd_codes,
        "flags": flags,
        "stage": stage,
        "acr_category": acr_cat,
        "retrieval_context": {
            "disease": primary_disease_tag(disease_tags),
            "disease_tags": disease_tags,
            "comorbidity_tags": comorbidity_tags,
            "stage": stage,
            "acr_category": acr_cat,
        },
        # Keep all raw DB columns from the fetched row.
        "source_fields": patient,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build clinical patient summary using uniqueempi."
    )
    parser.add_argument("uniqueempi", help="Patient uniqueempi value")
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty print JSON output",
    )
    args = parser.parse_args()

    from data.patient_fetcher import fetch_patient_from_gold

    patient = fetch_patient_from_gold(args.uniqueempi)
    summary = build_patient_summary(patient)
    indent = 2 if args.pretty else None
    print(json.dumps(summary, indent=indent, default=str))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise