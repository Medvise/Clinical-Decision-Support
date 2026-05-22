"""
Canonical tag vocabularies for Qdrant payloads, ingestion enrichment, and retrieval filters.

Runtime filter validation uses ALLOWED_* sets; keyword maps drive query/patient inference
and PDF chunk tagging. Keep tag strings stable — they must match ingested payload values.
"""

from __future__ import annotations

# ── Retrieval filter allowlists (LLM tagger + Qdrant must filters) ─────────────

ALLOWED_DISEASE_TAGS = frozenset({"CKD", "hypertension", "ACHD"})
ALLOWED_CKD_STAGE_TAGS = frozenset({"G1", "G2", "G3", "G3a", "G3b", "G4", "G5", "A1", "A2", "A3"})
ALLOWED_BP_STAGE_TAGS = frozenset({"normal", "elevated", "stage1", "stage2", "severe"})
ALLOWED_COMORBIDITY_TAGS = frozenset({
    "diabetes",
    "heart_failure",
    "coronary",
    "pregnancy",
    "atrial_fibrillation",
    "stroke",
    "obesity",
    "PAH",
    "cyanosis",
    "CKD",
})
ALLOWED_DRUG_TAGS = frozenset({
    "ACEi", "ARB", "SGLT2i", "MRA", "CCB", "thiazide", "betablocker", "GLP1", "statin",
})

DEFAULT_CHUNK_TYPES = [
    "recommendation",
    "practice_point",
    "rationale",
    "synopsis",
    "supportive_text",
]

# ── Query keyword inference (general + patient routes) ─────────────────────────

QUERY_DISEASE_KEYWORDS: dict[str, list[str]] = {
    "CKD": [
        "ckd", "chronic kidney", "kidney disease", "egfr", "albuminuria",
        "kdigo", "nice", "nephro",
    ],
    "hypertension": [
        "hypertension", "blood pressure", " bp ", "htn", "antihypertensive",
    ],
    "ACHD": [
        "achd", "congenital heart", "fontan", "cyanotic heart",
    ],
}

QUERY_COMORBIDITY_KEYWORDS: dict[str, list[str]] = {
    "diabetes": ["diabetes", "diabetic", "glycemic", "hba1c", "t2d"],
    "heart_failure": ["heart failure", "hfref", "hfpef", "bnp", "nt-probnp"],
    "pregnancy": ["pregnant", "pregnancy", "gestational", "preeclampsia"],
    "coronary": ["coronary", "cad", "acs", "myocardial infarction"],
}

# ── Flag text → drug class (patient rule flags) ────────────────────────────────

DRUG_FLAG_KEYWORDS: dict[str, list[str]] = {
    "ACEi": [
        "ramipril", "lisinopril", "enalapril", "perindopril",
        "captopril", "fosinopril", "ace inhibitor", "acei",
    ],
    "ARB": [
        "losartan", "valsartan", "candesartan", "irbesartan",
        "telmisartan", "arb",
    ],
    "SGLT2i": ["dapagliflozin", "empagliflozin", "canagliflozin", "sotagliflozin", "sglt2"],
    "MRA": ["spironolactone", "eplerenone", "finerenone", "mra"],
    "statin": ["atorvastatin", "rosuvastatin", "simvastatin", "statin"],
}

# ── Ingestion chunk enrichment (superset of ALLOWED_DRUG_TAGS) ─────────────────

DRUG_VOCAB: dict[str, list[str]] = {
    "ACEi": [
        "ace inhibitor", "acei", "lisinopril", "enalapril", "ramipril",
        "perindopril", "captopril", "benazepril", "quinapril",
    ],
    "ARB": [
        "arb", "angiotensin receptor blocker", "losartan", "valsartan",
        "candesartan", "irbesartan", "olmesartan", "telmisartan",
    ],
    "SGLT2i": [
        "sglt2", "dapagliflozin", "empagliflozin", "canagliflozin",
        "ertugliflozin",
    ],
    "MRA": [
        "mra", "mineralocorticoid receptor antagonist", "spironolactone",
        "eplerenone", "finerenone",
    ],
    "CCB": [
        "calcium channel blocker", "ccb", "amlodipine", "nifedipine",
        "felodipine", "diltiazem", "verapamil",
    ],
    "thiazide": [
        "thiazide", "hydrochlorothiazide", "hctz", "chlorthalidone",
        "indapamide",
    ],
    "betablocker": [
        "beta-blocker", "beta blocker", "metoprolol", "carvedilol",
        "bisoprolol", "atenolol", "nebivolol",
    ],
    "GLP1": [
        "glp-1", "glp1", "semaglutide", "liraglutide", "dulaglutide",
        "exenatide", "tirzepatide",
    ],
    "statin": [
        "statin", "atorvastatin", "rosuvastatin", "simvastatin",
        "pravastatin", "lovastatin",
    ],
    "diuretic": ["diuretic", "furosemide", "torsemide", "bumetanide"],
    "anticoag": [
        "anticoagulant", "warfarin", "apixaban", "rivaroxaban",
        "dabigatran", "edoxaban", "heparin",
    ],
    "antidiab": [
        "metformin", "insulin", "sulfonylurea", "glipizide", "glimepiride",
        "sitagliptin", "dpp-4",
    ],
}

COMORBIDITY_VOCAB: dict[str, list[str]] = {
    "diabetes": [
        "diabetes", "t2d", "t1d", "glycemic", "hba1c",
        "hyperglycemia", "hypoglycemia",
    ],
    "CKD": [
        "ckd", "chronic kidney disease", "egfr", "albuminuria",
        "proteinuria", "renal insufficiency", "kidney disease",
    ],
    "atrial_fibrillation": ["atrial fibrillation", "afib", "af ", "a-fib"],
    "heart_failure": [
        "heart failure", "hfref", "hfpef", "hfmref",
        "ejection fraction", "cardiac failure", "lvef",
    ],
    "stroke": ["stroke", "cerebrovascular", "tia", "transient ischemic"],
    "pregnancy": [
        "pregnant", "pregnancy", "gestational", "preeclampsia",
        "eclampsia", "obstetric",
    ],
    "obesity": ["obesity", "obese", "bmi", "overweight", "adiposity"],
    "coronary": [
        "coronary", "cad", "angina", "myocardial infarction",
        "acs", "acute coronary", "pcsk9", "atherosclerosis",
    ],
    "PAH": [
        "pulmonary hypertension", "pah", "pulmonary arterial",
        "pulmonary vascular",
    ],
    "cyanosis": ["cyanosis", "cyanotic", "oxygen saturation", "hypoxemia"],
    "anemia": [
        "anemia", "anaemia", "hemoglobin", "erythropoiesis",
        "esa", "iron deficiency",
    ],
    "metabolic_acidosis": ["metabolic acidosis", "bicarbonate", "serum bicarbonate"],
    "hyperkalemia": ["hyperkalemia", "potassium", "hyperkalaemia"],
    "dyslipidemia": [
        "dyslipidemia", "hyperlipidemia", "ldl", "hdl",
        "cholesterol", "triglyceride",
    ],
}

BP_STAGE_VOCAB: dict[str, list[str]] = {
    "normal": ["normal bp", "normal blood pressure", "<120"],
    "elevated": [
        "elevated bp", "elevated blood pressure", "120 to 129", "prehypertension",
    ],
    "stage1": ["stage 1 hypertension", "stage 1 htn", "130 to 139", "80 to 89"],
    "stage2": ["stage 2 hypertension", "stage 2 htn", "≥140", "≥90"],
    "severe": [
        "severe hypertension", ">180", "hypertensive emergency",
        "hypertensive urgency",
    ],
}

CKD_STAGE_VOCAB: dict[str, list[str]] = {
    "G1": ["g1", "gfr category 1", "gfr ≥90", "stage 1 ckd"],
    "G2": ["g2", "gfr category 2", "60 to 89", "stage 2 ckd"],
    "G3a": ["g3a", "45 to 59"],
    "G3b": ["g3b", "30 to 44"],
    "G3": ["g3 ", "gfr category 3", "stage 3 ckd"],
    "G4": ["g4", "gfr category 4", "15 to 29", "stage 4 ckd"],
    "G5": [
        "g5", "gfr category 5", "gfr <15", "stage 5 ckd", "esrd", "esrf",
        "kidney failure", "dialysis",
    ],
    "A1": ["a1", "acr <30", "normal to mildly increased"],
    "A2": ["a2", "acr 30", "moderately increased"],
    "A3": ["a3", "acr >300", "severely increased", "nephrotic"],
}

INTERVENTION_VOCAB: dict[str, list[str]] = {
    "closure": ["closure", "device closure", "transcatheter closure"],
    "surgery": [
        "surgery", "surgical repair", "open heart", "sternotomy",
        "cardiopulmonary bypass",
    ],
    "catheterization": [
        "catheterization", "catheter", "percutaneous", "intervention",
        "cath lab",
    ],
    "transplant": [
        "transplant", "transplantation", "heart transplant",
        "cardiac transplant",
    ],
    "pacemaker": [
        "pacemaker", "icd", "crt", "defibrillator", "ep study",
        "electrophysiology",
    ],
    "valvuloplasty": ["valvuloplasty", "balloon dilation", "valvotomy"],
    "fontan": ["fontan", "fontan circulation", "single ventricle"],
}

CROSS_DISEASE_MAP: dict[str, tuple[str, list[str]]] = {
    "HTN_CKD": ("HTN", ["ckd", "chronic kidney", "egfr", "albuminuria"]),
    "HTN_diabetes": ("HTN", ["diabetes", "t2d", "glycemic"]),
    "HTN_HF": ("HTN", ["heart failure", "hfref", "hfpef"]),
    "HTN_pregnancy": ("HTN", ["pregnant", "pregnancy", "gestational"]),
    "KDIGO_HTN": ("KDIGO", ["hypertension", "blood pressure", "bp control"]),
    "KDIGO_diabetes": ("KDIGO", ["diabetes", "glycemic", "hba1c"]),
    "KDIGO_CVD": ("KDIGO", ["cardiovascular", "coronary", "heart failure"]),
    "ACHD_HF": ("ACHD", ["heart failure", "hfref", "hfpef", "ejection fraction"]),
    "ACHD_pregnancy": ("ACHD", ["pregnant", "pregnancy", "obstetric"]),
}
