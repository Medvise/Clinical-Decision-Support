"""Fetch patient rows from Databricks gold tables."""

from data.databricks_client import get_connection
from data.queries.patient_gold import PATIENT_QUERY


def fetch_patient_from_gold(uniqueempi: str) -> dict:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(PATIENT_QUERY, [uniqueempi, uniqueempi])
            row = cur.fetchone()
            if row is None:
                raise ValueError(
                    f"Patient {uniqueempi} not found in Gold table"
                )
            cols = [d[0] for d in cur.description]
            patient = dict(zip(cols, row))

            meds = patient.get("medications", [])
            if isinstance(meds, str):
                patient["medications"] = [
                    m.strip() for m in meds.split(",") if m.strip()
                ]

            icds = patient.get("icd_codes", [])
            if isinstance(icds, str):
                patient["icd_codes"] = [
                    i.strip() for i in icds.split(",") if i.strip()
                ]

            return patient
