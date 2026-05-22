PATIENT_QUERY = """
WITH member_scope AS (
    SELECT memberkey
    FROM nhsdev.gold.member
    WHERE uniqueempi = ?
),
latest_labs AS (
    SELECT
        LAB_TEST_LOCAL_NAME,
        LAB_RESULT_VALUE,
        LAB_RESULT_UNIT,
        NORMAL_RESULT_RANGE,
        LAB_TEST_ORDER_DATE_KEY,
        LAB_RESULT_DATE_KEY,
        ROW_NUMBER() OVER (
            PARTITION BY LAB_TEST_LOCAL_NAME
            ORDER BY LAB_RESULT_DATE_KEY DESC
        ) AS rn
    FROM nhsdev.gold.lab
    WHERE member_key IN (SELECT memberkey FROM member_scope)
      AND LAB_TEST_LOCAL_NAME IN (
          'Album/creat ratio, urine',
          'Creatinine',
          'eGFR',
          'Potassium',
          'Aldosterone',
          'Renin',
          'Aldosterone/Renin Ratio',
          'Fasting Glucose',
          'Random Glucose',
          'HbA1c',
          'Total Cholesterol',
          'LDL',
          'HDL',
          'Triglycerides',
          'Troponin',
          'BNP',
          'NT-proBNP',
          'CRP',
          'hs-CRP',
          'Hemoglobin',
          'Sodium',
          'TSH'
      )
),
latest_vitals_bp AS (
    SELECT
        result           AS bp_result,
        unit             AS bp_unit,
        measurement_date AS bp_date,
        ROW_NUMBER() OVER (
            ORDER BY measurement_date DESC
        ) AS rn
    FROM nhsdev.gold.vitals
    WHERE member_key IN (SELECT memberkey FROM member_scope)
      AND vital_type = 'Blood pressure'
),
latest_vitals_spo2 AS (
    SELECT
        result           AS spo2_result,
        unit             AS spo2_unit,
        measurement_date AS spo2_date,
        ROW_NUMBER() OVER (
            ORDER BY measurement_date DESC
        ) AS rn
    FROM nhsdev.gold.vitals
    WHERE member_key IN (SELECT memberkey FROM member_scope)
      AND vital_type = 'Oxygen Saturation'
),
latest_vitals_rr AS (
    SELECT
        result           AS rr_result,
        unit             AS rr_unit,
        measurement_date AS rr_date,
        ROW_NUMBER() OVER (
            ORDER BY measurement_date DESC
        ) AS rn
    FROM nhsdev.gold.vitals
    WHERE member_key IN (SELECT memberkey FROM member_scope)
      AND vital_type = 'Respiratory Rate'
),
latest_vitals_height AS (
    SELECT
        result           AS height_result,
        unit             AS height_unit,
        measurement_date AS height_date,
        ROW_NUMBER() OVER (
            ORDER BY measurement_date DESC
        ) AS rn
    FROM nhsdev.gold.vitals
    WHERE member_key IN (SELECT memberkey FROM member_scope)
      AND vital_type = 'Height'
),
latest_vitals_weight AS (
    SELECT
        result           AS weight_result,
        unit             AS weight_unit,
        measurement_date AS weight_date,
        ROW_NUMBER() OVER (
            ORDER BY measurement_date DESC
        ) AS rn
    FROM nhsdev.gold.vitals
    WHERE member_key IN (SELECT memberkey FROM member_scope)
      AND vital_type = 'Weight'
),
latest_vitals_bmi AS (
    SELECT
        result           AS bmi_result,
        unit             AS bmi_unit,
        measurement_date AS bmi_date,
        ROW_NUMBER() OVER (
            ORDER BY measurement_date DESC
        ) AS rn
    FROM nhsdev.gold.vitals
    WHERE member_key IN (SELECT memberkey FROM member_scope)
      AND vital_type = 'BMI'
),
latest_medications AS (
    SELECT
        dosage_instruction,
        medication_request_date,
        ROW_NUMBER() OVER (
            ORDER BY medication_request_date DESC
        ) AS rn
    FROM nhsdev.gold.medication_request
    WHERE memberkey IN (SELECT memberkey FROM member_scope)
),
latest_conditions AS (
    SELECT
        icd.Diag_Code,
        c.condition_diagnosis_record_time,
        ROW_NUMBER() OVER (
            ORDER BY c.condition_diagnosis_record_time DESC
        ) AS rn
    FROM nhsdev.gold.condition c
    LEFT JOIN nhssource.dbo.rft_final_icd_diag icd
        ON c.DiagKey = icd.DiagKey
    WHERE c.member_key IN (SELECT memberkey FROM member_scope)
),
member_demographics AS (
    SELECT
        uniqueempi, memfirstname, memlastname, gender, memdob, memberkey,
        ROW_NUMBER() OVER (ORDER BY CREATEDDATE DESC) AS rn
    FROM nhsdev.gold.member
    WHERE uniqueempi = ?
)
SELECT
    md.uniqueempi,
    md.memfirstname,
    md.memlastname,
    md.gender,
    md.memdob,
    md.memberkey,

    acr.LAB_RESULT_VALUE        AS acr_result,
    acr.LAB_RESULT_UNIT         AS acr_unit,
    acr.NORMAL_RESULT_RANGE     AS acr_normal_range,
    acr.LAB_TEST_ORDER_DATE_KEY AS acr_date,
    acr.LAB_RESULT_DATE_KEY     AS acr_performed_date,

    cr.LAB_RESULT_VALUE         AS creatinine_result,
    cr.LAB_RESULT_UNIT          AS creatinine_unit,
    cr.NORMAL_RESULT_RANGE      AS creatinine_normal_range,
    cr.LAB_TEST_ORDER_DATE_KEY  AS creatinine_date,
    cr.LAB_RESULT_DATE_KEY      AS creatinine_performed_date,

    eg.LAB_RESULT_VALUE         AS egfr_result,
    eg.LAB_RESULT_UNIT          AS egfr_unit,
    eg.NORMAL_RESULT_RANGE      AS egfr_normal_range,
    eg.LAB_TEST_ORDER_DATE_KEY  AS egfr_date,
    eg.LAB_RESULT_DATE_KEY      AS egfr_performed_date,

    po.LAB_RESULT_VALUE         AS potassium_result,
    po.LAB_RESULT_UNIT          AS potassium_unit,
    po.NORMAL_RESULT_RANGE      AS potassium_normal_range,
    po.LAB_TEST_ORDER_DATE_KEY  AS potassium_date,
    po.LAB_RESULT_DATE_KEY      AS potassium_performed_date,

    ald.LAB_RESULT_VALUE        AS aldosterone_result,
    ald.LAB_RESULT_UNIT         AS aldosterone_unit,
    ald.NORMAL_RESULT_RANGE     AS aldosterone_normal_range,
    ald.LAB_TEST_ORDER_DATE_KEY AS aldosterone_date,
    ald.LAB_RESULT_DATE_KEY     AS aldosterone_performed_date,

    ren.LAB_RESULT_VALUE        AS renin_result,
    ren.LAB_RESULT_UNIT         AS renin_unit,
    ren.NORMAL_RESULT_RANGE     AS renin_normal_range,
    ren.LAB_TEST_ORDER_DATE_KEY AS renin_date,
    ren.LAB_RESULT_DATE_KEY     AS renin_performed_date,

    arr.LAB_RESULT_VALUE        AS aldosterone_renin_ratio_result,
    arr.LAB_RESULT_UNIT         AS aldosterone_renin_ratio_unit,
    arr.NORMAL_RESULT_RANGE     AS aldosterone_renin_ratio_normal_range,
    arr.LAB_TEST_ORDER_DATE_KEY AS aldosterone_renin_ratio_date,
    arr.LAB_RESULT_DATE_KEY     AS aldosterone_renin_ratio_performed_date,

    fg.LAB_RESULT_VALUE         AS fasting_glucose_result,
    fg.LAB_RESULT_UNIT          AS fasting_glucose_unit,
    fg.NORMAL_RESULT_RANGE      AS fasting_glucose_normal_range,
    fg.LAB_TEST_ORDER_DATE_KEY  AS fasting_glucose_date,
    fg.LAB_RESULT_DATE_KEY      AS fasting_glucose_performed_date,

    rg.LAB_RESULT_VALUE         AS random_glucose_result,
    rg.LAB_RESULT_UNIT          AS random_glucose_unit,
    rg.NORMAL_RESULT_RANGE      AS random_glucose_normal_range,
    rg.LAB_TEST_ORDER_DATE_KEY  AS random_glucose_date,
    rg.LAB_RESULT_DATE_KEY      AS random_glucose_performed_date,

    hba.LAB_RESULT_VALUE        AS hba1c_result,
    hba.LAB_RESULT_UNIT         AS hba1c_unit,
    hba.NORMAL_RESULT_RANGE     AS hba1c_normal_range,
    hba.LAB_TEST_ORDER_DATE_KEY AS hba1c_date,
    hba.LAB_RESULT_DATE_KEY     AS hba1c_performed_date,

    tc.LAB_RESULT_VALUE         AS total_cholesterol_result,
    tc.LAB_RESULT_UNIT          AS total_cholesterol_unit,
    tc.NORMAL_RESULT_RANGE      AS total_cholesterol_normal_range,
    tc.LAB_TEST_ORDER_DATE_KEY  AS total_cholesterol_date,
    tc.LAB_RESULT_DATE_KEY      AS total_cholesterol_performed_date,

    ldl.LAB_RESULT_VALUE        AS ldl_result,
    ldl.LAB_RESULT_UNIT         AS ldl_unit,
    ldl.NORMAL_RESULT_RANGE     AS ldl_normal_range,
    ldl.LAB_TEST_ORDER_DATE_KEY AS ldl_date,
    ldl.LAB_RESULT_DATE_KEY     AS ldl_performed_date,

    hdl.LAB_RESULT_VALUE        AS hdl_result,
    hdl.LAB_RESULT_UNIT         AS hdl_unit,
    hdl.NORMAL_RESULT_RANGE     AS hdl_normal_range,
    hdl.LAB_TEST_ORDER_DATE_KEY AS hdl_date,
    hdl.LAB_RESULT_DATE_KEY     AS hdl_performed_date,

    tg.LAB_RESULT_VALUE         AS triglycerides_result,
    tg.LAB_RESULT_UNIT          AS triglycerides_unit,
    tg.NORMAL_RESULT_RANGE      AS triglycerides_normal_range,
    tg.LAB_TEST_ORDER_DATE_KEY  AS triglycerides_date,
    tg.LAB_RESULT_DATE_KEY      AS triglycerides_performed_date,

    tro.LAB_RESULT_VALUE        AS troponin_result,
    tro.LAB_RESULT_UNIT         AS troponin_unit,
    tro.NORMAL_RESULT_RANGE     AS troponin_normal_range,
    tro.LAB_TEST_ORDER_DATE_KEY AS troponin_date,
    tro.LAB_RESULT_DATE_KEY     AS troponin_performed_date,

    bnp.LAB_RESULT_VALUE        AS bnp_result,
    bnp.LAB_RESULT_UNIT         AS bnp_unit,
    bnp.NORMAL_RESULT_RANGE     AS bnp_normal_range,
    bnp.LAB_TEST_ORDER_DATE_KEY AS bnp_date,
    bnp.LAB_RESULT_DATE_KEY     AS bnp_performed_date,

    ntbnp.LAB_RESULT_VALUE      AS nt_pro_bnp_result,
    ntbnp.LAB_RESULT_UNIT       AS nt_pro_bnp_unit,
    ntbnp.NORMAL_RESULT_RANGE   AS nt_pro_bnp_normal_range,
    ntbnp.LAB_TEST_ORDER_DATE_KEY AS nt_pro_bnp_date,
    ntbnp.LAB_RESULT_DATE_KEY   AS nt_pro_bnp_performed_date,

    crp.LAB_RESULT_VALUE        AS crp_result,
    crp.LAB_RESULT_UNIT         AS crp_unit,
    crp.NORMAL_RESULT_RANGE     AS crp_normal_range,
    crp.LAB_TEST_ORDER_DATE_KEY AS crp_date,
    crp.LAB_RESULT_DATE_KEY     AS crp_performed_date,

    hscrp.LAB_RESULT_VALUE      AS hs_crp_result,
    hscrp.LAB_RESULT_UNIT       AS hs_crp_unit,
    hscrp.NORMAL_RESULT_RANGE   AS hs_crp_normal_range,
    hscrp.LAB_TEST_ORDER_DATE_KEY AS hs_crp_date,
    hscrp.LAB_RESULT_DATE_KEY   AS hs_crp_performed_date,

    hb.LAB_RESULT_VALUE         AS hemoglobin_result,
    hb.LAB_RESULT_UNIT          AS hemoglobin_unit,
    hb.NORMAL_RESULT_RANGE      AS hemoglobin_normal_range,
    hb.LAB_TEST_ORDER_DATE_KEY  AS hemoglobin_date,
    hb.LAB_RESULT_DATE_KEY      AS hemoglobin_performed_date,

    na.LAB_RESULT_VALUE         AS sodium_result,
    na.LAB_RESULT_UNIT          AS sodium_unit,
    na.NORMAL_RESULT_RANGE      AS sodium_normal_range,
    na.LAB_TEST_ORDER_DATE_KEY  AS sodium_date,
    na.LAB_RESULT_DATE_KEY      AS sodium_performed_date,

    tsh.LAB_RESULT_VALUE        AS tsh_result,
    tsh.LAB_RESULT_UNIT         AS tsh_unit,
    tsh.NORMAL_RESULT_RANGE     AS tsh_normal_range,
    tsh.LAB_TEST_ORDER_DATE_KEY AS tsh_date,
    tsh.LAB_RESULT_DATE_KEY     AS tsh_performed_date,

    lv_bp.bp_result,
    lv_bp.bp_unit,
    lv_bp.bp_date,

    lv_spo2.spo2_result,
    lv_spo2.spo2_unit,
    lv_spo2.spo2_date,

    lv_rr.rr_result,
    lv_rr.rr_unit,
    lv_rr.rr_date,

    lv_height.height_result,
    lv_height.height_unit,
    lv_height.height_date,

    lv_weight.weight_result,
    lv_weight.weight_unit,
    lv_weight.weight_date,

    lv_bmi.bmi_result,
    lv_bmi.bmi_unit,
    lv_bmi.bmi_date,

    lm.dosage_instruction       AS latest_medication_dosage,
    lm.medication_request_date  AS latest_medication_date,

    lc.Diag_Code                AS latest_diag_code,
    lc.condition_diagnosis_record_time AS latest_condition_date

FROM member_demographics md
LEFT JOIN latest_labs acr ON acr.LAB_TEST_LOCAL_NAME = 'Album/creat ratio, urine' AND acr.rn = 1
LEFT JOIN latest_labs cr  ON cr.LAB_TEST_LOCAL_NAME  = 'Creatinine'               AND cr.rn = 1
LEFT JOIN latest_labs eg  ON eg.LAB_TEST_LOCAL_NAME  = 'eGFR'                     AND eg.rn = 1
LEFT JOIN latest_labs po  ON po.LAB_TEST_LOCAL_NAME  = 'Potassium'                AND po.rn = 1
LEFT JOIN latest_labs ald ON ald.LAB_TEST_LOCAL_NAME = 'Aldosterone'              AND ald.rn = 1
LEFT JOIN latest_labs ren ON ren.LAB_TEST_LOCAL_NAME = 'Renin'                    AND ren.rn = 1
LEFT JOIN latest_labs arr ON arr.LAB_TEST_LOCAL_NAME = 'Aldosterone/Renin Ratio'  AND arr.rn = 1
LEFT JOIN latest_labs fg  ON fg.LAB_TEST_LOCAL_NAME  = 'Fasting Glucose'          AND fg.rn = 1
LEFT JOIN latest_labs rg  ON rg.LAB_TEST_LOCAL_NAME  = 'Random Glucose'           AND rg.rn = 1
LEFT JOIN latest_labs hba ON hba.LAB_TEST_LOCAL_NAME = 'HbA1c'                    AND hba.rn = 1
LEFT JOIN latest_labs tc  ON tc.LAB_TEST_LOCAL_NAME  = 'Total Cholesterol'        AND tc.rn = 1
LEFT JOIN latest_labs ldl ON ldl.LAB_TEST_LOCAL_NAME = 'LDL'                      AND ldl.rn = 1
LEFT JOIN latest_labs hdl ON hdl.LAB_TEST_LOCAL_NAME = 'HDL'                      AND hdl.rn = 1
LEFT JOIN latest_labs tg  ON tg.LAB_TEST_LOCAL_NAME  = 'Triglycerides'            AND tg.rn = 1
LEFT JOIN latest_labs tro ON tro.LAB_TEST_LOCAL_NAME = 'Troponin'                 AND tro.rn = 1
LEFT JOIN latest_labs bnp ON bnp.LAB_TEST_LOCAL_NAME = 'BNP'                      AND bnp.rn = 1
LEFT JOIN latest_labs ntbnp ON ntbnp.LAB_TEST_LOCAL_NAME = 'NT-proBNP'            AND ntbnp.rn = 1
LEFT JOIN latest_labs crp ON crp.LAB_TEST_LOCAL_NAME = 'CRP'                      AND crp.rn = 1
LEFT JOIN latest_labs hscrp ON hscrp.LAB_TEST_LOCAL_NAME = 'hs-CRP'               AND hscrp.rn = 1
LEFT JOIN latest_labs hb  ON hb.LAB_TEST_LOCAL_NAME  = 'Hemoglobin'               AND hb.rn = 1
LEFT JOIN latest_labs na  ON na.LAB_TEST_LOCAL_NAME  = 'Sodium'                   AND na.rn = 1
LEFT JOIN latest_labs tsh ON tsh.LAB_TEST_LOCAL_NAME = 'TSH'                      AND tsh.rn = 1
LEFT JOIN latest_vitals_bp lv_bp      ON lv_bp.rn = 1
LEFT JOIN latest_vitals_spo2 lv_spo2  ON lv_spo2.rn = 1
LEFT JOIN latest_vitals_rr lv_rr      ON lv_rr.rn = 1
LEFT JOIN latest_vitals_height lv_height ON lv_height.rn = 1
LEFT JOIN latest_vitals_weight lv_weight ON lv_weight.rn = 1
LEFT JOIN latest_vitals_bmi lv_bmi    ON lv_bmi.rn = 1
LEFT JOIN latest_medications lm       ON lm.rn = 1
LEFT JOIN latest_conditions lc        ON lc.rn = 1

WHERE md.rn = 1
"""
