# Cohort files

This folder does not contain data from Hospital Universitario de Fuenlabrada.
The cultures behind the paper come from the ICU electronic health record.
The hospital asked that the file, including transformed tables, stay private.
Dates together with age, sex and admission details can re-identify a patient
even after the name has been removed.

`synthetic/cohort.csv` is an invented cohort with the same columns. It is
there so the code can be read and executed. Its predictions are not the
numbers in the paper.

## Columns

One row is one culture.

| Column | Meaning |
| --- | --- |
| `patient_id`, `culture_id`, `admission_id` | Identifiers. They are used for the split and then dropped. |
| `culture_date`, `admission_start`, `admission_end` | Timestamps. |
| `age`, `gender` | Demographics. |
| `origin`, `reason_admission`, `patient_category` | Admission context. |
| `goi_A` … `goi_G` | Comorbidity groups, 0 or 1. Pluripathology is their sum. |
| `culture_type`, `culture_type_group1`, `culture_type_group2` | Sample type and the two groupings used in the study. |
| `c&amg`, `c&car`, `c&cf4`, `c&pap`, `c&pol`, `c&qui` | Antibiogram. `r` resistant, `s` susceptible, empty if that family was not tested. |

The package builds the calendar fields, the stay length, and the past-resistance
features `p&` and `r&`. A local file with this header can be passed to
`python -m amr_icu --data your_cohort.csv`.

Put any real extract under `data/private/`. That directory is gitignored.
