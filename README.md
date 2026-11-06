# Antimicrobial resistance prediction in the ICU

[![DOI](https://img.shields.io/badge/DOI-10.9781%2Fijimai.2021.02.012-1F6F8B)](https://doi.org/10.9781/ijimai.2021.02.012)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-2E7D4F)](LICENSE)

Predict whether *Pseudomonas aeruginosa* cultured in an intensive care unit will be resistant to an antibiotic **before the antibiogram comes back**. The antibiogram takes about 48 hours. This repository is the code companion to the study, cleaned down to the experiment that was published.

> Hernández-Carnerero, À., Sànchez-Marrè, M., Mora-Jiménez, I., Soguero-Ruiz, C., Martínez-Agüero, S., & Álvarez-Rodríguez, J. (2021). *Antimicrobial Resistance Prediction in Intensive Care Unit for Pseudomonas aeruginosa using Temporal Data-Driven Models.* International Journal of Interactive Multimedia and Artificial Intelligence. [https://doi.org/10.9781/ijimai.2021.02.012](https://doi.org/10.9781/ijimai.2021.02.012)

The cohort is the ICU of Hospital Universitario de Fuenlabrada: 3,346 patients, 3,812 admissions, July 2004 to May 2019. Of 43,658 cultures, 764 grew *Pseudomonas aeruginosa*. Each of six antibiotic families is its own classifier.

| | Family |
| --- | --- |
| AMG | Aminoglycosides |
| CAR | Carbapenems |
| CF4 | 4th generation cephalosporins |
| PAP | Extended-spectrum penicillins |
| POL | Polymyxins |
| QUI | Quinolones |

<p align="center">
  <img src="assets/figures/windows.gif" alt="A four-year training window and a growing training window both move toward 2019, always leaving the next month for testing." width="820">
</p>

## Why the calendar matters

Resistance is not a fixed label. Across these 15 years the share of resistant *Pseudomonas* cultures rises, then breaks in 2013 when the ICU record system was replaced. A model trained on the early, mostly susceptible years will keep calling later cultures susceptible.

<p align="center">
  <img src="assets/figures/resistance_over_time.png" alt="Percentage of resistant Pseudomonas cultures per antibiotic family from 2004 to 2019." width="640">
</p>

Training therefore never sees the future. The test is always the next month, from January 2016 onward. Two ways of choosing the past are compared.

<p align="center">
  <img src="assets/figures/windows.png" alt="Four training geometries: a four-year window next to the test, two years back, four years back, and a window that grows from 2004." width="820">
</p>

A patient who appears in the test month is removed from training. When training sits against the test month, the last 48 hours are left out as well: that antibiogram is not back yet.

The original sketch from the paper is kept in [`assets/figures/paper_windows.png`](assets/figures/paper_windows.png). Read it for the geometry. Models M1–M14 also change which past-resistance features are used, which the sketch does not spell out. The mapping is below.

## What the model is allowed to remember

Demographic, admission and culture fields describe the current sample. Two constructed fields describe the recent past. Both look at *Pseudomonas* cultures between 21 days and 48 hours before the culture being predicted. A resistant culture contributes \(e^{-\lambda \Delta t}\) with \(\lambda = 0.095\). A susceptible culture contributes nothing.

$$
p\&a_i = \max_k \mathrm{NEF}(c_k, c)
\qquad
r\&a_i = \sum_{j \neq p} \max_k \mathrm{NEF}(c_{j,k}, c)
$$

`p&` is the patient's own strongest recent resistance. `r&` adds that quantity across the other patients who were in the unit. If the window contains no test for that family, the value is set to 0: in this study a missing test was read as no evidence of resistance.

Mutual information ranks the culture date first, and the ICU-wide features look important. A random forest, which sees the columns together, tells a different story. The patient's own recent result is the feature that actually moves the prediction. The culture date mostly leaks the fact that later years are more resistant, and it hurts the susceptible class.

<p align="center">
  <img src="assets/figures/mutual_information.png" alt="Top mutual-information features. The culture date is largest for every family." width="720">
  <br>
  <img src="assets/figures/random_forest_importance.png" alt="Top random-forest features. The patient's own past result leads for five of the six families." width="720">
</p>

## The fourteen models

Logistic regression is the baseline. Random forest is the second classifier. Classes are not rebalanced. Oversampling was left as later work, because the class ratio is part of the drift.

| Models | Training window | Past cultures used |
| --- | --- | --- |
| M1, M2, M3 | 4 years, placed 0 / 2 / 4 years before the test | neither |
| M4, M5, M6 | 4 years, placed 0 / 2 / 4 years before the test | ICU patients only (`r&`) |
| M7, M8, M9 | 4 years, placed 0 / 2 / 4 years before the test | own patient and ICU (`p&` and `r&`) |
| M10, M11, M12 | Incremental, from the first culture | neither / ICU only / both |
| M13, M14 | Incremental, with a time decay on each training row | ICU only / both |

Each model is run with and without `date_culture`. For M13 and M14 the decay is chosen from \(0, 10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}, 0.1, 1\). A decay of 0 gives every row the same weight.

Accuracy is accumulated across months, not averaged. A month with two cultures must not count as much as a month with twenty. Three numbers are reported: total, resistant, and susceptible.

## Published results

These are the best weighted incremental models in the paper, as total / resistant / susceptible accuracy. The full grid is in [`results/published_accuracy.csv`](results/published_accuracy.csv). The decay chosen for each run is in [`results/published_lambda.csv`](results/published_lambda.csv).

| Family | Model | Classifier | Culture date | Accuracy |
| --- | --- | --- | --- | --- |
| AMG | M14 | Random forest | no | **79.55 / 75.76 / 90.91** |
| CAR | M13 | Logistic regression | yes or no | **94.32 / 100 / 16.67** |
| CF4 | M14 | Random forest | no | **60.67 / 58.62 / 64.52** |
| PAP | M14 | Logistic regression | no | **71.91 / 69.81 / 75.00** |
| POL | M13 | Logistic regression | yes | **72.41 / 60.00 / 78.95** |
| QUI | M14 | Random forest | no | **83.02 / 83.33 / 80.00** |

CAR looks solved because, by the test years, almost every culture is resistant. The susceptible accuracy gives that away. POL is the other extreme, with very few resistant cultures, and its best result in the paper is not the weighted window: a 4-year window sitting next to the test month, logistic regression, no culture date and neither past feature, reaches 74.71 / 63.33 / 80.70.

Using only the patient's own past result (`p&`) is already strong for several families. Using only the culture date pushes logistic regression onto the majority class. Both tables are in [`results/published_single_feature.csv`](results/published_single_feature.csv).

## Run it

```bash
pip install -e ".[dev]"
python -m amr_icu --synthetic
```

The default command trains logistic regression and a random forest for three of the published models on the invented cohort. It takes a few minutes. The printed accuracies are a smoke test. They are not the table above.

```bash
python -m amr_icu --synthetic --family all --models all --date-culture both
python -m amr_icu --data path/to/cohort.csv --family qui --models M14 --methods rf
python -m amr_icu --synthetic --importance
pytest
```

A real cohort uses the columns in [`data/README.md`](data/README.md). One row is one culture. Resistance cells are `r`, `s`, or empty when that family was not tested. The package derives the calendar fields and the past-resistance features.

```text
src/amr_icu/
  features.py      p& and r&, the negative exponential
  preprocess.py    48-hour stays, integer codes, correlated dates removed
  windows.py       monthly test, sliding and incremental training, patient split
  models.py        logistic regression (standardized) and random forest
  metrics.py       accuracies accumulated across months
  experiment.py    M1–M14 and the decay search
  synthetic.py     invented cohort
```

The original experiment lived in a set of notebooks, copied once per antibiotic family, with SMOTE and an LSTM tried along the way. Those runs are not part of the published comparison, so they are not in this package. This code follows the paper's definitions: a 48-hour margin, \(\lambda = 0.095\), and calendar windows from January 2016. It will not reproduce the notebooks bitwise. The notebooks stepped by month index from the first culture and, in some revisions, called an oversampler the paper does not use.

## Data

No file from the hospital is in this repository. The extract is anonymized, and it still contains dates, age, sex, origin and reason for admission. That is enough to re-identify a stay, and the hospital asked for it to remain private. [`data/synthetic/cohort.csv`](data/synthetic/cohort.csv) has the same columns and none of the patients. Metrics from it must not be compared with the paper.

## Citation

```bibtex
@article{hernandez2021amr,
  title   = {Antimicrobial Resistance Prediction in Intensive Care Unit for {Pseudomonas} aeruginosa using Temporal Data-Driven Models},
  author  = {Hern{\'a}ndez-Carnerero, {\`A}lvar and S{\`a}nchez-Marr{\`e}, Miquel and Mora-Jim{\'e}nez, Inmaculada and Soguero-Ruiz, Cristina and Mart{\'\i}nez-Ag{\"u}ero, Sergio and {\'A}lvarez-Rodr{\'\i}guez, Joaqu{\'\i}n},
  journal = {International Journal of Interactive Multimedia and Artificial Intelligence},
  year    = {2021},
  doi     = {10.9781/ijimai.2021.02.012}
}
```

A machine-readable citation is in [`CITATION.cff`](CITATION.cff). Figures from the study are reproduced in `assets/figures/` and belong with the paper.
