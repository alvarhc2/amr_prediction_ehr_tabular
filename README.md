# Antimicrobial resistance prediction in the ICU

[![DOI](https://img.shields.io/badge/DOI-10.9781%2Fijimai.2021.02.012-1F6F8B)](https://doi.org/10.9781/ijimai.2021.02.012)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-2E7D4F)](LICENSE)

**Antimicrobial resistance** arises when a microorganism no longer responds to a drug that previously inhibited it. In an **intensive care unit (ICU)** the consequences are immediate: patients are already critically ill, and the usual laboratory test, the **antibiogram**, takes about **48 hours** to report whether the isolated bacterium is **resistant** or **susceptible**. During that interval the antibiotic treatment has to be chosen without the laboratory result.

The objective of this work is to predict, for a culture of ***Pseudomonas aeruginosa***, whether the bacterium will be resistant or susceptible to a given **antibiotic family**, using only information available before the antibiogram result. The study uses **electronic health records (EHR)** already stored by the hospital, together with the results of earlier antibiograms.

The cohort comes from the ICU of **Hospital Universitario de Fuenlabrada** (Madrid, Spain): **3,346 patients** and **3,812 admissions**, from July 2004 to May 2019. Of **43,658 cultures**, the study retains the **764** in which *Pseudomonas aeruginosa* was isolated.

Resistance is predicted separately for each family, with one **binary classifier** per family. The six families considered in the study are the following.

| Code | Antibiotic family |
| --- | --- |
| AMG | Aminoglycosides |
| CAR | Carbapenems |
| CF4 | 4th generation cephalosporins |
| PAP | Extended-spectrum penicillins |
| POL | Polymyxins |
| QUI | Quinolones |

The prediction is repeated month by month. Two constructions of the training set are compared. In a **sliding window**, training uses a fixed period of four years placed before the month being predicted. In an **incremental window**, training starts at the beginning of the cohort and grows by one month at every step, always stopping before the month being predicted. The animation below shows these two constructions. The orange mark is the test month.

**Training and test windows considered in the study: sliding and incremental**

<p align="center">
  <img src="assets/figures/windows.gif" alt="A four-year sliding training window and a growing incremental training window both advance toward 2019. The following month is left for testing." width="820">
</p>

This repository contains the code that corresponds to the published experiment.

> Hernández-Carnerero, À., Sànchez-Marrè, M., Mora-Jiménez, I., Soguero-Ruiz, C., Martínez-Agüero, S., & Álvarez-Rodríguez, J. (2021). *Antimicrobial Resistance Prediction in Intensive Care Unit for Pseudomonas aeruginosa using Temporal Data-Driven Models.* International Journal of Interactive Multimedia and Artificial Intelligence. [https://doi.org/10.9781/ijimai.2021.02.012](https://doi.org/10.9781/ijimai.2021.02.012)

## Evolution of resistance over time

The proportion of resistant cultures changes over the 15 years of the cohort. In the earlier years, susceptible results are the majority for most families. In the later years, resistant results become more frequent. The series also falls in 2013, when the ICU information system was replaced and fewer antibiograms were recorded. Polymyxins remain an exception: resistant results stay uncommon. Carbapenems move in the opposite direction and are resistant in nearly every culture by the end of the period.

<p align="center">
  <img src="assets/figures/resistance_over_time.png" alt="Percentage of resistant Pseudomonas aeruginosa cultures per antibiotic family, from 2004 to 2019." width="640">
</p>

A classifier trained on the early years therefore learns a period in which **susceptibility** predominates. Applied to the later years, it continues to predict susceptibility and misses a large part of the **resistant** cultures. The order of the cultures is also fixed by what is known at prediction time: only antibiograms **already collected** can be used. Cultures are sorted by **date**, and each culture is predicted only from those that precede it.

The **test set** is one calendar month. The first test month is **January 2016**, and the window then advances one month at a time, without overlap, until the end of the cohort. The training set that precedes each test month is built in one of the 
two ways shown above. The **sliding window** is also placed at three distances from the test month, so that the effect of 
predicting from an older period can be measured.

<p align="center">
  <img src="assets/figures/windows.png" alt="Four training schemes for a test month in 2016: a four-year window immediately before the test, the same window two years earlier, the same window four years earlier, and an incremental window from 2004." width="820">
</p>

In panel (a) the four-year training window ends just before the test month. In panels (b) and (c) the same four-year window is moved back by two and four years. In panel (d) training contains every culture from the start of the cohort up to the test month, and it grows as the test month advances. Models **M1–M14**, listed in the table further below, repeat these windows with different sets of past-resistance features.

Two further restrictions keep the experiment close to clinical use. A patient who has a culture in the test month is removed from training, so the classifier is not evaluated on a patient already seen. When the training window ends immediately before the test window, cultures from the final 48 hours are excluded as well, because the antibiogram result of those cultures is not yet available.

The figure originally published with the paper is kept in [`assets/figures/paper_windows.png`](assets/figures/paper_windows.png). It shows the same placement of the training and test intervals.

## Features derived from past antibiograms

Each culture is described by **demographic** variables, variables of the **ICU admission**, and variables of the **culture** itself, including its date. The study also proposes two groups of features that summarize antibiograms already reported before the culture being predicted.

Both groups use cultures of *Pseudomonas aeruginosa* collected between **21 days** and **48 hours** before the current culture. The 48-hour limit is the time required for an antibiogram result to become available. The 21-day limit is the period during which a resistant result is still treated as current.

A past **resistant** culture contributes a weight that decreases with the number of elapsed days. A past **susceptible** culture contributes zero. With $\lambda = 0.095$, a resistant culture $c_k$ collected on day $d_k$ contributes the following amount to the culture $c$ collected on day $d$:

$$\mathrm{NEF}(c_k, c) = e^{-\lambda (d - d_k)}.$$

The first group, written **`p&a_i`** in the paper, describes the **patient under study**. For antibiotic family $a_i$ it is the largest contribution among that patient's past cultures:

$$p_{a_i} = \max_k \mathrm{NEF}(c_k, c).$$

The second group, written **`r&a_i`**, describes the **other patients in the ICU**. For each of them the largest contribution is taken, and those values are added:

$$r_{a_i} = \sum_{j \neq p} \max_k \mathrm{NEF}(c_{j,k}, c).$$

Six features of each group are computed, one per family. If the window contains no test of that family, the value is set to **zero**: the absence of a test was read as an absence of evidence that a resistant bacterium was present.

**Mutual information**, computed for each variable separately, assigns the highest score to the **date of the culture** and also ranks the ICU-wide features `r&a_i` highly. **Random-forest importance**, computed with all variables used jointly, ranks the patient's own recent result `p&a_i` first for five of the six families. The culture date remains informative, but it mainly reflects the rise in resistance over time. Used as a predictor, it biases the classifier toward the class that predominates in the years being predicted, and the accuracy on susceptible cultures falls. For **polymyxins**, where resistant cultures are few, the culture date remains the most important variable.

<p align="center">
  <img src="assets/figures/mutual_information.png" alt="The five variables with the highest mutual information for each antibiotic family. The culture date is the largest marker in every column." width="520"><br>
  <sub>Mutual information, computed one variable at a time. The area of each marker is the score.</sub>
</p>

<p align="center">
  <img src="assets/figures/random_forest_importance.png" alt="The five variables with the highest random-forest importance for each antibiotic family. The patient's own past result leads for five families." width="520"><br>
  <sub>Random-forest importance, computed with all variables used jointly. The patient's own past result leads for every family except polymyxins.</sub>
</p>

## The fourteen models

Logistic regression is the baseline classifier. Random forest is the second classifier. Before logistic regression, the predictors are standardized with the mean and variance of the training window. The class distribution is left as it is observed: the published experiments do not oversample the minority class, because the change in that proportion is part of the phenomenon under study.

| Models | Training window | Past-resistance features |
| --- | --- | --- |
| M1, M2, M3 | Sliding, 4 years, placed 0, 2 and 4 years before the test | none |
| M4, M5, M6 | Sliding, 4 years, placed 0, 2 and 4 years before the test | ICU patients only (`r&`) |
| M7, M8, M9 | Sliding, 4 years, placed 0, 2 and 4 years before the test | the patient and the ICU (`p&` and `r&`) |
| M10, M11, M12 | Incremental, from the first culture | none, ICU only, or both |
| M13, M14 | Incremental, with a weight on each training culture | ICU only, or both |

Every model is trained twice, once with the culture date and once without it.

For M13 and M14 the training cultures do not all contribute equally. A culture close to the test month receives a larger weight than a culture from the early years. If $d_l$ is the date of the last culture in the training window and $d_c$ is the date of the culture being weighted, its weight is

$$w(c) = e^{-\lambda (d_l - d_c)}.$$

This $\lambda$ is a second parameter. It is distinct from the value $0.095$ used to build `p&` and `r&`. It is chosen, for each experiment, from

$$\{0,\ 10^{-5},\ 10^{-4},\ 10^{-3},\ 10^{-2},\ 0.1,\ 1\}$$

as the value that obtains the best figures of merit. When $\lambda = 0$, the exponent is zero for every culture, so every training culture receives the same weight, $w(c) = 1$. When $\lambda$ is large, only cultures collected shortly before the test month keep a weight close to 1, and the older cultures are almost ignored. The selected values are listed in [`results/published_lambda.csv`](results/published_lambda.csv).

Performance is summarized by three figures of merit. They are computed from counts accumulated over all test months, rather than by averaging the accuracy of each month. A month that contains many cultures then contributes more than a month that contains few. Let $N_s$ and $N_r$ be the number of susceptible and resistant cultures in the accumulated test set, and let $S_s$ and $S_r$ be the number of those cultures that are correctly classified. Total, resistant and susceptible accuracy are

$$A_{\mathrm{Tot}} = \frac{S_s + S_r}{N_s + N_r}, \qquad A_{\mathrm{Rst}} = \frac{S_r}{N_r}, \qquad A_{\mathrm{Scb}} = \frac{S_s}{N_s}.$$

## Published results

The table below reports, for each family, the configuration whose lower class accuracy is the highest, in the order total / resistant / susceptible accuracy. The complete grid is in [`results/published_accuracy.csv`](results/published_accuracy.csv).

| Family | Model | Window | Classifier | Culture date | Total / resistant / susceptible accuracy |
| --- | --- | --- | --- | --- | --- |
| AMG | M14 | Weighted incremental | Random forest | excluded | **79.55 / 75.76 / 90.91** |
| CAR | M12 | Incremental | Logistic regression | excluded | **75.00 / 74.39 / 83.33** |
| CF4 | M14 | Weighted incremental | Random forest | excluded | **60.67 / 58.62 / 64.52** |
| PAP | M14 | Weighted incremental | Logistic regression | excluded | **71.91 / 69.81 / 75.00** |
| POL | M1 | Sliding, 0 years | Logistic regression | excluded | **74.71 / 63.33 / 80.70** |
| QUI | M9 | Sliding, 4 years | Random forest | either | **83.02 / 81.25 / 100** |

Neither **logistic regression** nor **random forest** is ahead for every family. The best classifier changes with the family, and the scores vary substantially from one family to another.

One pattern holds across that variation. Accuracy falls when both the **culture date** and the patient's own past result **`p&a_i`** are left out: the classifier then follows the majority class of the training period, which often differs from the majority class of the test months. The culture date partly compensates, by biasing the prediction toward the class that predominates near the test month. Total accuracy rises, and accuracy on the minority class falls. The features `p&a_i` reach a high accuracy without that bias, and for several families `p&a_i` alone exceeds the configurations that use the culture date. A single-variable comparison of the two is in [`results/published_single_feature.csv`](results/published_single_feature.csv). Adding the ICU-wide features `r&a_i` on their own, without `p&a_i`, does not consistently improve on the same model trained without them.

The **sliding window** is the most restrictive of the three: training uses only a small set of recent cultures. The gap and the feature set that perform best change from one family to another, and no single combination does so throughout. The **incremental window** is more stable. It trains on every culture from the start of the cohort, and that larger set reduces the variance of the fit. Its best results include `p&a_i`, and its accuracy is comparable to the best sliding window, and better for some families. The **weighted incremental window** combines the two. Through the decay $\lambda$ it can behave as either of them: a small value retains the older cultures, and a large value retains only the recent ones. It is already the scheme with the best result for the most families, three of the six. In the present cohort that advantage is only partly visible. With more training data, or by combining several values of $\lambda$, the weighted window should be able to match either of the other two and to surpass them where they disagree.

## Running the code

```bash
pip install -e ".[dev]"
python -m amr_icu --synthetic
```

The default command trains logistic regression and a random forest for models M7, M12 and M14 on an invented cohort, for the aminoglycoside family. It takes a few minutes. The printed accuracies only confirm that the pipeline runs. They are not comparable to the published table.

```bash
python -m amr_icu --synthetic --family all --models all --date-culture both
python -m amr_icu --data path/to/cohort.csv --family qui --models M14 --methods rf
python -m amr_icu --synthetic --importance
pytest
```

A cohort file uses the columns described in [`data/README.md`](data/README.md). Each row is one culture. A resistance cell contains `r`, `s`, or is left empty when that family was not tested. The package derives the calendar variables and the past-resistance features.

```text
src/amr_icu/
  features.py      p& and r&, from the negative exponential
  preprocess.py    stays of at least 48 hours, integer codes, correlated dates removed
  windows.py       monthly test, sliding and incremental training, patient exclusion
  models.py        logistic regression, with standardization, and random forest
  metrics.py       total, resistant and susceptible accuracy, accumulated over months
  experiment.py    models M1–M14 and the search over the weight decay
  synthetic.py     invented cohort used to run the code
```

## Data

The hospital extract is not part of this repository. Due to confidentiality, it cannot be uploaded. The file [`data/synthetic/cohort.csv`](data/synthetic/cohort.csv) has the same columns and contains no patient from the study. Accuracies computed from it cannot be compared with the published results.

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

A machine-readable citation is in [`CITATION.cff`](CITATION.cff). The figures taken from the study are in `assets/figures/` and are reproduced with the paper.
