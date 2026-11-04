# Hotel Cancellations

## Overview

This project predicts whether a hotel booking will be canceled. It cleans a sample of real booking records, trains a logistic regression classifier with scikit-learn, and scores it on a held-out test set.

The workflow has three steps:

1. **Clean** (`src/features.py`): remove duplicate bookings, set missing travel-agent IDs to 0, convert City Hotel 2017 lead times from hours to days, drop columns that leak the outcome, and add `total_nights` and `has_agent` features.
2. **Train** (`src/train.py`): split the cleaned data 80/20 (stratified on the target, `random_state=42`), fit the model, and save the model and test set to `models/`.
3. **Evaluate** (`src/evaluate.py`): load the saved model and test set and print accuracy, precision, and recall.

Workshop instructions are in [WORKSHOP.md](WORKSHOP.md).

## Data

`data/hotel_bookings_sample.csv` contains 20,000 sampled bookings from 2015 to 2017 for one city hotel and one resort hotel in Portugal. Each row is one booking, and `is_canceled` is 1 if the booking was canceled. The sample was drawn with `random_state=42` and changed for teaching purposes (see `data/README.md`).

| Stage | Rows | Source |
|---|---|---|
| Supplied CSV | 20,000 | `data/README.md`, checked by loading the CSV |
| After removing duplicate rows | 16,756 | `features.clean_bookings` on the CSV |
| Training set (80%) | 13,404 | output of `python src/train.py` |
| Held-out test set (20%) | 3,352 | output of `python src/evaluate.py` |

After cleaning, 30.6% of bookings are canceled. The cleaned data has 10,433 City Hotel and 6,323 Resort Hotel bookings.

The model uses 16 numeric and 8 categorical columns (listed in `NUMERIC` and `CATEGORICAL` in `src/train.py`). `reservation_status` and `reservation_status_date` are dropped because they are only known after a booking is canceled or completed.

**Source:** Nuno Antonio, Ana de Almeida, and Luis Nunes, "Hotel booking demand datasets," *Data in Brief* 22 (2019): 41-49. Licensed under CC BY 4.0. We used the TidyTuesday copy (2020-02-11).

## Setup

The Codespace (`.devcontainer/devcontainer.json`) uses Python 3.11 and installs the requirements automatically. To install them yourself:

```bash
pip install -r requirements.txt
```

`requirements.txt` lists the direct dependencies: pandas, scikit-learn, and joblib.

## Usage

Run these from the repository root. Train the model first; it writes `models/model.joblib` and `models/test_set.joblib`:

```bash
python src/train.py
```

Then score it on the held-out test set:

```bash
python src/evaluate.py
```

To see the same metrics for each hotel separately:

```bash
python scripts/recall_by_hotel.py
```

## Results

These numbers come from a fresh run of `python src/train.py` followed by `python src/evaluate.py` with the current code. The model was evaluated on 3,352 held-out bookings.

| Metric | Value |
|---|---|
| Accuracy | 0.786 |
| Precision | 0.727 |
| Recall | 0.479 |

In plain terms: the model is right about 79% of the time. When it predicts a cancellation, it is right about 73% of the time. It catches about 48% of the bookings that were actually canceled.

**Note on an earlier report:** the repo used to include `results/metrics.json`, dated 2026-09-14, which reported higher scores on 4,000 test bookings. The current code does not reproduce those scores. That count matches 20% of the raw 20,000 rows, so the report was probably produced before duplicate bookings were removed. We deleted the file so readers don't take it as current. The table above is the only result this code produces.

## Limitations

- **The model misses many cancellations.** Overall recall is 0.479, so more than half of the bookings that were canceled are predicted as not canceled.
- **It does worse on some bookings than others.** From `python scripts/recall_by_hotel.py`:

  | Hotel | Test bookings | Recall | Precision |
  |---|---|---|---|
  | City Hotel | 2,051 | 0.506 | 0.748 |
  | Resort Hotel | 1,301 | 0.419 | 0.678 |

  For Resort Hotel bookings, the model catches about 4 in 10 cancellations, compared with about 5 in 10 for City Hotel. Someone using it to plan for Resort Hotel cancellations should expect most of them to go unflagged. These figures come from one test split, so small differences between hotels may not hold on other data.
- **One model, one split.** The results come from a single logistic regression with default settings, a single 80/20 split, and the default 0.5 decision threshold. There is no tuning, cross-validation, or comparison against other models.
- **Limited data.** The data covers two hotels in Portugal from 2015 to 2017, sampled and modified for teaching. The results may not carry over to other hotels or later years.
- **Cleaning assumptions.** The code assumes City Hotel lead times from 2017 were logged in hours and divides them by 24. Missing travel-agent IDs are treated as "no agent" (0). If either assumption is wrong, the related features are wrong too.

## Team

- syedaurooj06