# Deep Learning Knowledge Tracing (DLKT) for FeedBook

This repository contains the Deep Learning Knowledge Tracing (DLKT) experiments used to compare Knowledge Component (KC) models within the [FeedBook](https://feedbook.website/) Intelligent Language Tutoring System (ILTS). It evaluates the same model families on two related prediction tasks:

1. **First-attempt prediction** predicts whether a learner answers correctly before receiving feedback (`firstattemptcorrect`).
2. **After-feedback prediction** predicts whether a learner eventually answers correctly after receiving feedback from the system (`eventualcorrect`).

The repository focuses on reproducible, student-blocked evaluation. It does not include the FeedBook interaction data.

## Models

Each notebook is self-contained and contains the first-attempt and after-feedback experiments for one model.

| Notebook | Model |
| --- | --- |
| `models/dkt.ipynb` | Deep Knowledge Tracing (DKT) |
| `models/dkt_forget.ipynb` | DKT with forgetting features (DKT+Forget) |
| `models/dkvmn.ipynb` | Dynamic Key-Value Memory Network (DKVMN) |
| `models/skvmn.ipynb` | Sequential Key-Value Memory Network (SKVMN) |
| `models/deepirt.ipynb` | Deep Item Response Theory (DeepIRT) |
| `models/sakt.ipynb` | Self-Attentive Knowledge Tracing (SAKT) |
| `models/akt.ipynb` | Context-Aware Attentive Knowledge Tracing (AKT) |
| `models/saint.ipynb` | Separated Self-Attentive Neural Knowledge Tracing (SAINT) |
| `models/saint_plus_plus.ipynb` | SAINT++ |
| `models/gkt.ipynb` | Graph-based Knowledge Tracing (GKT) |

## Evaluation design

The experiments use nested, student-blocked cross-validation:

- Fixed outer folds define the held-out students used for final evaluation.
- Grouped inner folds tune hyperparameters using training students only.
- Early stopping is performed only on inner validation folds.
- Each inner fold records the epoch with the highest validation AUC.
- The selected configuration's median inner best epoch is used to retrain on all outer-training students.
- The outer test fold is evaluated once after final training.
- Row-level out-of-fold predictions retain `row_id`, `student_id`, `fold`, `y_true`, and `y_pred` for paired analyses.

This separation prevents the outer test fold from influencing the model, hyperparameters, or training duration.

## Repository layout

```text
.
├── models/                         # One notebook per DLKT model
├── analysis/
│   └── statistical_significance.ipynb
├── utilities/
│   └── oof_recovery.ipynb
├── data/                           # Local data and fixed fold files (not committed)
├── requirements.txt
└── README.md
```

The significance notebook contains paired student-cluster permutation tests and fold-stratified paired student-cluster bootstrap confidence intervals for pooled and fold-averaged metrics. The recovery notebook combines completed fold outputs, recalculates metrics, and repairs legacy AKT OOF files that lack student identifiers.

## Setup

Python 3.10 or 3.11 is recommended. Create an isolated environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The notebooks use model implementations from the [pyKT toolkit](https://github.com/pykt-team/pykt-toolkit). A CUDA-capable GPU is recommended for the full hyperparameter grids.

## Data preparation

Place the interaction data and fixed fold assignments under `data/`, or change the paths in the notebook setup cell.

Expected interaction columns include:

- `Anon Student Id`
- `row_id` (created from the unfiltered interaction table when absent)
- the selected KC column, such as `KC (exerciseid)`
- `firstattemptcorrect`
- `eventualcorrect`
- `First Transaction Time` for models that use temporal or forgetting features

Expected fold files:

- `data/fixed_outer_folds.csv` for first-attempt prediction
- `data/fixed_outer_folds_af.csv` for after-feedback prediction

Each fold file must contain `row_id` and `outer_fold`. Fold assignments should be created at the student level so that a student never occurs in more than one outer fold.

The raw dataset is intentionally excluded because it contains research data. Do not commit interaction logs, identifiers, checkpoints, OOF predictions, or local experiment databases unless their release has been approved.

## Running an experiment

1. Open the notebook for the desired model.
2. Set `DATA_PATH` in the setup cell.
3. Check the task configuration, KC column, fold file, sequence length, and hyperparameter grid.
4. Run the setup cell.
5. Run either the first-attempt section or the after-feedback section.

The two sections are independent. Restarting the kernel before switching tasks is recommended for long GPU runs.

## Hyperparameter grids

The same grids are used for first-attempt and after-feedback prediction.

| Parameter type | Values |
| --- | --- |
| Embedding/model/state dimension (`emb_size`, `d_model`, or `dim_s`) | 64, 128, 256 |
| Learning rate | 1e-2, 1e-3, 1e-4 |
| Dropout | 0.1, 0.3, 0.5 |
| Memory size (`size_m`) | 32, 64, 128 |
| Attention heads | 4, 8 |
| Encoder layers or blocks (`num_en` or `n_blocks`) | 1, 2, 4 |
| AKT feed-forward dimension (`d_ff`) | 64, 128, 256 |

This produces 27 configurations for DKT and DKT+Forget, 81 for DKVMN, SKVMN, and DeepIRT, 162 for SAKT, SAINT, and SAINT++, and 486 for AKT. GKT retains its existing 27-configuration grid.

## Outputs

Each model writes task-specific output directories such as `DKT_FA/` and `DKT_AF/`. The standard outputs include:

- one OOF CSV per outer fold;
- a combined `*_all.csv` OOF file;
- per-fold AUC, Accuracy, RMSE, MAE, Precision, Recall, and F1 score;
- pooled metrics over all OOF interactions; and
- the unweighted mean and sample standard deviation across outer folds.

GKT additionally maintains atomic checkpoints, an experiment journal, and a resumable final training state for long RunPod jobs, since we did not run them locally.

## Reproducibility notes

- Random seeds are set for Python, NumPy, and PyTorch.
- `row_id` is created before task-specific filtering so predictions remain aligned across models.
- Fixed outer folds are validated before training.
- Model selection uses AUC; all other metrics are computed only for evaluation.
- Precision, recall, and F1 use a probability threshold of 0.5.
- The code uses three outer folds by default to match the reported comparison. Change the fold count only when regenerating all model results under the same design.
