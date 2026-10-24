# Red Wine Quality Classification Project

## Goal

Build and compare machine-learning classifiers that predict the quality of a red
wine from its physicochemical measurements. The work is a compact university
project used to simulate ANISHA's multi-step, multi-agent workflow.

## Dataset

- File: `winequality-red.csv`
- Rows: 1,599
- Predictor columns: 11 numeric physicochemical measurements
- Target: `quality`
- Observed target values: 3, 4, 5, 6, 7, and 8
- Missing cells observed during the initial schema inspection: 0
- SHA-256: `d6a0d9bd24806944818795f22500c46cb6424cbff517aacda36595d3ed9b2daa`

The target is imbalanced: qualities 5 and 6 account for most rows. The project
must inspect this distribution during EDA before deciding whether to retain all
six classes or define a smaller set of quality bands. That decision is currently
open and must not be silently assumed by an agent.

## Planned workflow

1. **Exploratory data analysis** — validate the data, describe distributions,
   inspect correlations and outliers, and document class imbalance.
2. **Problem and evaluation design** — confirm the label strategy, split policy,
   preprocessing, baseline, and metrics without leaking test information.
3. **Model comparison** — train and tune several appropriate classifiers under
   the same reproducible evaluation protocol.
4. **Interpretation** — compare performance, errors, important features, and
   limitations.
5. **Report** — write a source- and run-grounded university-project report.

## Current state

The dataset has been registered and its basic schema has been inspected. EDA,
label design, model training, evaluation, and report writing have not started.

The next action is to perform reproducible EDA and bring the label-strategy and
evaluation decisions back to the user for confirmation.
