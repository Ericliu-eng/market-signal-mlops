# Model Card: Market Signal Classifier

## Model identity

- Registered model: `market-signal-classifier`
- Initial champion version: `1`
- Model type: Histogram Gradient Boosting Classifier
- Target: Next-day return direction
- Feature set: `pit_features_v1`

## Intended use

This model demonstrates a production-style financial ML lifecycle, including
point-in-time features, walk-forward evaluation, experiment tracking, model
registration, and controlled promotion.

It is intended for portfolio demonstration and engineering validation. It is
not approved for live trading, automated order execution, or financial advice.

## Training data

The model is trained on a versioned local market-bar snapshot after contract
validation and point-in-time feature generation. Training uses all currently
labeled historical rows only after walk-forward evaluation is complete.

## Evaluation method

- Expanding-window walk-forward validation
- Three validation folds
- No random train/test split
- Preprocessing is fitted independently inside each fold
- Challenger is compared with naïve-prior and logistic-regression baselines

## Initial champion metrics

| Metric | Mean | Standard deviation |
|---|---:|---:|
| Balanced accuracy | 0.778 | 0.048 |
| F1 | 0.775 | 0.098 |
| ROC-AUC | 0.667 | 0.236 |
| Brier score | 0.202 | 0.021 |

## Promotion requirements

A candidate must satisfy all of the following:

- Data contract passed
- Model signature present
- Balanced accuracy mean >= 0.60
- Balanced accuracy standard deviation <= 0.15
- ROC-AUC mean >= 0.60
- Brier score mean <= 0.25

A failed promotion decision must not change the existing `champion` alias.

## Limitations

- The current dataset is intentionally small and suitable only for an MVP.
- Some validation folds contain only one observed target class.
- Metrics are not evidence of live trading profitability.
- Transaction costs, slippage, liquidity, and market impact are not modeled.
- Performance may degrade under market-regime or data-distribution changes.

## Reproducibility

The MLflow run records:

- Git SHA
- Snapshot ID
- Feature-set version
- Fold configuration
- Evaluation metrics and predictions
- Model signature and input example
- Python dependency environment
- Promotion decision