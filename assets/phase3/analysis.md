# Phase 3 cross-market evidence

A non-baseline model is selected only if validation mean RMSE and mean plus one fold standard deviation both improve by at least 5% over last value. The locked holdout is shown only after selection.
Horizon is the next 1, 5, or 20 observed sessions, so wall-clock spans differ by market.
The 20-observation holdout and preceding three 20-observation validation blocks use identical row counts; dates are in each run manifest.

## Recommendations

```text
            asset_id  horizon        model   rmse_mean   rmse_std  final_test_rmse
       forex:EUR-USD        1   last_value    0.002233   0.001172         0.003000
       forex:EUR-USD        5   last_value    0.008163   0.003825         0.003724
       forex:EUR-USD       20   last_value    0.009249   0.003616         0.013123
oil:WTI-CUSHING-SPOT        1        ridge    0.191720   0.119367         0.650048
oil:WTI-CUSHING-SPOT       20      xgboost    1.968590   0.460711         2.472662
oil:WTI-CUSHING-SPOT        5      xgboost    1.679688   1.484766         1.304735
          equity:SPY        1   last_value    3.107707   2.866015         0.809021
          equity:SPY        5   last_value    8.423764   5.391261         5.172930
          equity:SPY       20   last_value   14.388303  10.154740         8.160799
      crypto:BTC-USD        1 holt_winters  574.898445 525.719643      4323.124812
      crypto:BTC-USD        5   last_value 1437.129152 928.755940      3065.491804
      crypto:BTC-USD       20      xgboost 3552.363735 750.389941      7606.214168
```

## Stability and failure cases

The fold standard deviations in recommendations.csv are often large relative to mean RMSE; treat small differences as inconclusive.
Compare selected final_test_rmse with last-value holdout in comparison.csv: several validation gains reverse on the holdout, particularly longer oil and BTC forecasts.
A 20-observation recursive forecast can compound model error; one-step results are a different strategy and are not pooled here.

## Failures

0 recorded run failures. See failures.csv and each run's warnings.json for convergence or storage warnings.

## Traceability

Every row in fold_metrics.csv carries run_id, config_file, manifest_file and prediction_file.