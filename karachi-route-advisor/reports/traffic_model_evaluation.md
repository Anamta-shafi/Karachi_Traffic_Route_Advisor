# External Traffic-Volume Model Evaluation

This is an external-region benchmark, not a Karachi route model.

## Dataset

- Source: UCI Metro Interstate Traffic Volume (10.24432/C5X60B)
- Coverage: I-94 westbound ATR station 301, Minneapolis-St Paul, Minnesota, USA
- Licence: CC BY 4.0
- Raw rows: 48,204
- Unique timestamp rows: 40,575
- Missing hourly slots: 11,976
- Usable lagged rows: 31,738
- Raw time span: 2012-10-02T09:00:00 to 2018-09-30T23:00:00
- Usable time span: 2012-10-09T10:00:00 to 2018-09-30T23:00:00

Duplicate timestamps were collapsed only because their traffic-volume values were identical. Target values were never interpolated.

## Chronological Split

- Train: 22,216 rows, 2012-10-09T10:00:00 to 2017-08-20T12:00:00
- Validation: 4,761 rows, 2017-08-20T13:00:00 to 2018-03-11T22:00:00
- Test: 4,761 rows, 2018-03-11T23:00:00 to 2018-09-30T23:00:00

## Validation Comparison

| Predictor | MAE | RMSE | R2 |
| --- | ---: | ---: | ---: |
| persistence_lag_1 | 591.376 | 820.754 | 0.8277 |
| seasonal_naive_lag_24 | 592.692 | 1036.187 | 0.7254 |
| calendar_mean | 320.413 | 543.846 | 0.9244 |
| hist_gradient_boosting | 163.190 | 250.454 | 0.9840 |
| random_forest | 162.625 | 252.910 | 0.9836 |

Validation MAE selected **random_forest**.

## Held-Out Future Test

| Predictor | MAE | RMSE | R2 |
| --- | ---: | ---: | ---: |
| persistence_lag_1 | 593.263 | 820.419 | 0.8279 |
| seasonal_naive_lag_24 | 539.380 | 1015.559 | 0.7364 |
| calendar_mean | 253.252 | 430.896 | 0.9525 |
| hist_gradient_boosting | 138.497 | 214.521 | 0.9882 |
| random_forest | 136.920 | 216.551 | 0.9880 |

Selected model test MAE: **136.920 vehicles/hour**.
Calendar-mean baseline test MAE: **253.252 vehicles/hour**.
MAE improvement over calendar baseline: **45.9%**.
Selected model test RMSE: **216.551**; R2: **0.9880**.

## Uncertainty

These are empirical held-out residuals, not a calibrated confidence interval.

- Residual p05: -314.9
- Residual median: -6.9
- Residual p95: 298.5
- Absolute-error p90: 307.0

## Applicability To Gulistan-e-Jauhar

This model is **not applicable** to the selected Karachi route. It predicts hourly vehicle volume at one Minnesota interstate sensor. The dataset has no Karachi observations, no Karachi road identifiers, no route geometry, and no speed or travel-time target. No sensor IDs were mapped to Karachi roads. The application therefore reports the selected-route ML forecast as unavailable.

## Limitations

- Single external sensor, not a route network.
- Target is vehicle volume, not congestion, speed, or travel time.
- Absolute timestamps are local CST and cover 2012-2018.
- Missing hours remain missing; lags are shifted over the true hourly index.
- The model requires recent lag observations and is not a long-horizon departure forecast.
- The test split is held out, but this is not an independent Karachi validation.
