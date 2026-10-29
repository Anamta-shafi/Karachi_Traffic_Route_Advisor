# Departure-Time External Experimental ML Evaluation

This report covers two deliberately separate, calendar-only external models. Neither model is validated for Karachi.

## Applicability

**Experimental external-data model — not validated for Gulistan-e-Jauhar.**

> ML model trained on external traffic data — not locally validated for Gulistan-e-Jauhar.

> Accident-risk model trained on external UK STATS19 collision data — not locally validated for Gulistan-e-Jauhar and not corridor-matched to Karachi.

The models predict a calendar-slot traffic band and a calendar-slot accident-time prior. They do not use Karachi route or crash observations, and they do not claim a Karachi accident probability or ETA.

## Traffic Model

- Dataset: Metro Interstate Traffic Volume
- Source: https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume
- DOI: 10.24432/C5X60B
- Licence: CC BY 4.0
- Coverage: I-94 westbound ATR station 301, Minneapolis-St Paul, Minnesota, USA
- Raw rows: 48,204
- Unique hourly rows: 40,575
- Time span: 2012-10-02T09:00:00 to 2018-09-30T23:00:00
- Target: traffic_volume (vehicles per hour)
- Classes: low, moderate, high; train terciles [2213.0, 4570.0]
- Split: train 28,378, validation 6,062, test 6,087
- Selected regressor: random_forest
- Selected classifier: hist_gradient_boosting

### Traffic Held-Out Test

| Predictor | MAE | RMSE | R2 | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| calendar_mean | 257.351 | 445.385 | 0.9494 | 0.8921 | 0.8935 |
| random_forest_refit | 236.360 | 429.160 | 0.9530 | — | — |
| hist_gradient_boosting_classifier_refit | — | — | — | 0.9093 | 0.9094 |

Traffic classifier test accuracy: **0.9093**; macro F1: **0.9094**; multiclass Brier: **0.1333**; ROC-AUC OVR macro: **0.9799**.

## Accident-Time Model

- Dataset: STATS19 Road Safety Data (collision tables)
- Source: https://www.gov.uk/government/statistics/road-safety-data
- Licence: Open Government Licence v3.0
- Coverage: Great Britain (England, Scotland, Wales)
- Raw collision records: 306,710
- Hourly cells: 26,304
- Time span: 2023-01-01T00:00:00 to 2025-12-31T23:00:00
- Target: severity-weighted collisions/hour (fatal=10, serious=3, slight=1)
- Classes: low, moderate, high; train terciles [10.0, 23.0]
- Split: train 18,388, validation 3,922, test 3,946
- Selected regressor: hist_gradient_boosting_poisson
- Selected classifier: hist_gradient_boosting

### Accident Held-Out Test

| Predictor | MAE | RMSE | R2 | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| calendar_mean | 6.220 | 8.707 | 0.6413 | 0.6994 | 0.7003 |
| hist_gradient_boosting_poisson_refit | 6.045 | 8.352 | 0.6700 | — | — |
| hist_gradient_boosting_classifier_refit | — | — | — | 0.7182 | 0.7158 |

Accident classifier test accuracy: **0.7182**; macro F1: **0.7158**; multiclass Brier: **0.3727**; ROC-AUC OVR macro: **0.8791**.

## Leakage Controls

- Calendar-only features: hour, weekday, month, weekend.
- No traffic lags, no Mapbox evidence, no target-hour weather, and no target-derived features.
- Chronological 70/15/15 split with a 24-hour embargo.
- Baselines, class thresholds, and models are fitted on training data; the test split is untouched until final evaluation.
- The accident model uses collision timestamps only; it does not use post-departure records.
- No sensor-to-road or UK-to-Karachi corridor mapping is performed.

## Limitations

- Both models are trained outside Karachi and are not locally validated.
- The traffic model is one Minnesota interstate sensor; volume is not speed, travel time, or congestion.
- The accident model is a Great Britain collision-time prior; it is not a Karachi crash history, not corridor-matched, and not a probability that a crash will occur on a Karachi route.
- Calendar-only models cannot distinguish Karachi routes; route differences are shown separately through OSM context and current Mapbox evidence.
- Class probabilities are model outputs; they are not calibrated confidence intervals.
- Mapbox evidence is request-time only and is never used as a label or feature.
