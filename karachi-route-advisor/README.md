# Karachi Route Advisor

Course prototype for a landmark-to-landmark route advisor in Gulistan-e-Jauhar,
Karachi. The system combines a real OpenStreetMap road graph, independent custom A*
and Dijkstra pathfinders, a Python forward-chaining expert system, a route-specific
future-traffic model, and optional Mapbox `driving-traffic` evidence.

The architecture is layered. A*/Dijkstra costs, the academic ranking, and expert
rules remain unchanged and independently testable. A separate final AI decision
layer ranks OSM candidates using matched Mapbox traffic evidence, future-ML
scenarios, and expert constraints, then produces one explainable recommendation plus
up to two distinct alternatives. Mapbox routes are evidence only, never competing
route options. Provider evidence adjusts an OSM ETA once and is never presented as
the OSM future-traffic ML prediction or added again to the unchanged A* cost.

## What Is Implemented

- Real OSM `drive` graph filtered to drive highways only, with the whitelist enforced
  on cache load and routing-graph conversion.
- Nine verified landmarks inside the bbox, with drive-access snapping and a warning
  when the nearest drive node is more than 75 m away.
- Independent `src/pathfinding/` package containing the custom A* pathfinder, the
  Dijkstra baseline, candidate alternative generation, and shared graph helpers.
- Custom A* shortest-time routing over OSM-derived edge travel times.
- Dijkstra baseline over the same graph and edge weights, with expanded-node counts.
- Final AI policy `ai-final-v2` that ranks OSM A*/Dijkstra candidates using
  matched Mapbox traffic evidence, future-ML scenarios, and expert constraints.
  Mapbox routes are evidence only, not competing candidates.
- Exactly one recommended route plus up to two distinct alternatives, with OSM
  geometry deduplication and transparent decision evidence.
- Alternative generation with edge penalization, duplicate removal, time limits, and
  pairwise edge-overlap checks against every accepted alternative.
- Analytical ETA, road-risk, toll, impassable, and unlit-ratio calculations.
- Python forward-chaining expert system with rules `R01`-`R16`, conflict resolution,
  vetoes, penalties, bonuses, warnings, a derived congestion-band fact, and a
  deterministic trace.
- Preference-aware ranking for `Fastest`, `Shortest`, `Least congested`, and `Safest`.
- Optional Mapbox Directions API evidence using `mapbox/driving-traffic`, requested at
  the selected Karachi departure converted to UTC, with an ordinary `mapbox/driving`
  fallback.
- Structured Mapbox traffic-evidence status: `annotated`, `partial_eta_delta_only`, or
  `no_traffic_evidence`.
- Mapbox-to-OSM traffic-evidence matching with two-way corridor overlap, segment
  coverage, provider-baseline comparability, and explicit limitations.
- Segment-level congestion adjustment when annotations align with the provider
  geometry; otherwise a defended provider departure/typical ratio applied to the OSM
  analytical ETA. The provider's absolute ETA is never copied.
- One authoritative route assessment per displayed OSM route. It selects exactly one
  decision ETA: matched provider-adjusted, future-ML scenario, or plain analytical.
- A future-ML scenario prediction per OSM route with LOW/MODERATE/HIGH probabilities,
  confidence, top drivers, and explicit synthetic-label provenance. It is shown in
  technical details and never changes the A* score.
- An external calendar-only forecast module retained as a technical benchmark. It
  combines external traffic-volume and UK collision-time models, with explicit
  provenance and non-Karachi-validation disclosures; it is not the main UI.
- Advisory expert-system pass `R17`-`R22` retained for technical disclosures and
  external-model separation.
- A genuine external ML traffic-volume benchmark trained on UCI Metro Interstate
  Traffic Volume (UCI 492), with chronological validation/test evaluation and saved
  artifacts.
- A reproducible synthetic route-time simulation and route-level traffic classifier
  trained on 34 route/calendar/weather/external-profile features, with chronological
  train/validation/test splits and a 24-hour embargo.
- An opt-in local traffic-observation validator/importer for future Karachi-specific
  training; it never collects data automatically and never trains on its own.
- Streamlit UI with a small main screen: one recommended route with its future-ML
  prediction, up to two selectable alternatives, a selected-route details panel, a
  Folium map that highlights only the active OSM route, and one technical-details
  expander. The expander holds the ES trace, Dijkstra comparison, matched Mapbox
  evidence, traffic-evidence status, snap warnings, provenance, and limitations.
- Explicit unknown-evidence handling: absent Mapbox road-risk/toll metadata is not
  treated as zero risk or toll-free.
- Open-Meteo weather lookup with an explicit dry-assumption fallback.
- Cached GraphML plus OSM API and Overpass graph-building paths.
- Automated unit, integration, and Streamlit AppTest coverage.

## Study Area And Landmarks

The bbox is:

```text
south=24.898, west=67.108, north=24.944, east=67.168
```

OSMnx bbox order is `(west, south, east, north)`:

```text
(67.108, 24.898, 67.168, 24.944)
```

The nine landmarks are defined in `src/config.py`:

- Johar Chowk Mosque
- Gulistan-e-Jauhar Police Station
- Jauhar Inn Banquet Block 1
- Block 8 bus stop
- Shell Jauhar
- Block 7 Decent Houses
- NED University
- Safari Park
- Karachi Race Club

Safari Park and Karachi Race Club use drive-access coordinates snapped to the nearest
drive-network node. The service still warns for any endpoint more than 75 m from a
drive road, so future coordinate changes cannot silently produce a large map gap.

## Quick Start

Run from the project root in PowerShell:

```powershell
cd C:\Users\syeda\karachi-route-advisor

# The virtual environment already exists in this workspace.
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt

# Create .env from the example only if it does not already exist.
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# Edit .env and set:
# MAPBOX_ACCESS_TOKEN=your_public_token_here

# Optional: rebuild the OSM graph cache.
& '.\.venv\Scripts\python.exe' '.\scripts\fetch_graph.py'

# Rebuild the external traffic benchmark artifacts (already saved in this workspace).
& '.\.venv\Scripts\python.exe' '.\scripts\train_traffic_model.py'

# Rebuild the experimental departure-time external models (already saved).
& '.\.venv\Scripts\python.exe' '.\scripts\train_departure_models.py'

# Rebuild the route-specific traffic model and synthetic simulation (already saved).
& '.\.venv\Scripts\python.exe' '.\scripts\train_route_traffic_model.py'

# Run the test suite.
& '.\.venv\Scripts\python.exe' -m pytest -q

# Start the app.
& '.\.venv\Scripts\python.exe' -m streamlit run app.py --server.port 8501
```

The current workspace already contains a verified drive-only graph cache at
`data/cache/gulistan_jauhar_drive.graphml`, so rebuilding is optional. A rebuild
filters non-drive ways and exits nonzero if any remain.

The external benchmark artifacts are already saved under `models/traffic_flow/`, and
the source archive is under `data/external/metro_interstate/`. The training script
requires that extracted dataset; it does not download data.

The experimental departure-time artifacts are saved under
`models/departure_forecast/`. Their training script reads the same extracted UCI 492
CSV plus `data/external/stats19/collision-2023.csv` through `collision-2025.csv`; it
does not download data and does not use Mapbox evidence as a label.

The route-specific traffic artifacts are saved under `models/route_traffic/`. The
training script builds routes from the real cached OSM graph, calibrates a documented
synthetic BPR-style simulation with the real UCI/STATS19 hour-of-week profiles, and
writes the model, metadata, evaluation, synthetic dataset, and Markdown report. It
reads the extracted UCI 492 and STATS19 CSVs directly, does not download data, and
does not treat Mapbox evidence as a label. The prediction service separately loads the
saved STATS19 accident prior from `models/departure_forecast/`.

## Using The App

1. Choose a **From** and **To** landmark.
2. Choose departure date and time. The default date is tomorrow; the traffic model
   requires a future departure.
3. Choose a preference.
4. Optionally enable strict toll avoidance.
5. Optionally use matched Mapbox traffic evidence.
6. Click **Find Best Route**.

The main screen is intentionally small:

- The recommended route is selected by default. Every recommended and alternative
  route has its own stable route-ID selector; the selection is held in session state.
- Selecting a route updates the highlighted map line and one selected-route details
  panel with that route's ETA, future-ML prediction, provider evidence, incident risk,
  and decision explanation.
- Up to two distinct alternatives remain visible as cards with ETA, distance,
  traffic-evidence type, incident-risk status, and plain-language departure advice.
- A Folium map shows the OSM routes with only the active route in bold green; other
  routes are muted and dashed. A legend names the selected OSM route.
- A single **Technical Details & Data Limitations** expander containing the
  assessment table, per-route evidence and limitations, expert-system trace, future-ML
  details, matched provider evidence, A* versus Dijkstra baseline, local-observation
  status, warnings, and provenance. The assessment table and expanders mark the active
  route.
- Changing the trip context and running a new search resets the selection to the new
  recommendation; an invalid or stale route ID also falls back to it.

Mapbox routes never appear as separate route options. When the geometry and provider
baseline are defensible, the matched provider delay is applied once to the OSM
analytical ETA. Otherwise the app falls back to the future-ML scenario estimate or the
plain analytical OSM ETA and says so. Incident risk remains **Unavailable** unless a
matched provider route reports an incident or closure; the external UK STATS19 prior is
shown only as a non-route-specific limitation.

## Architecture

```text
app.py                      Streamlit UI and Folium map
src/config.py               bbox, landmarks, weights, speeds, environment helpers
src/contracts.py            frozen dataclasses shared across modules
src/calendar_features.py    shared calendar-only feature builder
src/route_features.py       route fingerprint, road composition, OSM context bands
src/route_traffic_features.py  shared 34-feature route/time/weather builder
src/route_traffic_ml.py     route-specific traffic classifier, advice, provenance
src/osm_graph.py            OSM API/Overpass loading, graph prep, geometry, stats
src/pathfinding/core.py     shared graph/path types, search helpers, overlap
src/pathfinding/astar.py    independent custom A* pathfinder
src/pathfinding/dijkstra.py independent Dijkstra baseline
src/pathfinding/alternatives.py candidate alternative generation
src/routing.py              compatibility facade re-exporting pathfinding APIs
src/eta.py                  analytical ETA, risk, toll, access, lighting
src/expert_system.py        forward-chaining rules and working memory
src/decision.py             unchanged academic normalization/ranking
src/ai/optimizer.py         final AI decision policy ai-final-v2
src/final_optimizer.py      compatibility facade re-exporting the AI layer
src/traffic_evidence.py     Mapbox-to-OSM corridor and segment evidence matching
src/mapbox_client.py        token-safe Mapbox Directions client with depart_at
src/traffic_prediction.py   route-specific evidence fallback and predictor interface
src/ml_traffic.py           external benchmark loader, feature builder, ML assessment
src/departure_forecast.py   external calendar forecast loader and combination policy
src/local_traffic.py        opt-in local observation validation and storage
src/weather.py              Open-Meteo client and fallback
src/service.py              orchestration, evidence construction, final AI invocation
scripts/fetch_graph.py      graph cache builder
scripts/fetch_osm_api.py    raw OSM API fetch/parse helper
scripts/train_traffic_model.py  external benchmark training and evaluation
scripts/train_departure_models.py  external calendar traffic/accident training
scripts/train_route_traffic_model.py  synthetic simulation and route traffic training
scripts/validate_routes.py  72-pair route validation
scripts/mapbox_feasibility.py
tests/                      unit and integration tests
data/cache/                 GraphML and raw OSM cache
data/external/              downloaded candidate datasets and audit notes
data/local/                 opt-in local observations (Git-ignored)
data/simulated/             documented synthetic route-time dataset (Git-ignored)
models/traffic_flow/        saved benchmark model, metadata, features, evaluation
models/departure_forecast/  experimental external traffic and accident artifacts
models/route_traffic/       route traffic model, metadata, features, evaluation
reports/                    evaluation reports
```

## Algorithms

### OSM Graph Preparation

The graph is a directed `drive` network filtered to a drive-highway whitelist. The
filter runs when building from OSM XML, when loading the cached GraphML, and again
when converting to the routing graph. The largest strongly connected component is
retained so landmark pairs are mutually reachable. Each edge receives:

- `length_m`
- `road_type`
- `road_name`
- `speed_kph` from `maxspeed` or highway-type defaults
- `base_time_s = length_m / (speed_kph * 1000 / 3600)`
- `surface`, `lit`, `toll`, and `access` attributes

Parallel edges are collapsed to the fastest edge for routing.

### Custom A*

A* searches the directed routing graph for minimum `base_time_s`.

- `g(n)` is the accumulated edge travel time.
- `h(n)` is straight-line haversine distance to the goal divided by the graph's
  maximum speed.
- The heuristic is admissible because it assumes the maximum possible speed.
- A counter breaks ties deterministically.
- `expanded_nodes` records how many nodes were popped from the heap.

### Dijkstra Baseline

Dijkstra is the same search with `h(n) = 0`. It uses exactly the same graph and edge
weights as A*. The service runs it separately and stores a `BaselineComparison`
containing:

- Dijkstra route
- A* expanded-node count
- Dijkstra expanded-node count
- whether the paths are identical
- whether the optimal base-time costs match

The UI displays the comparison. A* is expected to expand no more nodes than Dijkstra;
a different path with the same cost is treated as a valid tie.

### Alternative Generation

The primary route is generated first. Its edges are then penalized and A* is rerun at
factors `1.8`, `3.0`, and `5.0`.

An alternative is accepted only if:

- it is not an exact duplicate;
- its base time is no more than 45% above the primary;
- its edge overlap with every already accepted alternative is no more than 70%;
- the maximum of three alternatives is not exceeded.

This gives diverse routes without allowing arbitrary detours or two near-duplicate
alternatives that only differ from each other.

### Analytical ETA And Risk

ETA is not live traffic. It uses:

- OSM-derived edge speed;
- a departure-time factor;
- a weather factor;
- minimum speed clamping;
- edge length.

Road risk uses highway type, surface, night-time lighting, and high-speed exposure.
Toll and impassable flags are derived from OSM tags. Unlit ratio is the fraction of
route edges tagged as unlit.

### Forward-Chaining Expert System

The expert system uses a small working memory of route and context facts. Rules fire
in deterministic strata:

| Stratum | Rules | Purpose |
| --- | --- | --- |
| `veto` | `R01`-`R03` | Closure, impassable segment, strict toll violation |
| `soft` | `R04`-`R10`, `R16` | Congestion, incidents, risk, rain, night, slow route, congestion-band derivation |
| `preference` | `R11`-`R14` | Fastest, shortest, least-congested, safest bonuses |
| `data` | `R15` | Missing live-traffic warning |

Rules are sorted by stratum, priority, specificity, and rule ID. Each rule fires at
most once per route. Every firing is recorded in the route verdict trace.

`R16` derives a `congestion_band` fact from numeric congestion, and `R04`/`R05`
consume that derived fact in the same pass. This is real forward chaining rather than
a flat set of independent checks. Vetoes exclude a route from ranking. Soft rules add
penalties. Preference rules add bonuses. Data rules add warnings without changing the
score.

### Academic Ranking

`src/decision.py` remains the pre-existing comparison ranking. For surviving custom
routes, it:

1. Normalizes ETA, distance, risk, and congestion to `[0, 1]`.
2. Applies preference weights.
3. Omits congestion when it is unavailable and renormalizes the remaining weights.
4. Applies the expert-system penalty/bonus multiplier.
5. Adds a confidence penalty.
6. Produces a `0`-`100` score and a human-readable explanation.

This academic ranking is displayed unchanged for comparison. It is not the final AI
recommendation, and its cost is not added again to the final policy score.

### Final AI Decision Layer

The final policy is implemented in `src/ai/optimizer.py` as `ai-final-v2`.

1. Apply hard expert vetoes and eligibility checks before scoring.
2. Keep only OSM A*/Dijkstra candidates. Mapbox routes are evidence, never competing
   route candidates.
3. Match each Mapbox route to an OSM corridor using two-way geometry overlap at 35 m
   tolerance. Shared endpoints alone are insufficient.
4. Prefer segment-level congestion when at least 40% of the OSM route is matched and
   the annotation arrays align with the provider geometry. Otherwise apply the
   provider departure/typical duration ratio only when the provider and OSM baselines
   are comparable. Never copy the provider's absolute ETA onto the OSM route.
5. Use exactly one decision ETA per route: matched traffic-adjusted OSM ETA first,
   then the future-ML scenario premium, then the plain analytical OSM ETA. If provider
   evidence and ML are both present, provider evidence drives the ETA and ML is shown
   only as a future scenario, avoiding double counting.
6. Remove duplicate fingerprints and OSM near-duplicates above 70% directed-edge or
   geometry overlap.
7. Normalize decision time, distance, and road risk. Unknown road risk is scored at
   `0.50` (`1.00` for Safest), never zero.
8. Merge the preference congestion weight into effective time, apply expert
   `1 + 0.15*penalty - 0.10*bonus` (floored at `0.75`), and add a confidence penalty
   only when the ML scenario drives the ETA.
9. Sort deterministically and return exactly one recommendation plus at most two
   distinct alternatives.

The UI exposes the assessment label, decision ETA and basis, confidence, matched
evidence, incident status, and limitations. Detailed scores and traces live in the
technical-details expander rather than the main screen.

## Mapbox Evidence

Mapbox is optional. Set the public token in `.env`:

```text
MAPBOX_ACCESS_TOKEN=your_public_token_here
```

The app requests `mapbox/driving-traffic` at the selected departure. The Karachi
departure is converted from UTC+5 to UTC and sent as Mapbox `depart_at`; the provider
ETA is therefore evidence for the requested departure, not an unaligned request-time
snapshot. If the traffic profile fails, the app tries ordinary `mapbox/driving` and
labels it explicitly as non-traffic-aware. If both fail, the OSM analytical route
remains available.

A traffic-aware Mapbox route is used only as matched traffic evidence. An ordinary
Mapbox driving route is not traffic-aware and cannot adjust the OSM ETA. Mapbox does
not expose the OSM road-risk or toll attributes used by the custom router, so those
values are marked unavailable and handled conservatively.

Mapbox results are used transiently. They are not cached, exported, or stored. Display
attribution is `© Mapbox © OpenStreetMap`.

Important integrity rule: a duration-versus-typical ETA delta is **not** segment-level
congestion. The service exposes the structured `traffic_status` field with these
values:

```text
annotated
partial_eta_delta_only
no_traffic_evidence
no_mapbox
```

`annotated` means at least one non-unknown congestion annotation, incident, or closure
was returned. `partial_eta_delta_only` means only an ETA-versus-typical delta was
available. `no_traffic_evidence` means neither was available. `no_mapbox` means no
Mapbox route was returned. The latest live check returned `annotated`; this is
request-dependent, not a permanent guarantee. The app does not claim verified live
congestion detection when annotations are absent.

## Route-Specific Traffic Assessment

### Dataset audit and external benchmark

The repository was audited before choosing an approach. The cached drive graph has
**5,520 nodes** and **14,585 edges**, with OSM road attributes but no timestamped
traffic observations, route-level travel-time history, or segment-level historical
congestion labels. Mapbox returns requested-departure provider evidence only, and
Open-Meteo provides current weather, not a historical traffic archive.

Candidate datasets were audited:

- **UCI Traffic Flow Forecasting (734):** 36 sensors in Northern Virginia/Washington
  D.C., 15-minute normalized volume, no absolute timestamps or coordinates. Rejected
  because it cannot be aligned to calendar departures or Karachi routes.
- **UCI Metro Interstate Traffic Volume (492):** hourly volume at one I-94 westbound
  sensor in Minneapolis-St Paul. Selected for a genuine external benchmark and as the
  new calendar-only traffic prior; not Karachi route prediction.
- **UK STATS19 2023-2025:** Great Britain collision records with dates, times, and
  severity. Selected as the new calendar-only accident prior, with OGL v3.0 and an
  explicit non-Karachi, non-corridor-matched disclosure.
- **RTIRPC 2007-2015:** 298,654 Karachi accident records, but request-only and not
  publicly redistributable. Rejected for this prototype.
- **UTD19:** 40 cities and 23,541 detectors, registration required, no Karachi, far
  larger than needed. Rejected.
- **Karachi study paper (`10.61797/ijaaiml.v4i2.337`):** smartphone GPX tracks on
  three roads, but no released raw dataset or data-availability statement was found.
  Rejected because the observations are not available for reuse.
- **GitHub Karachi candidate:** 26,064 rows, all marked `is_synthetic=True`, no
  licence, Google Maps/Open-Meteo-derived. Rejected as synthetic.
- **Kaggle AetherGrid:** explicitly simulated data. Rejected.

The older external benchmark is trained, but it is not mapped to Karachi roads and is
not used for route ETAs, congestion levels, or travel-time forecasts. The external
calendar-only comparison forecast is described below; it also does not produce a
Karachi ETA or a crash probability.

### Trained external benchmark

The selected dataset is UCI 492 (CC BY 4.0, DOI `10.24432/C5X60B`). The raw archive
is 405,547 bytes and the extracted CSV is 405,373 bytes. It contains 48,204 hourly
rows from 2012-10-02 09:00 to 2018-09-30 23:00. There are 7,629 duplicate timestamp
rows with identical volume values; after keeping one row per timestamp there are
40,575 unique hours, 11,976 missing hourly slots in the full span, and 31,738 usable
rows after lag/calendar construction. Missing target values were not interpolated.

The target is `traffic_volume` in vehicles per hour. Features are lags 1, 2, 3, 6,
12, 24, 48, and 168 hours; hour/day-of-week/month sine and cosine encodings;
weekend; and holiday. Contemporaneous weather is excluded to avoid future leakage.
The split is chronological 70/15/15: 22,216 train rows, 4,761 validation rows, and
4,761 test rows. The test period is 2018-03-11 23:00 to 2018-09-30 23:00.

Validation comparison:

| Predictor | MAE | RMSE | R2 |
| --- | ---: | ---: | ---: |
| Persistence (lag 1) | 591.376 | 820.754 | 0.8277 |
| Seasonal naive (lag 24) | 592.692 | 1,036.187 | 0.7254 |
| Calendar mean | 320.413 | 543.846 | 0.9244 |
| HistGradientBoosting | 163.190 | 250.454 | 0.9840 |
| RandomForest | 162.625 | 252.910 | 0.9836 |

Held-out future test:

| Predictor | MAE | RMSE | R2 |
| --- | ---: | ---: | ---: |
| Persistence (lag 1) | 593.263 | 820.419 | 0.8279 |
| Seasonal naive (lag 24) | 539.380 | 1,015.559 | 0.7364 |
| Calendar mean | 253.252 | 430.896 | 0.9525 |
| HistGradientBoosting | 138.497 | 214.521 | 0.9882 |
| RandomForest | 136.920 | 216.551 | 0.9880 |

RandomForest was selected by validation MAE and refit on train+validation. Its
held-out test MAE is **136.920 vehicles/hour**, versus **253.252** for the
calendar-mean baseline, a **45.9%** MAE improvement. The held-out residual p05 is
-314.9, median -6.9, and p95 298.5; the absolute-error p90 is 307.0. These are
empirical residuals, not a calibrated confidence interval.

Artifacts are saved at:

```text
models/traffic_flow/traffic_volume_model.joblib
models/traffic_flow/metadata.json
models/traffic_flow/feature_definitions.json
models/traffic_flow/evaluation.json
reports/traffic_model_evaluation.md
data/external/metro_interstate/provenance.json
```

The model artifact is about 15.5 MB. The model is one-hour-ahead and requires recent
lag observations. It is **not applicable** to the selected Karachi route: the data
have no Karachi observations, route IDs, route geometry, speed, travel time, or
congestion target, and no sensor was mapped to a Karachi road.

### Implemented route-specific future traffic model

`src/traffic_prediction.py` still defines the ETA evidence fallback. The route-level
traffic classifier lives in `src/route_traffic_ml.py`. It is never mixed into the
fallback ETA, the custom A* cost, the academic ranking, or the Mapbox assessment. The
final AI layer consumes its congestion band once as a disclosed free-flow time premium.

No real Karachi route-level traffic labels exist in the repository. The model
therefore learns from a clearly disclosed synthetic target while using real data for
route geometry and real external time-of-week calibration. It does not present the
synthetic labels as observed Karachi traffic.

For each OSM route, the technical-details expander shows:

- **Future scenario:** LOW, MODERATE, or HIGH with class probabilities and the
  classifier's confidence, capped at `Medium` because there is no Karachi
  real-world validation.
- **External accident prior:** the UK STATS19 collision-time prior, explicitly
  labeled as not Karachi crash history and not a route-specific crash probability.
- **Evidence selection:** whether matched provider evidence, the future-ML scenario,
  or the plain analytical OSM ETA supplied the single decision ETA.
- **Model details:** model name/version, training rows, real/synthetic record counts,
  top drivers, feature list, and limitations.

Route cards show ETA, distance, provider traffic evidence, incident status, the
route-specific future-traffic ML band with its class probabilities, confidence,
provenance, selected departure, and plain-language advice. The selected-route details
panel repeats the active route's evidence and adds its ETA breakdown and decision
explanation. Requested-departure Mapbox evidence remains separately labelled and is
never presented as the future-ML forecast or used as a training label. The
future-traffic model is restricted to OSM routes because it needs edge-level route
features; Mapbox routes are not route candidates in the UI.

### Simulation and classifier design

- **Routes:** 149 route variants built from reachable landmark pairs with the real
  cached OSM drive graph.
- **Samples:** 500 route-time scenarios per route over 2025, giving **74,500**
  synthetic rows (45,617 LOW, 15,106 MODERATE, 13,777 HIGH).
- **Demand calibration:** real external UCI 492 hour-of-week traffic profile and real
  external STATS19 hour-of-week collision profile.
- **Simulation:** documented BPR-style volume/capacity delay with route road
  composition, free-flow speed, segment count, weather, night/weekend/Friday effects,
  bounded lognormal noise, and an accident-incident delay term. Constant:
  `delay_ratio = 0.15 * max(vc_ratio, 0.01)^4 + 0.02`; generator version
  `route-bpr-sim-v1`.
- **Label policy:** LOW `< 0.18`, MODERATE `0.18-0.45`, HIGH `>= 0.45` delay ratio.
- **Features (34):** route distance/ETA/delay/segment count/mean speed/mean segment
  length; shares of primary, secondary, tertiary, residential, trunk, motorway,
  service, and other roads; unlit ratio, toll, OSM risk, OSM context band; hour,
  day-of-week, month sin/cos; weekend/Friday/morning-peak/evening-peak/night flags;
  weather factor and rain flag; real external traffic and accident indices.
- **Split:** chronological 70/15/15 with a 24-hour embargo:
  51,899 train / 11,052 validation / 11,118 test rows.
- **Selection:** LogisticRegression, RandomForest, and HistGradientBoosting compared
  on validation macro F1. RandomForest was selected.

Held-out synthetic test metrics for the selected RandomForest:

```text
accuracy=0.7928; balanced accuracy=0.7112
macro precision=0.7360; macro recall=0.7112; macro F1=0.7179
LOW F1=0.8999; MODERATE F1=0.5964; HIGH F1=0.6573
```

These metrics measure recovery of the synthetic simulation policy. They are **not**
real Karachi traffic accuracy claims. The confusion matrix and per-class metrics are
in `reports/route_traffic_evaluation.md`.

Artifacts:

```text
models/route_traffic/traffic_model.joblib
models/route_traffic/metadata.json
models/route_traffic/evaluation.json
models/route_traffic/feature_definitions.json
data/simulated/route_traffic_simulation.csv.gz
reports/route_traffic_evaluation.md
```

Rebuild with `scripts/train_route_traffic_model.py`. If artifacts are missing, the UI
degrades to an explicit unavailable state rather than fabricating a prediction. The
model is an advisory layer only: it never changes the A* cost, route ordering, or
expert-system score.

## External Calendar-Only Comparison Forecast

This is the **External calendar-only forecast (comparison)** card. It is deliberately
separate from the route-specific future traffic model and the route ETA evidence
assessment.

### Why it uses external data

No legitimate downloadable dataset was found with Karachi route-level timestamped
traffic, travel time, or record-level crashes. The credible Karachi accident source is
the RTIRPC 2007-2015 compilation, which is request-only and not publicly
redistributable. The model therefore uses two external calendar signals and labels
them as experimental:

- **Traffic:** UCI Metro Interstate Traffic Volume (UCI 492), one I-94 westbound
  sensor in Minneapolis-St Paul. CC BY 4.0, DOI `10.24432/C5X60B`. Target: hourly
  vehicle volume, not speed or travel time.
- **Accident:** UK Department for Transport STATS19 collision tables for 2023-2025.
  Open Government Licence v3.0. 306,710 collisions; fatal=4,477, serious=72,196,
  slight=230,037. Target: hourly severity-weighted collisions using the documented
  policy fatal=10, serious=3, slight=1.

The exact user-facing disclosure is:

```text
ML model trained on external traffic data — not locally validated for Gulistan-e-Jauhar.
```

The accident panel also states that it is an external UK prior, not Karachi crash
records, and not corridor-matched.

### Features and leakage controls

Both models use only seven calendar features: sine/cosine hour, sine/cosine day of
week, sine/cosine month, and weekend. There are no traffic lags, no target-hour
weather, no Mapbox ETA differences, and no route identifiers. Mapbox evidence is never
a label or feature. Splits are chronological 70/15/15 with a 24-hour embargo; class
thresholds and baselines are fitted on the training split only, and the test split is
untouched during model selection.

Traffic split: 28,378 train / 6,062 validation / 6,087 test. Accident split: 18,388 /
3,922 / 3,946. Class bands are low/moderate/high using training-split terciles.

### Actual held-out metrics

Traffic (test):

```text
calendar-mean baseline: MAE 257.351, accuracy 0.8921, macro F1 0.8935
selected regressor: RandomForest, MAE 236.360, RMSE 429.160, R2 0.9530
selected classifier: HistGradientBoosting, accuracy 0.9093, macro F1 0.9094
multiclass Brier: 0.1333; ROC-AUC OVR macro: 0.9799
```

Accident (test):

```text
calendar-mean baseline: MAE 6.220, accuracy 0.6994, macro F1 0.7003
selected regressor: HistGradientBoosting Poisson, MAE 6.045, RMSE 8.352, R2 0.6700
selected classifier: HistGradientBoosting, accuracy 0.7182, macro F1 0.7158
multiclass Brier: 0.3727; ROC-AUC OVR macro: 0.8791
```

These are modest, honest external-calendar improvements. They are not Karachi
accuracy claims, and they are not travel-time or crash-probability metrics.

### Assessment policy

The card computes `worst(predicted_traffic, accident_prior)`. A high OSM route-context
band can raise a low result to moderate. If the selected route is a Mapbox route,
requested-departure provider evidence is applied only when it belongs to that exact
route and can worsen the result, never improve it. A custom A* route does not inherit
Mapbox geometry evidence; that evidence is displayed separately. The card also shows
a route comparison table at the selected departure.

Artifacts:

```text
models/departure_forecast/traffic_model.joblib
models/departure_forecast/accident_model.joblib
models/departure_forecast/metadata.json
models/departure_forecast/evaluation.json
models/departure_forecast/feature_definitions.json
reports/departure_forecast_evaluation.md
data/external/stats19/provenance.json
```

Rebuild with `scripts/train_departure_models.py`. The app degrades to an explicit
model-unavailable state if artifacts are missing, and the UI never substitutes a
fabricated prediction.

### Opt-in local data path

`src/local_traffic.py` validates user-supplied CSV observations with these required
columns:

```text
timestamp,route_id,segment_id,travel_time_min
```

Optional columns are `distance_km`, `speed_kph`, `source`, and `notes`. Timestamps
must parse, identifiers must be non-empty, travel time must be positive and finite,
and duplicate timestamp/route/segment keys are rejected. Valid rows can be explicitly
saved to `data/local/traffic_observations.csv`, which is Git-ignored. The app never
collects observations automatically and never trains a model automatically. At least
168 observations are required before a preliminary route-specific training experiment
can be considered; the app reports that state but does not claim a trained local
model.

### What a valid Karachi model would need

A defensible Karachi model requires timestamped historical travel-time, speed, or
volume observations with route or segment identifiers, repeated across many days and
times, plus a chronological train/test split. Candidate features would include
departure time, day of week, route distance, road composition, weather, and repeated
segment history. Evaluation should report MAE and RMSE against a simple baseline. The
opt-in local path is the start of that data collection; no Karachi accuracy claim is
made.

## Data Sources

| Source | Use | Notes |
| --- | --- | --- |
| OpenStreetMap | Road graph, tags, geometry | © OpenStreetMap contributors, ODbL |
| Mapbox Directions API | External route/ETA comparison | Optional public token; transient use only |
| Open-Meteo | Optional current weather | CC BY 4.0; dry fallback if unavailable |
| UCI 492 Metro Interstate Traffic Volume | External benchmark, calendar traffic prior, route-simulation calibration | CC BY 4.0; one Minnesota sensor; not Karachi route data |
| UK STATS19 2023-2025 | External calendar accident prior and route-simulation calibration | OGL v3.0; Great Britain; not Karachi crash data |
| Synthetic route-time simulation | Training labels for the route traffic classifier | Clearly labeled synthetic; not observed Karachi traffic |
| UCI 734 Traffic Flow Forecasting | Rejected candidate | CC BY 4.0; no absolute timestamps or coordinates |
| RTIRPC 2007-2015 | Rejected Karachi accident source | Request-only; not publicly redistributable |
| Karachi study paper | Rejected candidate | No released raw data found |
| GitHub/Kaggle synthetic candidates | Rejected | Synthetic and/or unlicensed; not used |
| Opt-in local observations | Future Karachi-specific training | User-supplied CSV; Git-ignored; no automatic collection |

The project does not use Prolog, TomTom, or `karachi_weather.csv`.

## Verified Results

Current local graph:

```text
nodes=5,520
edges=14,585
non_drive_edges=0
largest_scc=5,520
largest_scc_ratio=1.0
```

Route validation:

```text
pairs=72
route_count_distribution={1: 14, 2: 40, 3: 17, 4: 1}
failures=[]
eta_range_min=1.7
eta_range_max=18.5
distance_range_km_min=0.99
distance_range_km_max=8.80
```

A*/Dijkstra and alternative checks over the same 72 pairs:

```text
same_optimal_cost=72/72
max_astar_dijkstra_expanded_ratio=0.9657
alternative_overlap_violations=0
alternative_time_violations=0
```

Route-specific traffic assessment coverage includes custom-route prediction, route
switching, departure-time refresh, annotated/partial/missing Mapbox evidence,
route-specific status, degraded mode, and UI labels. Interactive selection coverage
includes the default recommendation, selecting each alternative, stale/invalid
selection fallback, selected-route evidence, highlighted map lines, and selection reset
after a new trip.

Final AI coverage includes the unchanged academic ranking tests, hard-veto exclusion,
future-traffic re-ranking, analytical fallback without fabricated evidence, matched
Mapbox corridor/segment evidence, provider-versus-ML no-double-counting, incident-risk
unavailability, deterministic ties, one recommendation, and the three-route display
limit. Service tests verify requested-departure UTC conversion, Mapbox evidence-only
behavior, stale-evidence exclusion, unknown Mapbox risk/toll metadata, and the clean
Streamlit rendering path.

Departure-time forecast coverage includes calendar-feature parity, model loading,
missing-artifact degradation, exact disclosures, worst-component combination, route
context escalation, Mapbox-route-only evidence application, route comparison rows,
advisory rules `R17`-`R22`, and Streamlit AppTest rendering.

External ML benchmark verification:

```text
dataset=UCI Metro Interstate Traffic Volume (492)
raw_rows=48,204
unique_timestamp_rows=40,575
usable_rows=31,738
split=22,216 / 4,761 / 4,761
selected_model=random_forest
validation_mae=162.625
test_mae=136.920
test_rmse=216.551
test_r2=0.9880
calendar_mean_test_mae=253.252
model_artifact_size_bytes=15,531,638
route_applicable=false
```

The external benchmark is loaded and tested for finite volume prediction, but it is
not mapped to Karachi routes. Local CSV validation, explicit save, deduplication,
training-readiness reporting, missing-artifact degradation, past-departure rejection,
route switching, and prediction-failure handling are covered by tests.

Route-specific future traffic model verification:

```text
routes=149
synthetic_rows=74,500
class_counts={low: 45,617, moderate: 15,106, high: 13,777}
split=51,899 / 11,052 / 11,118
selected_model=random_forest
test_accuracy=0.7928
test_balanced_accuracy=0.7112
test_macro_precision=0.7360
test_macro_recall=0.7112
test_macro_f1=0.7179
real_external_records=347,285
karachi_route_level_traffic_records=0
karachi_validated=false
```

The route traffic model is tested for artifact loading, disclosure text, future-only
prediction, time sensitivity, route sensitivity, advice mapping, provenance, the
confidence cap, missing-artifact degradation, and preservation of academic route
scores. The synthetic metrics are not presented as real-world Karachi accuracy.

Experimental departure-time comparison model verification:

```text
traffic_dataset=UCI Metro Interstate Traffic Volume (492)
traffic_rows=40,575
traffic_split=28,378 / 6,062 / 6,087
traffic_selected_regressor=random_forest
traffic_test_mae=236.360
traffic_selected_classifier=hist_gradient_boosting
traffic_test_accuracy=0.9093
traffic_test_macro_f1=0.9094
accident_dataset=STATS19 2023-2025
accident_raw_rows=306,710
accident_hourly_cells=26,304
accident_split=18,388 / 3,922 / 3,946
accident_selected_regressor=hist_gradient_boosting_poisson
accident_test_mae=6.045
accident_selected_classifier=hist_gradient_boosting
accident_test_accuracy=0.7182
accident_test_macro_f1=0.7158
route_applicable=false
karachi_validated=false
```

Current test baseline:

```text
193 passed
```

The integration tests skip automatically if the GraphML cache is missing.

## Verification Commands

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -q
& '.\.venv\Scripts\python.exe' -m compileall -q app.py src scripts tests
& '.\.venv\Scripts\python.exe' '.\scripts\train_traffic_model.py'
& '.\.venv\Scripts\python.exe' '.\scripts\train_departure_models.py'
& '.\.venv\Scripts\python.exe' '.\scripts\train_route_traffic_model.py'
& '.\.venv\Scripts\python.exe' '.\scripts\validate_routes.py'
& '.\.venv\Scripts\python.exe' '.\scripts\mapbox_feasibility.py'
& '.\.venv\Scripts\python.exe' -m streamlit run app.py --server.port 8501
```

`scripts/mapbox_feasibility.py` returns:

- `0` when the traffic profile responds with routes;
- `2` when the token is missing;
- `1` when the traffic profile fails.

## Limitations

- The custom A* path itself uses analytical OSM-derived travel time, not live traffic.
  The final AI layer may add one future-ML traffic premium or compare it with Mapbox
  provider evidence.
- The final recommendation is a transparent prototype policy, not a validated
  navigation or safety system.
- The route-specific **Traffic at Your Departure** model is trained on synthetic
  labels generated by a documented simulator. It is not validated against real
  Karachi traffic, and its confidence is capped at Medium. Its metrics measure
  recovery of the simulation policy, not real-world accuracy.
- Accident risk/probability comes from a real external UK STATS19 collision-time
  prior. It is not Karachi crash history, not corridor-matched, and not a crash
  probability for the selected route.
- The external benchmark uses one Minnesota interstate sensor and is not validated for
  Karachi routes.
- The external calendar-only comparison forecast is experimental. Its traffic
  component is one Minnesota interstate sensor and its accident component is a Great
  Britain collision-time prior; neither is locally validated or corridor-matched for
  Gulistan-e-Jauhar.
- No observed Karachi route-level traffic data was used for training; all synthetic
  labels are disclosed as such.
- The route traffic model predicts a congestion band, not a travel time. The final AI
  converts it to one disclosed free-flow time premium; the route ETA panel still shows
  the OSM analytical estimate or requested-departure Mapbox provider evidence.
- The opt-in local data path stores observations for future work but does not train a
  local model automatically.
- Mapbox segment-level congestion/incident/closure annotations may be unavailable.
- A Mapbox ETA delta is evidence of a traffic-profile effect, not segment congestion.
- Mapbox `depart_at` aligns the provider request with the selected Karachi departure,
  but its traffic model, horizon, and reliability are provider-controlled and not
  documented as a guarantee for this corridor.
- Mapbox does not expose OSM road-risk or toll attributes; unknown values are not
  assumed safe, and strict toll avoidance excludes Mapbox when toll status is unknown.
- Safari Park and Karachi Race Club coordinates are drive-access points rather than
  facility centroids.
- OSM turn restrictions, lane-level details, and temporary road conditions are not
  fully modeled.
- The departure-time and weather factors are coarse prototype assumptions.
- The expert system is implemented in Python, not Prolog.
- This is a course prototype, not a production navigation or safety service.

## Security

- `.env` is Git-ignored. Never commit or print the token.
- The configured token is a public `pk.` token; still restrict it and rotate it if it
  has been shared.
- Do not add tokens to source files, logs, screenshots, or README examples.
- Mapbox responses must not be cached or exported.
- Local traffic observations are stored only under `data/local/`, which is
  Git-ignored. Do not commit user-supplied observations or include them in reports
  without permission.

## Troubleshooting

- **OSM API or Overpass fails:** use the existing GraphML cache; no rebuild is needed.
- **Mapbox returns 401/403:** verify the token in `.env`; the app falls back to OSM.
- **Mapbox returns no annotations:** inspect `traffic_status`; treat
  `partial_eta_delta_only` as ETA evidence only, and `no_traffic_evidence` as a
  traffic-profile estimate.
- **Traffic at Your Departure says unavailable:** choose a future departure and verify
  `models/route_traffic/` exists; rerun `scripts/train_route_traffic_model.py` if it
  does not. A past departure is rejected explicitly.
- **Accident evidence is unavailable:** verify
  `models/departure_forecast/accident_model.joblib`, `metadata.json`, and
  `evaluation.json`; the traffic band can still be shown while accident evidence is
  marked unavailable.
- **External comparison forecast says model unavailable:** rerun
  `scripts/train_departure_models.py`; verify the UCI 492 extracted CSV and all three
  STATS19 collision CSVs exist. The app reports the missing-artifact state instead of
  guessing.
- **External benchmark artifact missing:** rerun `scripts/train_traffic_model.py`; the
  benchmark is separate from the route traffic model.
- **Local observations are rejected:** check the required columns, positive travel
  time, non-empty route/segment IDs, and duplicate timestamp/route/segment keys.
- **Large map gap at a landmark:** the service warns when the snap distance exceeds
  75 m; verify the landmark's drive-access coordinate.
- **Map tiles are blank:** Folium tiles require internet access.
- **Graph cache missing:** run `scripts/fetch_graph.py`.
- **Tests skip integration cases:** rebuild the GraphML cache first.
