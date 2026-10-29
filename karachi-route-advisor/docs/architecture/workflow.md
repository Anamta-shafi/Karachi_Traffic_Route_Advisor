# Workflow

## The one-sentence version

The app takes "take me from A to B at 9am", builds a few possible driving routes on a real Karachi road map, guesses how bad traffic/risk will be on each, scores them, and shows you one best route plus a couple of alternatives.

## The actual cast (what each part does)

- **`app.py`** – the screen. It shows the form, the buttons, the map. It does no thinking.
- **`src/service.py` (`RouteService`)** – the manager. It takes the user's request and calls everyone else in the right order.
- **`src/osm_graph.py`** – loads the real road network from OpenStreetMap into memory (a big web of roads/nodes).
- **`src/pathfinding/`** – finds routes. `astar.py` finds the fastest, `dijkstra.py` finds the same answer the slow "check everything" way so we can compare, `alternatives.py` finds other reasonable routes.
- **`src/eta.py`** – does simple arithmetic: road length ÷ speed, plus time-of-day and weather adjustments = estimated minutes. Also flags tolls, bad roads, unlit roads.
- **`src/expert_system.py`** – a set of if/then rules written by hand. Example: "if the road is closed, throw this route out", "if it's raining, add a penalty", "if the user wants the safest route, reward low-risk routes."
- **`src/decision.py`** – the original class-project scoring. Numbers each route 0–100. Kept for comparison.
- **`src/mapbox_client.py`** – optionally asks Mapbox (a commercial map service) "how long does this drive take at that time?"
- **`src/traffic_evidence.py`** – decides whether Mapbox's answer actually lines up with our road route, and if so, uses it to adjust our estimate.
- **`src/route_traffic_ml.py`** – a machine-learning model that guesses LOW / MEDIUM / HIGH traffic for our route at that hour. It was trained on simulations, not real Karachi traffic.
- **`src/ai/optimizer.py`** – the final judge. Takes every route + "what we know about it" and picks the winner.
- **`src/contracts.py`** – just the shapes of data passed around (think: labelled boxes). `RouteCandidate` = "a possible route", `RouteAssessment` = "that route plus our best traffic guess", `FinalRecommendation` = "the answer".

## What actually happens, step by step

Example: Johar Chowk Mosque → Block 8 bus stop, tomorrow 9am, "Fastest".

1. **User clicks Find Best Route.** The screen grabs the weather (or pretends it's dry if the weather site fails).
2. **Service checks the request.** Are both places real? Different? Inside the allowed area? If not, show an error and stop.
3. **Find the nearest actual road** to each landmark. If it's more than 75 m away, warn the user.
4. **Build routes.** A* finds the fastest path. Then it temporarily makes those roads "expensive" and searches again, up to three times, to get different-looking routes. Routes that are duplicates, way too slow, or too similar to another are thrown away. So you end up with maybe 1–4 candidate routes.
5. **Run Dijkstra** on the same roads. Not to compete — just to show "A* found the same cheapest route, using fewer steps."
6. **Turn each route into a candidate with numbers:** estimated minutes, kilometres, risk score, toll yes/no, percentage of unlit road.
7. **Apply the hand-written rules.** Each route gets a verdict: *vetoed* (can't be used), *penalised*, or *rewarded*, with a written trace explaining why.
8. **Score them the old way** (`decision.py`). This is the "academic" score you see in the details panel.
9. **Optionally ask Mapbox.** If there's a token and the toggle is on, ask for the drive time at *your selected departure time*. If that fails, ask for a plain drive time. If that also fails, carry on without it.
10. **Ask the ML model about each surviving route.** It returns a traffic band, probabilities, and advice. It cannot change the route itself — it's only used later to adjust the time estimate.
11. **Check if Mapbox's route matches ours.** If the roads overlap enough and the numbers look trustworthy, apply Mapbox's slowdown to *our* route's time. If not, ignore Mapbox and keep our own estimate.
12. **Write one assessment per route.** This assessment contains exactly one "final time" and a label saying where that time came from: Mapbox-adjusted, or ML-adjusted, or plain estimate. Never all three stacked.
13. **Pick the winner.** Normalise time, distance, and risk; weight them by the preference (Fastest, Shortest, etc.); apply the rule penalties/bonuses; sort. Take the top route plus up to two different runners-up.
14. **Show it.** Map highlights the selected route, cards list the routes, and one expander has all the technical evidence.

## The rules that stop it lying

- Mapbox never becomes its own route option. It's only a second opinion on our route's time.
- The time estimate is adjusted **once**, not repeatedly.
- If we don't know something (incident risk, toll status), we say "unknown", not "zero".
- The ML model was trained on fake/simulated traffic, so it's always labelled that way, and its confidence is capped.
- If Mapbox data is from an earlier departure time, it's thrown out rather than reused.

## When something breaks

- Weather down → assume dry.
- Mapbox down → use our own estimate.
- ML model missing → use our own estimate.
- No road route → say so.
- Every route vetoed → refuse to recommend, don't quietly pick a bad one.

## Going deeper

The routing algorithm, the rules, how the ML prediction is built, or how the final score is computed.
