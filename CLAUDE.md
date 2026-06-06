# Claude operating notes for this repo

## Non-negotiable rules
1. **No data leakage.** Every rolling feature must call `.shift(1)` **before**
   `.rolling()`. `tests/test_leakage.py` enforces this — never weaken it.
2. **No odds in the feature matrix.** `features.build_dataset._assert_no_odds_columns`
   runs on every dataset build. Touching it requires test coverage to match.
3. **2020 is dropped.** See `features.build_dataset.EXCLUDED_SEASONS`. Don't add it back without re-approving the plan.
4. **Per-model calibration gate.** `models.calibration` writes a per-model report; `pipeline.predict._gate_passed(model_name)` is the only thing that decides flagging.

## Data conventions
* `game_id = f"{YYYY-MM-DD}_{AWAY}_{HOME}_{double_header_index}"`
* Teams use Retrosheet 3-letter codes throughout (`NYY`, `BOS`, `LAD`, ...).
* Odds API team names are full-name strings; sports-statistics uses 3-letter
  codes. The historical-odds loader in `ingestion/sports_statistics.py` keeps
  the 3-letter codes; `pipeline/train.py::_join_odds_to_games` is where the
  two namespaces meet and will need a mapping table if you extend it.

## Running things
* Unit tests: `pytest -q` from `mlb-predictor/`
* First-time ingest: `python -m ingestion.retrosheet` (then the others)
* Train: `python -m pipeline.train`
* Backtest: `python -m pipeline.train --backtest`
* Daily picks (cron hourly): `python -m pipeline.predict --date today`

## Bankroll
Update with `from config import update_bankroll; update_bankroll(2000)` — this
rewrites `.env` so subsequent runs pick up the new value.
