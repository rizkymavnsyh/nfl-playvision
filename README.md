# NFL PlayVision

**An interactive study of NFL player movement and short-horizon position forecasting.**

[Explore the dashboard](https://rizkymavnsyh.github.io/nfl-playvision/) · [Open the Colab notebook](NFL_PlayVision_v2_Colab.ipynb) · [Read the model card](MODEL_CARD.md) · [View full results](reports/RESULTS.md)

PlayVision replays recorded tracking data, projects where a player will be **0.5 or 1.0 seconds later**, and lets you compare that projection with the actual path. The field view also measures receiver–defender separation and closing speed. It is a retrospective analysis tool, not a live game or catch-outcome predictor.

## What you can explore

- **21 clips from 13 real games:** 18 clips from 2021 tracking and three from a complete 2017 game.
- **Player-level replay:** choose a play and receiver, scrub frame by frame, and switch forecast horizons.
- **Forecast comparison:** see the model's predicted position, a constant-velocity baseline, and optional future ground truth.
- **Matchup geometry:** inspect the nearest defensive back's distance and closing speed. Proximity does not identify coverage assignment.

The dashboard runs on GitHub Pages. For an internet-free version, download [PlayVision_Offline.html](PlayVision_Offline.html) and open it locally. A short [replay video](demo/PlayVision_Replay.mp4) is also available.

## Held-out results

The table reports **position error**, not accuracy of wins, catches, or player performance. Results come from two held-out 2021 games.

| Forecast horizon | Mean position error | Within tolerance | Tolerance |
| --- | ---: | ---: | ---: |
| 0.5 seconds | **0.221 yd** | **95.86%** | 0.5 yd |
| 1.0 seconds | **0.919 yd** | **61.60%** | 1.0 yd |

At 0.5 seconds, the v2 model reduced mean error from **0.251 to 0.221 yd** relative to v1, an **11.9% reduction**. The 1.0-second model remained unchanged after a candidate performed worse on validation. See [the evaluation report](reports/RESULTS.md) for baselines, movement-only and phase analyses, and the separate 2017 test.

**Evaluation boundary:** frames within a play are correlated, and the held-out set contains only two games. The v2 comparison reused a test set already examined during v1 development; it is a repeated benchmark, not a fresh blind test. The 2021 clips end shortly after the pass release, so these numbers mostly describe movement before release. They do not establish league-wide performance.

## How the model works

The v1 pipeline uses FLAML AutoML to select a residual motion model. V2 adds a LightGBM specialist for the 0.5-second horizon, selected on two validation games. Inputs include observed position and motion history, current neighbor geometry, and player position. The model predicts a correction to a physics-based motion baseline.

Game splits are chronological: **8 training games, 2 validation games, and 2 test games** from 2021. Player ID, play ID, future coordinates, ball landing location, and play outcome are excluded from model inputs. Future tracking appears only as an evaluation overlay. The 2017 game is a separate cross-era check, not training data.

The project does **not** train on Kaggle's 2026 competition data. The optional [2026 importer guide](KAGGLE_2026_GUIDE.md) documents an adapter that has been tested with a synthetic schema fixture, not the competition files.

## Run locally

Use Python 3.11 or newer:

```bash
python -m pip install -r requirements.txt
python src/playvision.py download
python src/playvision.py prepare
python -m pytest tests -q
streamlit run app.py
```

To reproduce the v2 experiment after downloading and preparing data:

```bash
python experiments/horizon_models.py
python experiments/build_hybrid_v2.py
python src/export_demo.py --model-path models/hybrid_v2/model.joblib --metrics-path models/hybrid_v2/metrics.json
```

The [Colab notebook](NFL_PlayVision_v2_Colab.ipynb) loads the packaged model by default. Set `TRAIN_NEW=True` in the notebook to retrain. The original v1 notebook remains in the repository for reference.

## Repository guide

| Path | Contents |
| --- | --- |
| [index.html](index.html), [assets/](assets/) | GitHub Pages dashboard and replay data |
| [PlayVision_Offline.html](PlayVision_Offline.html) | Standalone dashboard with embedded data and JavaScript |
| [src/](src/) | Tracking preparation, forecasting, importers, and dashboard exporters |
| [models/hybrid_v2/](models/hybrid_v2/) | Packaged v2 model and metrics |
| [experiments/](experiments/) | Horizon comparison and model selection |
| [reports/](reports/) | Evaluation details and verification artifacts |
| [tests/](tests/) | Split, leakage, physics, adapter, and inference checks |
| [DATA_SOURCES.md](DATA_SOURCES.md) | Source attribution and provenance |

Raw source CSVs and full processed datasets are not included. This independent portfolio project is not affiliated with the NFL, its teams, Kaggle, or the data providers.
