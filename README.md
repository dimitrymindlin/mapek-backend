# MAPEK Conversational XAI System - Backend

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](requirements.txt)

> Official backend for the HAI 2025 paper "Towards Co-Constructed Explanations: A Multi-Agent Reasoning-Based Conversational System for Adaptive Explanations".

## Quick Links
- [Paper](https://dl.acm.org/doi/pdf/10.1145/3765766.3765768)
- [Frontend repository](https://github.com/dimitrymindlin/mapek-frontend)
- [Experiment data & analysis workflow](./experiment_analysis/README.md)
- [Ethics approval summary](./Ethics_approval.pdf)
- [Docker image instructions](#docker-workflow)

## Requirements & Installation

### Prerequisites
- macOS or Linux (tested on macOS 13 and Ubuntu 22.04)
- Python 3.10 (create a virtual environment or use Conda)
- OpenAI API access (for intent recognition and response generation)
- Optional: Docker 24+ for containerised deployments

### Environment variables
Create a `.env` file in the repository root (or export these variables in your shell). The following keys are required:

```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL_NAME=gpt-4o-mini
OPENAI_ORGANIZATION_ID=...
ML_EXECUTOR_THREADS=4
# Optional overrides
OPENAI_REASONING_MODEL_NAME=gpt-4.1
XAI_CONFIG_PATH=./configs/adult-config.gin
```

Additional keys such as `OLLAMA_HOST` can be provided when swapping the LLM provider (see `parsing/llm_intent_recognition`).

### Local installation
1. Create and activate a Python 3.10 environment
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. (Optional) Warm the explanation cache
   ```bash
   # Start the backend (see "Quick Start") and in another shell run:
   curl "http://localhost:4555/init?user_id=cache-warmup"
   ```
   The first request triggers SHAP/CF computation and persists results under
   `cache/`.

## Quick Start
1. Start the Flask server on port 4555
   ```bash
   python -m flask --app flask_app run --host=0.0.0.0 --port=4555
   ```
2. Launch the [MAPEK frontend](https://github.com/dimitrymindlin/mapek-frontend) and set `PUBLIC_BACKEND_URL=http://localhost:4555/` in the frontend `.env`.
3. Visit `http://localhost:3000/prolific-xai-904dm` (frontend) to run the study loop end-to-end.

The first request will initialise the explanation caches under `cache/`.

## Docker workflow
The provided `Makefile` wraps common Docker Buildx targets:

| Command | Description |
|---------|-------------|
| `make build` | Build an ARM64 image tagged `dialogue-xai-app` (default local target on Apple Silicon). |
| `make build-amd64` | Build an AMD64 image for x86 hosts. |
| `make run` | Run the container locally on port 4000 with constrained CPU/RAM. |
| `make build-push` | Build and push a multi-arch image (requires Docker Hub credentials). |

When running with Docker, expose the API on port 4000 and update the frontend’s `PUBLIC_BACKEND_URL` accordingly (e.g. `http://localhost:4000/`).

## Project structure
```
mapek-backend/
├── configs/                 # Gin configuration files (dataset- and study-specific)
├── data/                    # Training data, trained models, question bank
├── explain/                 # Explanation generators and dialogue policies
├── llm_agents/              # LLM-based agents for intent recognition and response planning
├── experiment_analysis/     # Scripts and preprocessed results for the HAI25 study
├── flask_app.py             # Flask entrypoint and route definitions
├── Makefile                 # Docker build/run helpers
└── requirements.txt         # Python dependencies pinned for the paper release
```

## Pretrained model, caches & study analysis
- **Model artifact:** the random-forest classifier used throughout the experiments lives at `data/adult/adult_model_rf.pkl`. Train it beforehand with `python data/adult_train.py`; the script documents preprocessing steps and hyperparameters.
- **Explanation cache:** deterministic SHAP/CF artefacts are computed lazily the first time the backend receives an `/init` request and saved under `cache/`. If you need to recompute them, delete `cache/` and make a fresh `/init` call after restarting the app.
- **Experiment analysis:** follow the step-by-step guide in [`experiment_analysis/README.md`](./experiment_analysis/README.md) to regenerate cleaned datasets (scripts `1_`–`4_`) and run the cross-condition comparison pipeline in `experiment_analysis/compare_conditions/`.

## Troubleshooting
- **OpenAI authentication errors** - confirm `OPENAI_API_KEY`, `OPENAI_MODEL_NAME`, and `OPENAI_ORGANIZATION_ID` are set; restart the app after editing `.env`.
- **Missing cache files** - delete `cache/` to force regeneration or run the pre-computation script in the "Local installation" section.
- **Thread pool errors** - `ML_EXECUTOR_THREADS` must be a positive integer; the default configuration expects `4`.

## Ethics & Data Handling
- **IRB approval:** the HAI 2025 user study and data collection were authorised under the Faculty of Technology Ethics Committee at Bielefeld University. The approval letter is archived in [`Ethics_approval.pdf`](./Ethics_approval.pdf); cite protocol ID `HAI-2025-Mapek-02` when referencing the study.
- **Sanitised datasets:** running `python experiment_analysis/1_get_raw_experiment_data.py --salt <secret>` generates anonymised Stage 1 outputs in `experiment_analysis/results/<handle>/1_raw/` and a public-friendly pipeline in stages 2–4. Never publish the private exports stored in `0_private_raw/` or the auxiliary `0_private_raw/user_id_mapping.csv`.
- **Participant privacy:** the repository only includes de-identified metrics and hashed participant IDs. Free-text responses can be dropped (`--drop-free-text`) if you require an even stricter release.

## Citation
```
@inproceedings{Mindlin2025MAPEK,
  author    = {Mindlin, Dimitry, Meisam Booshehri, and Philipp Cimiano},
  title     = {Towards Co-Constructed Explanations: A Multi-Agent Reasoning-Based Conversational System for Adaptive Explanations.},
  booktitle = {Proceedings of the 13th International Conference on Human-Agent Interaction. 2025},
  year      = {2025},
}
```

## License
This project is released under the [MIT License](./LICENSE).

## Contact
For questions about the system or the HAI25 study, email dimitry.mindlin@uni-bielefeld.de.
