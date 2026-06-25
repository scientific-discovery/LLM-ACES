# LLM-ACES: Closed-Loop Discovery of Dynamical Systems with LLM-Guided Adaptive Search

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Arxiv](https://img.shields.io/badge/arxiv-red.svg)(https://arxiv.org/abs/2606.25039)
Official implementation of **"LLM-ACES: Closed-Loop Discovery of Dynamical Systems with LLM-Guided Adaptive Search"**

---

## 🧠 What is LLM-ACES?

LLM-ACES is a framework for automated discovery of ordinary differential equations (ODEs) from time-series data. It combines **LLM operator-concept guidance** with **PySR symbolic regression** and **active initial-condition (IC) acquisition** to recover governing equations from limited observations.

At each iteration, the LLM proposes per-dimension operator sets (unary/binary) that bias the PySR search toward physically meaningful equations. The **active variant** further selects the most informative initial conditions to query from a ground-truth oracle using Bayesian Optimization, maximizing information gain per experiment.

---

## 🚀 Key Contributions

- **LLM-guided operator concepts** — per-dimension unary/binary operator sets proposed by the LLM to constrain and accelerate symbolic regression
- **Active IC acquisition** — Bayesian Optimization selects maximally informative initial conditions to query from the ground-truth ODE oracle
- **Supports 1D–4D ODE systems** — automatic spec selection based on state dimension
- **Two run modes** — active (with IC acquisition) and non-active (fixed training set)
- **Two benchmark suites** — ODEBench (63 Strogatz ODE systems) and ODEBase (60 dynamical systems, 2D/3D)
- **Local and API LLMs** — works with a local inference server or OpenAI / Azure OpenAI

---

## 🔧 Getting Started

**Requirements:** Python 3.11+

### 1. Clone this repository

```bash
git clone https://github.com/your-org/LLM-ACES.git
cd LLM-ACES/
```

### 2. Create and activate the environment

```bash
conda env create -f environment.yml   # creates the llm-aces environment
conda activate llm-aces
```

### 3. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔑 API Keys and Configuration

Provide your API key before running any experiment:

```bash
# OpenAI
export OPENAI_API_KEY=sk-...

# Azure OpenAI (if using --api_provider azure)
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
```

To use a **local LLM inference server** instead, start your server at `http://127.0.0.1:5000/completions` and pass `--use_api false` (the default).

Key configuration parameters in `llm-aces/aces/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `api_model` | `gpt-4o-mini` | Model name / Azure deployment |
| `api_provider` | `openai` | `openai` or `azure` |
| `azure_api_version` | `2024-02-15-preview` | Azure API version |

---

## 📂 Data Generation

### ODEBench (Strogatz ODE systems)

Generates 63 ODE systems from the Strogatz catalogue, saving NPZ files to `data/ode/`:

```bash
python generate_ode.py
# optional: python generate_ode.py --save_dir /custom/path
```

### ODEBase (dynamical systems benchmark)

Generates 60 ODEBase systems (24 × 2D, 36 × 3D) from bundled equation definitions, saving to `data/odebase/`:

```bash
python generate_odebase.py
# skip noisy variants to save disk space:
python generate_odebase.py --no_snr
```

Equation definitions are bundled in `scripts/scibench/` — no external download required.

Each generated NPZ contains: `t`, `u`, `du`, `u_gen`, `du_gen`, `t_ood`, `u_ood`, `du_ood`.

---

## ⚙️ Quick Start

### Run on ODEBench (all 1D + 2D systems)

```bash
export OPENAI_API_KEY=sk-...
bash run_odebench.sh
```

### Run on ODEBase

```bash
export OPENAI_API_KEY=sk-...
bash run_odebase.sh
```

Both scripts skip already-completed runs and accept optional overrides:

```bash
bash run_odebench.sh --use_api false                      # local LLM server
bash run_odebench.sh --api_provider azure --api_model my-deployment
bash run_odebase.sh --n_iterations 15
```

### Run a single system directly

```bash
python llm-aces/active_llm_aces.py \
  --data_path data/ode/duffing-equation/duffing-equation.npz \
  --use_api true \
  --api_model gpt-4o-mini \
  --n_iterations 10 \
  --n_virtual 10
```

Pass `--n_virtual 0` to disable IC acquisition and run on the fixed training set only.

### Key CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--data_path` | required | Path to NPZ file |
| `--n_iterations` | `10` | LLM + PySR iterations |
| `--max_concepts_per_round` | `3` | LLM operator concepts per iteration |
| `--n_virtual` | `10` | Virtual IC candidates for BO (active only) |
| `--bo_init_points` | `3` | Random IC evaluations before BO (active only) |
| `--pysr_niterations` | `40` | PySR search iterations per concept |
| `--use_api` | `false` | Use hosted LLM API |
| `--api_provider` | `openai` | `openai` or `azure` |
| `--api_model` | `gpt-4o-mini` | Model / deployment name |

Results are written to `logs/` (per-iteration JSONL) and `outputs/` (best equations JSON).

---

## 📚 Citation

```bibtex
@inproceedings{llmaces2026,
  title     = {LLM-ACES: Active Concept-Guided Symbolic Regression for ODE Discovery},
  author    = {<Authors>},
  booktitle = {<Conference>},
  year      = {2026},
}
```

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE).

---

## 📬 Contact

For questions or issues, open an issue in this repository or contact us at [nikhilsa@vt.edu](mailto:nikhilsa@vt.edu).
