from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable

# Auth is read from environment variables — do not put real keys here.
OPENAI_API_KEY = ""
AZURE_OPENAI_API_KEY = ""
AZURE_OPENAI_ENDPOINT = ""
AZURE_OPENAI_DEPLOYMENT = ""

_HERE = Path(__file__).resolve().parent
_LLMACES_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_LLMACES_ROOT))


def _resolve_user_path(raw: str | None) -> Path | None:
    if raw is None:
        return None
    p = Path(raw)
    if p.is_absolute():
        return p
    for base in (Path.cwd(), _HERE):
        cand = (base / p).resolve()
        if cand.exists():
            return cand
    return (Path.cwd() / p).resolve()


def _preflight_api_env(provider: str) -> None:
    provider = (provider or "").lower().strip()
    if provider == "openai":
        if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")):
            raise EnvironmentError(
                "Missing OpenAI credentials. Set OPENAI_API_KEY (preferred) or API_KEY."
            )
    elif provider == "azure":
        if not (os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("AZURE_API_KEY")):
            raise EnvironmentError(
                "Missing Azure OpenAI credentials. Set AZURE_OPENAI_API_KEY (preferred) or AZURE_API_KEY."
            )
        if not (os.environ.get("AZURE_OPENAI_ENDPOINT") or os.environ.get("AZURE_ENDPOINT")):
            raise EnvironmentError(
                "Missing Azure OpenAI endpoint. Set AZURE_OPENAI_ENDPOINT (preferred) or AZURE_ENDPOINT, "
                "e.g. 'https://<resource>.openai.azure.com/'."
            )
    else:
        raise ValueError(f"Unsupported --api_provider {provider!r} (use 'openai' or 'azure').")

# Keep PySR/BLAS/Julia from oversubscribing shared machines by default.
for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "JULIA_NUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

from aces import config as config_lib
from aces.concept_prompting import (
    build_data_summary,
    extract_spec_instruction,
    format_existing_concepts,
    is_duplicate_concept,
    is_stop_response,
    render_prompt_template,
)
from aces.concept_sampler import ConceptLocalLLM

import aces_utils as active_lib  # noqa: E402


parser = argparse.ArgumentParser(
    description="LLM operator-concept + PySR symbolic regression with active IC acquisition."
)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--spec_path", type=str, default=None)
parser.add_argument("--log_path", type=str, default=None)
parser.add_argument("--n_iterations", type=int, default=10)
parser.add_argument("--n_init", type=int, default=30)
parser.add_argument("--use_api", type=lambda x: x.lower() == "true", default=False)
parser.add_argument("--api_provider", type=str, choices=["openai", "azure"], default="openai")
parser.add_argument("--api_model", type=str, default="gpt-4o-mini")
parser.add_argument("--azure_api_version", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--n_virtual", type=int, default=10)
parser.add_argument("--acq_method", choices=["random", "bo"], default="bo")
parser.add_argument("--bo_init_points", type=int, default=3)
parser.add_argument("--bo_candidate_pool", type=int, default=256)
parser.add_argument("--bo_kappa", type=float, default=2.0)
parser.add_argument("--max_concepts_per_round", type=int, default=3)
parser.add_argument("--concept_temperature", type=float, default=0.8)
parser.add_argument("--concept_stop_token", type=str, default="NO_NEW_CONCEPT")
parser.add_argument("--pysr_niterations", type=int, default=40)
parser.add_argument("--pysr_populations", type=int, default=15)
parser.add_argument("--pysr_procs", type=int, default=1)
parser.add_argument("--pysr_timeout_seconds", type=float, default=0.0)
parser.add_argument("--fit_pause_seconds", type=float, default=5.0)
parser.add_argument("--output_dir", type=str, default=None)


# Operator concept parsing

_VALID_UNARY = {
    "sin", "cos", "tan", "exp", "log", "sqrt", "abs",
    "tanh", "sinh", "cosh", "square", "cube", "inv",
    "neg", "cbrt", "log2", "log10", "exp2",
}
_VALID_BINARY = {"+", "-", "*", "/", "^"}


def _operators_in_sympy(eq_str: str) -> tuple[list[str], list[str]]:
    """Return (unary_ops, binary_ops) actually present in a PySR/sympy equation string."""
    unary: list[str] = []
    for name in ["exp2", "exp", "log2", "log10", "log", "sin", "cos", "tan",
                 "sqrt", "cbrt", "abs", "tanh", "sinh", "cosh"]:
        if re.search(rf'\b{name}\(', eq_str) and name not in unary:
            unary.append(name)
    if re.search(r'\*\*\s*2\b', eq_str) and "square" not in unary:
        unary.append("square")
    if re.search(r'\*\*\s*3\b', eq_str) and "cube" not in unary:
        unary.append("cube")
    binary: list[str] = []
    if re.search(r'\*\*', eq_str):
        binary.append("^")
    if re.search(r'(?<!\*)\*(?!\*)', eq_str):
        binary.append("*")
    if "/" in eq_str:
        binary.append("/")
    if "+" in eq_str:
        binary.append("+")
    if re.search(r'(?<![eE0-9])-', eq_str):
        binary.append("-")
    return unary, binary


def parse_operator_concept(response: str, dim: int) -> dict | None:
    """Parse per-dimension operator lists and reasoning from LLM response."""
    result: dict = {"reasoning": ""}
    for d in range(dim):
        result[f"unary_{d}"] = []
        result[f"binary_{d}"] = []

    for line in response.splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("reasoning:"):
            result["reasoning"] = stripped.split(":", 1)[1].strip()
            continue
        for d in range(dim):
            if lower.startswith(f"dx{d}_unary_operators:"):
                tokens = re.split(r"[,\s]+", stripped.split(":", 1)[1].strip())
                result[f"unary_{d}"] = [t.lower() for t in tokens if t.lower() in _VALID_UNARY]
            elif lower.startswith(f"dx{d}_binary_operators:"):
                tokens = re.split(r"[,\s]+", stripped.split(":", 1)[1].strip())
                result[f"binary_{d}"] = [t for t in tokens if t in _VALID_BINARY]

    for d in range(dim):
        if not result[f"unary_{d}"] or not result[f"binary_{d}"]:
            return None
        if "+" not in result[f"binary_{d}"]:
            result[f"binary_{d}"].append("+")
        if "*" not in result[f"binary_{d}"]:
            result[f"binary_{d}"].append("*")

    return result


def concept_key(op_concept: dict) -> str:
    dims = sorted(int(k[6:]) for k in op_concept if k.startswith("unary_") and k[6:].isdigit())
    return " ".join(
        f"dx{d}:[{','.join(sorted(op_concept[f'unary_{d}']))}|{','.join(sorted(op_concept[f'binary_{d}']))}]"
        for d in dims
    )


# LLM operator concept exploration

def _make_dim_replacements(dim: int) -> dict:
    dim_description = ", ".join(f"dx{d}/dt" for d in range(dim))
    block_lines = []
    for d in range(dim):
        block_lines += [
            f"dx{d}:",
            "    operators: [<list of operator categories>]",
            "    structure: <equation structure>",
        ]
    op_format_lines = []
    for d in range(dim):
        op_format_lines += [
            f"dx{d}_unary_operators: <comma-separated, e.g.: sin, cos, exp>",
            f"dx{d}_binary_operators: <comma-separated from: +, -, *, /, ^>",
        ]
    return {
        "dim_description": dim_description,
        "dim_blocks": "\n".join(block_lines),
        "dim_operator_blocks": "\n".join(op_format_lines),
    }


def explore_operator_concepts(
    llm: ConceptLocalLLM,
    cfg: config_lib.Config,
    spec_instruction: str,
    island_context: str,
    island_context_brief: str,
    failure_cases: str,
    data_summary: str,
    max_concepts: int,
    temperature: float,
    stop_token: str,
    dim: int = 3,
    all_seen_keys: list[str] | None = None,
) -> list[dict]:
    # Seed with cross-iteration history so the LLM never repeats a prior operator set.
    concept_strings: list[str] = list(all_seen_keys) if all_seen_keys else []
    concepts: list[dict] = []
    dim_replacements = _make_dim_replacements(dim)

    for i in range(max_concepts):
        if i == 0 and not island_context:
            template = "ode_concept_initial.txt"   # first iteration, no feedback yet
            ctx = island_context
        elif i == 0 and island_context:
            template = "ode_concept_exploit.txt"   # later iterations: exploit best + failure cases
            ctx = island_context
        else:
            template = "ode_concept_more.txt"      # diversity: only best equations
            ctx = island_context_brief
        replacements = {
            "spec_text": spec_instruction,
            "island_context": ctx,
            "failure_cases": failure_cases,
            "data_summary": data_summary,
            "existing_concepts": format_existing_concepts(concept_strings),
            "concept_stop_token": stop_token,
            **dim_replacements,
        }
        prompt = render_prompt_template(template, replacements)
        try:
            raw = llm.draw_text(prompt, config=cfg, temperature=temperature, num_samples=1, trim_code=False)[0]
        except Exception as exc:
            print(f"  [WARN] LLM concept call failed: {exc}")
            break

        if is_stop_response(raw, stop_token):
            break

        parsed = parse_operator_concept(raw, dim)
        if parsed is None:
            continue

        key = concept_key(parsed)
        if is_duplicate_concept(key, concept_strings):
            print(f"  [WARN] Exact duplicate, skipping: {key}")
            continue

        concepts.append(parsed)
        concept_strings.append(key)
        for d in range(dim):
            print(f"  [Concept {len(concepts)}] dx{d}: unary={parsed[f'unary_{d}']}  binary={parsed[f'binary_{d}']}")
        if parsed.get("reasoning"):
            print(f"  [Concept {len(concepts)}] reasoning: {parsed['reasoning']}")

    return concepts


# PySR fitting per operator concept

def fit_pysr_concept(
    op_concept: dict,
    train: dict,
    dim: int,
    niterations: int,
    populations: int,
    procs: int,
    timeout_seconds: float,
    rng_seed: int,
    pysr_log: str,
) -> tuple[list[Callable], list[str]] | None:
    """Fit one PySR model per state dimension. Returns (callables, eq_strings) or None on failure."""
    try:
        from pysr import PySRRegressor
    except ImportError:
        print("  [ERROR] PySR not installed. Run: pip install pysr")
        return None

    u = np.asarray(train["u"], dtype=float)
    du = np.asarray(train["du"], dtype=float)

    X = u
    callables = []
    eq_strings = []

    with open(pysr_log, "a", encoding="utf-8") as lf:
        for d in range(dim):
            lf.write(f"dx{d}: unary={op_concept[f'unary_{d}']}  binary={op_concept[f'binary_{d}']}\n")
        lf.write(f"n_samples={len(u)}\n")

        for d in range(dim):
            y = du[:, d]
            unary_d = op_concept[f"unary_{d}"]
            binary_d = op_concept[f"binary_{d}"]
            constraints = {"^": (-1, 1)} if "^" in binary_d else {}
            extra_kwargs = {}
            if timeout_seconds and timeout_seconds > 0:
                extra_kwargs["timeout_in_seconds"] = timeout_seconds
            model = PySRRegressor(
                niterations=niterations,
                populations=populations,
                unary_operators=unary_d,
                binary_operators=binary_d,
                constraints=constraints,
                verbosity=0,
                random_state=rng_seed,
                deterministic=True,
                parallelism="serial",
                temp_equation_file=True,
                **extra_kwargs,
            )
            try:
                model.fit(X, y)
                eq_str = "(unknown)"
                try:
                    eq_str = str(model.sympy())
                except Exception:
                    pass
                try:
                    table = model.equations_[["complexity", "loss", "equation"]].to_string()
                except Exception:
                    table = "(table unavailable)"
                lf.write(f"dx{d}/dt = {eq_str}\n")
                lf.write(f"{table}\n")
                lf.write("-" * 60 + "\n")
                eq_strings.append(eq_str)
                callables.append(model.predict)
            except Exception as exc:
                lf.write(f"[WARN] PySR fit failed for dim {d}: {exc}\n")
                lf.write("-" * 60 + "\n")
                return None

        lf.write("=" * 60 + "\n\n")

    if len(callables) != dim:
        return None

    print(f"  [PySR] equations: " + "  |  ".join(f"dx{d}/dt={eq}" for d, eq in enumerate(eq_strings)))
    return callables, eq_strings


def pysr_to_acquisition_eq(dim_fns: list[Callable]) -> Callable:
    """Wrap per-dimension PySR callables into a single (u, t, params) -> du callable."""
    def eq_fn(u: np.ndarray, _t, _params) -> np.ndarray:
        u2d = np.atleast_2d(u)
        preds = np.stack([fn(u2d) for fn in dim_fns], axis=1)
        return preds  # (N, dim)
    return eq_fn


def _eval_eq(dim_fns: list[Callable], u: np.ndarray) -> np.ndarray:
    u2d = np.atleast_2d(u)
    return np.stack([fn(u2d) for fn in dim_fns], axis=1)


# Acquisition wrapper for PySR equations

def compute_pysr_acquisition(
    pysr_eqs: list[Callable],
    candidate_u0s: np.ndarray,
    n_steps: int = 10,
    dt: float = 0.1,
) -> np.ndarray:
    """Score candidate ICs by median pairwise NMSE across short forward-Euler trajectories."""
    M = len(candidate_u0s)
    scores = np.zeros(M)

    for m, u0 in enumerate(candidate_u0s):
        u0_arr = np.asarray(u0, dtype=float).reshape(-1)
        trajectories = []

        for eq_fn in pysr_eqs:
            try:
                traj = []
                u = u0_arr.copy()
                valid = True
                for _ in range(n_steps):
                    traj.append(u.copy())
                    du = np.asarray(
                        eq_fn(np.atleast_2d(u), None, None), dtype=float
                    ).reshape(-1)
                    if not np.all(np.isfinite(du)):
                        valid = False
                        break
                    u = u + dt * du
                    if not np.all(np.isfinite(u)):
                        valid = False
                        break
                if valid and len(traj) == n_steps:
                    trajectories.append(np.concatenate(traj))  # (n_steps * dim,)
            except Exception:
                continue

        if len(trajectories) < 2:
            continue

        traj_norm = np.linalg.norm(np.mean(trajectories, axis=0)) + 1e-8
        normalized = [t / traj_norm for t in trajectories]

        pairwise_nmses = [
            float(np.mean((normalized[i] - normalized[j]) ** 2))
            for i in range(len(normalized))
            for j in range(i + 1, len(normalized))
        ]
        scores[m] = float(np.median(pairwise_nmses)) if pairwise_nmses else 0.0

    return scores


# Main loop

def _dim_val_nmse(entry_d: dict, val: dict, d: int) -> float:
    """Val NMSE for a single dimension's pool entry, evaluated independently."""
    if len(val.get("u", [])) == 0:
        return float("inf")
    try:
        u2d = np.atleast_2d(np.asarray(val["u"], dtype=float))
        du_pred = np.asarray(entry_d["dim_fn"](u2d), dtype=float).reshape(-1)
        du_true = np.asarray(val["du"][:, d], dtype=float)
        nmse = float(np.sum((du_pred - du_true) ** 2) / (np.sum(du_true ** 2) + 1e-10))
        return nmse if np.isfinite(nmse) else float("inf")
    except Exception:
        return float("inf")


def run_llm_pysr(
    system_name: str,
    spec_text: str,
    train: dict,
    test: dict,
    val: dict,
    dim: int,
    n_iterations: int,
    cfg: config_lib.Config,
    log_path: str,
    rng: np.random.Generator,
    n_virtual: int,
    true_rhs,
    acq_method: str,
    bo_init_points: int,
    bo_candidate_pool: int,
    bo_kappa: float,
    ic_bounds_per_dim: list[list[float]] | None,
    max_concepts_per_round: int,
    concept_temperature: float,
    concept_stop_token: str,
    pysr_niterations: int,
    pysr_populations: int,
    pysr_procs: int,
    pysr_timeout_seconds: float,
    fit_pause_seconds: float,
    output_dir: str | None = None,
):
    if true_rhs is None and n_virtual > 0:
        print("  [INFO] No ground-truth ODE found — IC acquisition disabled.")
        n_virtual = 0

    os.makedirs(log_path, exist_ok=True)
    results_path = os.path.join(log_path, "active_llm_pysr_results.jsonl")
    pysr_log_path = os.path.join(log_path, "pysr_equations.txt")
    open(results_path, "w", encoding="utf-8").close()
    open(pysr_log_path, "w", encoding="utf-8").close()
    best_per_dim_nmse = [float("inf")] * dim  # tracked by val NMSE (fallback: train NMSE)
    best_per_dim_eq: list[dict | None] = [None] * dim

    llm = ConceptLocalLLM()
    spec_instruction = extract_spec_instruction(spec_text)
    t_ic_query = np.linspace(0.0, 1.0, 20)

    best_test_nmse = float("inf")
    queried_ics: list[np.ndarray] = []
    island_context = ""
    island_context_brief = ""
    failure_cases = "(none yet)"
    all_seen_keys: list[str] = []
    per_dim_pool: list[list[dict]] = [[] for _ in range(dim)]  # per-dim independent pool

    print(f"\nLLM-PySR Active — system: '{system_name}'  dim={dim}")
    print(f"  n_iterations={n_iterations}  n_virtual={n_virtual}  acq_method={acq_method}")
    print(f"  D_train={len(train['t'])} pts  |  test={len(test['t'])} pts\n")

    # Baseline: PySR with default full operator set, no LLM biasing
    print("=" * 65)
    print("Baseline PySR run (no LLM operator restriction)")
    _all_unary = ["sin", "cos", "tan", "exp", "log", "sqrt", "abs", "tanh", "sinh", "cosh",
                  "square", "cube", "inv", "neg", "cbrt", "log2", "log10", "exp2"]
    _all_binary = ["+", "-", "*", "/", "^"]
    baseline_concept = {f"unary_{d}": _all_unary for d in range(dim)}
    baseline_concept.update({f"binary_{d}": _all_binary for d in range(dim)})
    with open(pysr_log_path, "a", encoding="utf-8") as lf:
        lf.write("BASELINE RUN\n" + "=" * 60 + "\n")
    baseline_result = fit_pysr_concept(
        op_concept=baseline_concept,
        train=train,
        dim=dim,
        niterations=pysr_niterations,
        populations=pysr_populations,
        procs=pysr_procs,
        timeout_seconds=pysr_timeout_seconds,
        rng_seed=int(rng.integers(0, 2**31)),
        pysr_log=pysr_log_path,
    )
    if fit_pause_seconds > 0:
        time.sleep(fit_pause_seconds)
    if baseline_result is not None:
        baseline_fns, baseline_eq_strings = baseline_result
        u = np.asarray(train["u"], dtype=float)
        du_true_test = test["du"]
        du_pred = pysr_to_acquisition_eq(baseline_fns)(u, train["t"], None)
        train_norm = float(np.sum(train["du"] ** 2)) + 1e-10
        baseline_train_nmse = float(np.sum((du_pred - train["du"]) ** 2)) / train_norm
        u_test = np.asarray(test["u"], dtype=float)
        du_pred_test = pysr_to_acquisition_eq(baseline_fns)(u_test, test["t"], None)
        test_norm = float(np.sum(du_true_test ** 2)) + 1e-10
        baseline_test_nmse = float(np.sum((du_pred_test - du_true_test) ** 2)) / test_norm
        baseline_per_dim_nmse = [
            float(np.sum((du_pred_test[:, d] - du_true_test[:, d]) ** 2) /
                  (np.sum(du_true_test[:, d] ** 2) + 1e-10))
            for d in range(dim)
        ]
        print(f"  [Baseline] train NMSE: {baseline_train_nmse:.4e}  test NMSE: {baseline_test_nmse:.4e}")

        b_unary = baseline_concept["unary_0"]
        b_binary = baseline_concept["binary_0"]
        b_train_size = int(len(train["t"]))
    print("=" * 65 + "\n")

    for it in range(n_iterations):
        t_iter = time.time()
        print(f"{'='*65}")
        print(f"Iter {it+1}/{n_iterations}  |  D_train={len(train['t'])} pts  |  queried_ICs={len(queried_ics)}")

        data_summary = build_data_summary(
            {"t": train["t"], "u": train["u"], "du": train["du"]}, queried_ics
        )

        # Step 1: LLM generates operator concepts
        op_concepts = explore_operator_concepts(
            llm=llm,
            cfg=cfg,
            spec_instruction=spec_instruction,
            island_context=island_context,
            island_context_brief=island_context_brief,
            failure_cases=failure_cases,
            data_summary=data_summary,
            max_concepts=max_concepts_per_round,
            temperature=concept_temperature,
            stop_token=concept_stop_token,
            dim=dim,
            all_seen_keys=all_seen_keys,
        )
        for c in op_concepts:
            k = concept_key(c)
            if k not in all_seen_keys:
                all_seen_keys.append(k)
        if not op_concepts:
            fallback = {f"unary_{d}": sorted(_VALID_UNARY) for d in range(dim)}
            fallback.update({f"binary_{d}": sorted(_VALID_BINARY) for d in range(dim)})
            op_concepts = [fallback]

        # Step 2: PySR fits one equation set per operator concept
        pysr_eqs: list[Callable] = []
        iter_concepts_log = []
        rng_seed = int(rng.integers(0, 2**31))


        for ci, op_concept in enumerate(op_concepts, start=1):
            with open(pysr_log_path, "a", encoding="utf-8") as lf:
                lf.write(f"ITER {it+1}  CONCEPT {ci}\n" + "=" * 60 + "\n")
            result = fit_pysr_concept(
                op_concept=op_concept,
                train=train,
                dim=dim,
                niterations=pysr_niterations,
                populations=pysr_populations,
                procs=pysr_procs,
                timeout_seconds=pysr_timeout_seconds,
                rng_seed=rng_seed + ci,
                pysr_log=pysr_log_path,
            )
            if fit_pause_seconds > 0:
                time.sleep(fit_pause_seconds)
            if result is not None:
                dim_fns, dim_eq_strings = result
                eq_fn = pysr_to_acquisition_eq(dim_fns)
                pysr_eqs.append(eq_fn)

                # Train NMSE
                u = np.asarray(train["u"], dtype=float)
                du_pred_train = eq_fn(u, train["t"], None)
                train_norm = float(np.sum(train["du"] ** 2)) + 1e-10
                fit_train_nmse = float(np.sum((du_pred_train - train["du"]) ** 2)) / train_norm
                per_dim_train_nmse = [
                    float(np.sum((du_pred_train[:, d] - train["du"][:, d]) ** 2) /
                          (np.sum(train["du"][:, d] ** 2) + 1e-10))
                    for d in range(dim)
                ]
                per_dim_train_str = "  ".join(f"dx{d}: {v:.4e}" for d, v in enumerate(per_dim_train_nmse))
                print(f"  [Concept {ci}] train NMSE: {fit_train_nmse:.4e}  per-dim -> {per_dim_train_str}")

                # Val NMSE (t in [1, 2])
                if len(val.get("u", [])) > 0:
                    du_pred_val = _eval_eq(dim_fns, val["u"])
                    du_true_val = np.asarray(val["du"], dtype=float)
                    val_norm = float(np.sum(du_true_val ** 2)) + 1e-10
                    fit_val_nmse = float(np.sum((du_pred_val - du_true_val) ** 2)) / val_norm
                    per_dim_val_nmse = [
                        float(np.sum((du_pred_val[:, d] - du_true_val[:, d]) ** 2) /
                              (np.sum(du_true_val[:, d] ** 2) + 1e-10))
                        for d in range(dim)
                    ]
                    per_dim_val_str = "  ".join(f"dx{d}: {v:.4e}" for d, v in enumerate(per_dim_val_nmse))
                    print(f"  [Concept {ci}] val   NMSE: {fit_val_nmse:.4e}  per-dim -> {per_dim_val_str}")
                else:
                    fit_val_nmse = fit_train_nmse
                    per_dim_val_nmse = per_dim_train_nmse

                # Test NMSE
                u_test = np.asarray(test["u"], dtype=float)
                du_true_test = test["du"]
                du_pred_test = eq_fn(u_test, test["t"], None)
                test_norm = float(np.sum(du_true_test ** 2)) + 1e-10
                test_nmse = float(np.sum((du_pred_test - du_true_test) ** 2)) / test_norm
                test_rmse = float(np.sqrt(np.mean((du_pred_test - du_true_test) ** 2)))
                per_dim_test_nmse = [
                    float(np.sum((du_pred_test[:, d] - du_true_test[:, d]) ** 2) /
                          (np.sum(du_true_test[:, d] ** 2) + 1e-10))
                    for d in range(dim)
                ]
                per_dim_str = "  ".join(f"dx{d}: {v:.4e}" for d, v in enumerate(per_dim_test_nmse))
                print(f"  [Concept {ci}] test  NMSE: {test_nmse:.4e}  RMSE: {test_rmse:.4e}  per-dim -> {per_dim_str}")

                if test_nmse < best_test_nmse:
                    best_test_nmse = test_nmse

                # Store each dimension independently so bad dims can't drag good ones down
                for d in range(dim):
                    per_dim_pool[d].append({
                        "iteration": it + 1,
                        "concept_index": ci,
                        "dim_fn": dim_fns[d],
                        "eq_string": dim_eq_strings[d],
                        "train_nmse": per_dim_train_nmse[d],
                    })

                # Track best per-dim by val NMSE
                for d, (d_val_nmse, d_test_nmse, d_eq) in enumerate(
                    zip(per_dim_val_nmse, per_dim_test_nmse, dim_eq_strings)
                ):
                    if d_val_nmse < best_per_dim_nmse[d]:
                        best_per_dim_nmse[d] = d_val_nmse
                        best_per_dim_eq[d] = {
                            "dimension": d,
                            "equation": d_eq,
                            "test_nmse": d_test_nmse,
                            "val_nmse": d_val_nmse,
                            "iteration": it + 1,
                            "train_size": int(len(train["t"])),
                            "reasoning": op_concept.get("reasoning", ""),
                        }
                        print(f"  [Best dx{d}] updated -> val NMSE={d_val_nmse:.4e}  test NMSE={d_test_nmse:.4e}  eq={d_eq}")

                iter_concepts_log.append({
                    "concept_index": ci,
                    "operators": op_concept,
                    "equations": dim_eq_strings,
                    "fit_train_nmse": fit_train_nmse,
                    "fit_val_nmse": fit_val_nmse,
                    "per_dim_train_nmse": per_dim_train_nmse,
                    "per_dim_val_nmse": per_dim_val_nmse,
                    "test_nmse": test_nmse,
                    "test_rmse": test_rmse,
                    "per_dim_test_nmse": per_dim_test_nmse,
                })

        print(f"  [Step 2] {len(pysr_eqs)}/{len(op_concepts)} concepts fitted by PySR")

        # Step 3: IC acquisition using PySR ensemble
        n_ic_added = 0
        acq_log = None
        queried_ic = None

        if n_virtual > 0 and len(pysr_eqs) >= 2 and true_rhs is not None:
            if ic_bounds_per_dim is None:
                print("  [Step 3] Acquisition skipped — no IC bounds provided (set --ic_bounds_json).")
                continue
            lows  = np.array([b[0] for b in ic_bounds_per_dim], dtype=float)
            highs = np.array([b[1] for b in ic_bounds_per_dim], dtype=float)
            candidate_u0s = active_lib._sample_uniform_u0s(lows, highs, bo_candidate_pool, rng)

            if acq_method == "bo":
                from sklearn.gaussian_process import GaussianProcessRegressor
                from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

                candidate_unit = (candidate_u0s - lows) / (highs - lows + 1e-12)
                all_idx = np.arange(bo_candidate_pool)
                rng.shuffle(all_idx)
                init_idx = list(all_idx[:bo_init_points])
                remaining_idx = list(all_idx[bo_init_points:])

                bo_scores: dict[int, float] = {}
                init_scores = compute_pysr_acquisition(pysr_eqs, candidate_u0s[init_idx])
                for idx, s in zip(init_idx, init_scores):
                    bo_scores[int(idx)] = float(s)

                while len(bo_scores) < n_virtual and remaining_idx:
                    x_tr = candidate_unit[np.array(list(bo_scores.keys()))]
                    y_tr = np.array(list(bo_scores.values()), dtype=float)
                    if len(np.unique(y_tr)) <= 1:
                        next_i = int(remaining_idx.pop(0))
                    else:
                        kernel = (
                            ConstantKernel(1.0, (1e-3, 1e3))
                            * Matern(length_scale=np.ones(dim), nu=2.5)
                            + WhiteKernel(noise_level=1e-6)
                        )
                        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
                        gp.fit(x_tr, y_tr)
                        x_rem = candidate_unit[np.array(remaining_idx)]
                        mu, sigma = gp.predict(x_rem, return_std=True)
                        ucb = mu + bo_kappa * sigma
                        next_i = int(remaining_idx.pop(int(np.argmax(ucb))))
                    s = compute_pysr_acquisition(pysr_eqs, candidate_u0s[[next_i]])
                    bo_scores[next_i] = float(s[0])

                best_i = max(bo_scores, key=bo_scores.__getitem__)
                best_u0 = candidate_u0s[best_i]
                best_ic_score = bo_scores[best_i]
                acq_log = {"method": "bo", "n_evaluated": len(bo_scores)}
            else:
                ic_scores = compute_pysr_acquisition(pysr_eqs, candidate_u0s[:n_virtual])
                best_i = int(np.argmax(ic_scores))
                best_u0 = candidate_u0s[best_i]
                best_ic_score = float(ic_scores[best_i])
                acq_log = {"method": "random", "n_evaluated": n_virtual}

            queried_ic = np.asarray(best_u0, dtype=float)
            queried_ics.append(queried_ic.copy())
            print(f"  [Step 3] Best IC: divergence={best_ic_score:.4e}  u0={np.round(best_u0, 3).tolist()}")

            oracle_data = active_lib.query_oracle_ic(true_rhs, best_u0, t_ic_query)
            if oracle_data is not None:
                # Alternate oracle points between train (even) and val (odd)
                all_idx = np.arange(len(oracle_data["t"]))
                tr_idx  = all_idx[0::2]
                v_idx   = all_idx[1::2]
                n_tr = len(tr_idx)
                n_v  = len(v_idx)
                train = {
                    "t":   np.concatenate([train["t"],   oracle_data["t"][tr_idx]]),
                    "u":   np.concatenate([train["u"],   oracle_data["u"][tr_idx]]),
                    "du":  np.concatenate([train["du"],  oracle_data["du"][tr_idx]]),
                    "idx": np.concatenate([train["idx"], np.full(n_tr, -1, dtype=int)]),
                }
                val = {
                    "t":  np.concatenate([val["t"],  oracle_data["t"][v_idx]]),
                    "u":  np.concatenate([val["u"],  oracle_data["u"][v_idx]]),
                    "du": np.concatenate([val["du"], oracle_data["du"][v_idx]]),
                }
                print(f"  [Step 4] Oracle added {n_tr} train + {n_v} val pts -> D_train={len(train['t'])}  D_val={len(val['t'])}")
        elif len(pysr_eqs) < 2:
            print("  [Step 3] Acquisition skipped — need >=2 PySR equations.")
        else:
            print("  [Step 3] Acquisition disabled or unavailable.")

        # Build feedback per dimension independently — good dims are never penalised by bad ones
        has_pool = any(len(per_dim_pool[d]) > 0 for d in range(dim))
        has_val = len(val.get("u", [])) > 0

        if has_pool and has_val:
            best_lines = [
                "Best equations per dimension across all iterations"
                " (validation NMSE on t∈[1,2], lower is better):"
            ]
            fc_lines = [
                "Worst equations per dimension (validation NMSE on t∈[1,2])"
                " — avoid these structures:"
            ]
            for d in range(dim):
                if not per_dim_pool[d]:
                    continue
                ranked_d = sorted(per_dim_pool[d], key=lambda e, _d=d: _dim_val_nmse(e, val, _d))
                top2_d = ranked_d[:2]
                worst2_d = list(reversed(ranked_d[-2:]))
                best_lines.append(f"  dx{d}/dt:")
                for rank, e in enumerate(top2_d, 1):
                    vnmse = _dim_val_nmse(e, val, d)
                    best_lines.append(
                        f"    Rank {rank} (iter={e['iteration']}  val NMSE={vnmse:.3e}):"
                        f"  {e['eq_string']}"
                    )
                for e in worst2_d:
                    vnmse = _dim_val_nmse(e, val, d)
                    fc_lines.append(
                        f"  dx{d}/dt  (iter={e['iteration']}  val NMSE={vnmse:.3e}):"
                        f"  {e['eq_string']}"
                    )
            island_context = "\n".join(best_lines)
            island_context_brief = "\n".join(best_lines)
            failure_cases = "\n".join(fc_lines)
        elif has_pool:
            # Val set not yet available — fall back to per-dim train NMSE
            best_lines = [
                "Best equations per dimension this iteration"
                " (train NMSE, val set not yet available):"
            ]
            fc_lines = [
                "Worst equations per dimension this iteration (train NMSE)"
                " — avoid these structures:"
            ]
            for d in range(dim):
                if not per_dim_pool[d]:
                    continue
                ranked_d = sorted(per_dim_pool[d], key=lambda e: e.get("train_nmse", float("inf")))
                top2_d = ranked_d[:2]
                worst2_d = list(reversed(ranked_d[-2:]))
                best_lines.append(f"  dx{d}/dt:")
                for rank, e in enumerate(top2_d, 1):
                    tnmse = e.get("train_nmse", float("inf"))
                    best_lines.append(
                        f"    Rank {rank} (iter={e['iteration']}  train NMSE={tnmse:.3e}):"
                        f"  {e['eq_string']}"
                    )
                for e in worst2_d:
                    tnmse = e.get("train_nmse", float("inf"))
                    fc_lines.append(
                        f"  dx{d}/dt  (iter={e['iteration']}  train NMSE={tnmse:.3e}):"
                        f"  {e['eq_string']}"
                    )
            island_context = "\n".join(best_lines)
            island_context_brief = "\n".join(best_lines)
            failure_cases = "\n".join(fc_lines)
        else:
            island_context = ""
            island_context_brief = ""
            failure_cases = "(none yet)"
        print(f"  [LLM Feedback]\n{island_context if island_context else '  (none)'}")

        iter_log = {
            "iteration": it + 1,
            "train_size": int(len(train["t"])),
            "n_pysr_eqs": len(pysr_eqs),
            "n_ic_added": n_ic_added,
            "queried_ic_count": len(queried_ics),
            "queried_ic_latest": queried_ic.tolist() if queried_ic is not None else None,
            "best_test_nmse_so_far": best_test_nmse if best_test_nmse < float("inf") else None,
            "acquisition": acq_log,
            "concepts": iter_concepts_log,
            "wall_time_s": round(time.time() - t_iter, 2),
        }
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(iter_log) + "\n")

    print(f"\n{'='*65}")
    print(f"Finished. system='{system_name}'")
    print(f"  Final D_train: {len(train['t'])} pts  |  Queried ICs: {len(queried_ics)}")
    print(f"  Best test NMSE: {best_test_nmse:.4e}")
    print(f"  Log: {results_path}")

    outputs_dir = output_dir if output_dir else os.path.join(_HERE, "outputs", "active_llm_pysr_concept")
    if output_dir:
        od = _resolve_user_path(output_dir)
        assert od is not None
        outputs_dir = str(od)
    os.makedirs(outputs_dir, exist_ok=True)
    summary = {
        "problem": system_name,
        "queried_initial_conditions": [ic.tolist() for ic in queried_ics],
        "best_equations": [
            {
                "dimension": e["dimension"],
                "equation": e["equation"],
                "val_nmse": e["val_nmse"],
                "test_nmse": e["test_nmse"],
                "iteration": e["iteration"],
            }
            for e in best_per_dim_eq if e is not None
        ],
    }
    summary_path = os.path.join(outputs_dir, f"{system_name}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {summary_path}")


def main() -> None:
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    if args.use_api:
        _preflight_api_env(args.api_provider)

    data_path = _resolve_user_path(args.data_path)
    assert data_path is not None
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    raw = np.load(data_path)
    t_full, u_full, du_full = raw["t"], raw["u"], raw["du"]
    if u_full.ndim == 1:
        u_full = u_full[:, np.newaxis]
    if du_full.ndim == 1:
        du_full = du_full[:, np.newaxis]

    dim = u_full.shape[1]
    system_name = data_path.stem
    spec_path = active_lib.select_spec(dim, args.spec_path, system_name)
    spec_path_p = _resolve_user_path(spec_path)
    assert spec_path_p is not None
    spec_text = spec_path_p.read_text(encoding="utf-8")

    # Test: last 30 samples
    n_test = 30
    test_idx = np.arange(len(t_full) - n_test, len(t_full))
    pool_idx = np.arange(len(t_full) - n_test)

    def _subset(idx):
        return {"t": t_full[idx], "u": u_full[idx], "du": du_full[idx], "idx": idx}

    test = _subset(test_idx)

    cfg = config_lib.Config(
        use_api=args.use_api,
        api_provider=args.api_provider,
        api_model=args.api_model,
        azure_api_version=(args.azure_api_version or config_lib.Config().azure_api_version),
    )
    if args.log_path:
        lp = _resolve_user_path(args.log_path)
        assert lp is not None
        log_path = str(lp)
    else:
        log_path = str(_HERE / "logs" / "active_llm_pysr" / system_name)
    true_rhs = active_lib.load_true_ode(system_name)

    # Generate exactly 20 points in t∈[0,1] from the initial IC via oracle → 10 train, 10 val
    t_init_query = np.linspace(0.0, 1.0, 20)
    u0_init = u_full[0]
    init_oracle = active_lib.query_oracle_ic(true_rhs, u0_init, t_init_query) if true_rhs is not None else None
    if init_oracle is not None:
        all_idx = np.arange(20)
        train = {
            "t":   init_oracle["t"][all_idx[0::2]],
            "u":   init_oracle["u"][all_idx[0::2]],
            "du":  init_oracle["du"][all_idx[0::2]],
            "idx": np.full(10, -1, dtype=int),
        }
        val = {
            "t":  init_oracle["t"][all_idx[1::2]],
            "u":  init_oracle["u"][all_idx[1::2]],
            "du": init_oracle["du"][all_idx[1::2]],
        }
    else:
        # Fallback: use data points from t∈[0,1]
        mask = t_full[pool_idx] <= 1.0
        early_idx = pool_idx[mask][:20]
        train = _subset(early_idx[0::2])
        val   = _subset(early_idx[1::2])

    import json
    _ic_bounds_path = _HERE / "ic_bounds.json"
    ic_bounds_per_dim: list[list[float]] | None = None
    if _ic_bounds_path.exists():
        with open(_ic_bounds_path, encoding="utf-8") as f:
            ic_bounds_db = json.load(f)
        if system_name in ic_bounds_db:
            ic_bounds_per_dim = ic_bounds_db[system_name]
            print(f"IC bounds : {ic_bounds_per_dim}")
        else:
            print(f"IC bounds : '{system_name}' not found in ic_bounds.json — acquisition will be skipped.")

    print(f"System : {system_name}  dim={dim}")
    print(f"Spec   : {spec_path_p}")
    print(f"Data   : t={t_full.shape}  u={u_full.shape}  du={du_full.shape}")
    print(f"Split  : train={len(train['t'])} pts  val={len(val['t'])} pts  test={len(test['t'])} pts")

    run_llm_pysr(
        system_name=system_name,
        spec_text=spec_text,
        train=train,
        test=test,
        val=val,
        dim=dim,
        n_iterations=args.n_iterations,
        cfg=cfg,
        log_path=log_path,
        rng=rng,
        n_virtual=args.n_virtual,
        true_rhs=true_rhs,
        acq_method=args.acq_method,
        bo_init_points=args.bo_init_points,
        bo_candidate_pool=args.bo_candidate_pool,
        bo_kappa=args.bo_kappa,
        ic_bounds_per_dim=ic_bounds_per_dim,
        max_concepts_per_round=args.max_concepts_per_round,
        concept_temperature=args.concept_temperature,
        concept_stop_token=args.concept_stop_token,
        pysr_niterations=args.pysr_niterations,
        pysr_populations=args.pysr_populations,
        pysr_procs=args.pysr_procs,
        pysr_timeout_seconds=args.pysr_timeout_seconds,
        fit_pause_seconds=args.fit_pause_seconds,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
