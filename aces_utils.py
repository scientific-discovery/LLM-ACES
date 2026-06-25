"""
active_aces.py — Active LLM-ACES pipeline.

Pipeline overview
-----------------
Data is loaded from an NPZ file (t, u, du).  The entire dataset is used as
D_train (no holdout split).  An optional --test_path NPZ is evaluated
read-only and never influences training.

Each active learning iteration:
  1. Build a data-aware prompt from D_train (summary statistics + best equation;
     no raw numeric rows — LLMs reason poorly over tables of numbers).
  2. Query the LLM → K equation candidates.
  3. Each candidate is scored in a sandbox: params fitted and NMSE evaluated on
     the full accumulated D_train.
  4. Compile each equation, optimise free params on D_train for IC acquisition.
     If a test set is provided, NMSE/RMSE on D_test is also logged (not fed back).
  5. IC acquisition: generate M candidate initial conditions, integrate all K
     equations from each over [0, 8] (120 time points), score by mean pairwise
     trajectory RMSE (information gain via divergence).  The IC with highest
     divergence is queried from the oracle.
  6. Oracle (ground-truth ODE from strogatz_ode.py) integrates from the best IC
     over [0, 8] and returns 120 exact (t, u, du) points appended to D_train.
  7. Repeat until iteration budget exhausted.

Final evaluation: best equation's params re-fitted on full D_train; RMSE/NMSE
reported on D_train and (if provided) D_test.

Spec auto-selection
-------------------
Datasets with a named spec in LLM-ACES/specs/ get the system-specific prompt
automatically.  All others fall back to the generic 1D/2D/3D spec.

Datasets with a dedicated spec:
  aizawa-attractor                → specification_aizawa_attractor_numpy.txt
  apoptosis-model                 → specification_apoptosis_model_numpy.txt
  chen-lee-attractor              → specification_chen_lee_attractor_numpy.txt
  lorenz-equations-chaotic        → specification_lorenz_chaotic_numpy.txt
  lorenz-equations-periodic       → specification_lorenz_chaotic_numpy.txt
  lorenz-equations-complex-periodic → specification_lorenz_chaotic_numpy.txt
  maxwell-bloch-equations         → specification_maxwell_bloch_numpy.txt
  rössler-attractor-chaotic       → specification_rossler_chaotic_numpy.txt
  rössler-attractor-periodic      → specification_rossler_chaotic_numpy.txt
  rössler-fixed-point             → specification_rossler_chaotic_numpy.txt

All other datasets (e.g. lotka-volterra-simple, van-der-pol-oscillator, …):
  1D → specification_ode_1d_numpy.txt
  2D → specification_ode_2d_numpy.txt
  3D → specification_ode_3d_numpy.txt

Usage
-----
    cd <repo-root>

    # Dataset with a dedicated spec (spec chosen automatically):
    conda run -n aces python active_aces.py \
        --data_path LLM-ACES/data/ode/lorenz-equations-chaotic/lorenz-equations-chaotic.npz

    # Generic dataset (generic 2D spec used automatically):
    conda run -n aces python active_aces.py \
        --data_path LLM-ACES/data/ode/lotka-volterra-simple/lotka-volterra-simple.npz

    # With a held-out test set and OpenAI API:
    conda run -n aces python active_aces.py \
        --data_path  LLM-ACES/data/ode/lorenz-equations-chaotic/lorenz-equations-chaotic.npz \
        --test_path  LLM-ACES/data/ode/lorenz-equations-chaotic/lorenz-equations-chaotic_snr_10.npz \
        --use_api True --api_model gpt-4o-mini \
        --n_iterations 20 --samples_per_prompt 8
"""

from __future__ import annotations

import argparse
import ast
import copy
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Add LLM-ACES and scripts to Python path
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_ACES_DIR = _HERE / 'LLM-ACES'
_SCRIPTS_DIR = _HERE / 'scripts'
sys.path.insert(0, str(_ACES_DIR))
sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    # Upstream ACES modules. Some environments (like this repo's LLM-ACES subset)
    # don't vendor the full stack; keep imports optional so helpers like
    # `select_spec`, `load_true_ode`, and `query_oracle_ic` remain usable.
    from aces import code_manipulation  # type: ignore
    from aces import buffer as buffer_lib  # type: ignore
    from aces import config as config_lib  # type: ignore
    from aces import evaluator as evaluator_lib  # type: ignore
    from aces import sampler as sampler_lib  # type: ignore
except Exception:  # pragma: no cover
    code_manipulation = None  # type: ignore
    buffer_lib = None  # type: ignore
    config_lib = None  # type: ignore
    evaluator_lib = None  # type: ignore
    sampler_lib = None  # type: ignore


# ===========================================================================
# Argument parsing
# ===========================================================================

parser = argparse.ArgumentParser(
    description="Active LLM-ACES: LLM equation ensemble + pool-based active sampling."
)
parser.add_argument('--data_path', type=str, required=True,
                    help="Path to MDBench ODE NPZ file (contains t, u, du).")
parser.add_argument('--spec_path', type=str, default=None,
                    help="Path to LLM-ACES spec. Auto-selected by ODE dim if omitted.")
parser.add_argument('--log_path', type=str, default=None,
                    help="Output directory. Defaults to ./logs/active_aces/<name>.")
parser.add_argument('--n_iterations', type=int, default=5,
                    help="Number of active learning iterations.")
parser.add_argument('--samples_per_prompt', type=int, default=8,
                    help="Number of LLM equations generated per iteration (ensemble size).")
parser.add_argument('--n_init', type=int, default=30,
                    help="Number of points held out from the tail of the dataset as val/test. "
                         "D_train starts with total-n_init points (e.g. 150-30=120 for mdbench data).")
parser.add_argument('--timeout', type=int, default=60,
                    help="Sandbox execution timeout per equation (seconds).")
parser.add_argument('--use_api', type=lambda x: x.lower() == 'true', default=False,
                    help="Use OpenAI API instead of local LLM server.")
parser.add_argument('--api_model', type=str, default='gpt-4o-mini',
                    help="OpenAI model name when --use_api True.")
parser.add_argument('--gamma', type=float, default=1.0,
                    help="Unused — kept for CLI compatibility.")
parser.add_argument('--seed', type=int, default=42,
                    help="Random seed for reproducibility.")
parser.add_argument('--n_virtual', type=int, default=2,
                    help="Number of candidate ICs evaluated per acquisition step (0 to disable).")
parser.add_argument('--acq_method', choices=['random', 'bo', 'odeformer'], default='bo',
                    help="IC acquisition method: random disagreement search or BO over disagreement.")
parser.add_argument('--bo_init_points', type=int, default=None,
                    help="Initial random BO evaluations. Defaults to min(8, n_virtual).")
parser.add_argument('--bo_candidate_pool', type=int, default=256,
                    help="Number of candidate ICs in the BO search pool.")
parser.add_argument('--bo_kappa', type=float, default=2.0,
                    help="GP-UCB exploration weight for BO acquisition.")
parser.add_argument('--ic_domain_margin', type=float, default=0.2,
                    help="Margin used to expand train-state min/max bounds for candidate ICs.")
parser.add_argument('--temperature', type=float, default=1.0,
                    help="LLM sampling temperature.")
parser.add_argument('--num_islands', type=int, default=2,
                    help="Number of experience-buffer islands.")


# ===========================================================================
# Spec auto-selection
# ===========================================================================

# In this repo, specs live at `llmaces/LLM-ACES/specs/` (next to this file).
# Some upstream layouts use `LLM-ACES/specs/`. Support both.
SPECS_DIR = (_HERE / "specs") if (_HERE / "specs").exists() else (_ACES_DIR / "specs")

# Generic fallback specs, keyed by ODE dimensionality.
DIM_TO_SPEC = {
    1: SPECS_DIR / 'specification_ode_1d_numpy.txt',
    2: SPECS_DIR / 'specification_ode_2d_numpy.txt',
    3: SPECS_DIR / 'specification_ode_3d_numpy.txt',
    4: SPECS_DIR / 'specification_ode_4d_numpy.txt',
}

# Dataset-specific specs.  Keys are the NPZ stem (data_path.stem) that the
# dataset produces.  Multiple stems can map to the same spec (e.g. all Lorenz
# variants share the chaotic-Lorenz spec; all Rössler variants share the
# Rössler spec).  When a stem matches, this spec is used instead of the
# generic dim-based fallback.
NAME_TO_SPEC: dict = {}  # all systems now resolved via DIM_TO_SPEC


def select_spec(dim: int, override: Optional[str], system_name: str = '') -> str:
    """Return the spec path to use for this run.

    Priority order:
      1. --spec_path CLI override (always honoured if given).
      2. Dataset-specific spec from NAME_TO_SPEC (matched on system_name stem).
      3. Generic dimension-based fallback from DIM_TO_SPEC.
    """
    if override:
        return override
    if system_name in NAME_TO_SPEC:
        return str(NAME_TO_SPEC[system_name])
    if dim not in DIM_TO_SPEC:
        raise ValueError(
            f"No built-in spec for '{system_name}' ({dim}D ODE). "
            f"Pass --spec_path explicitly."
        )
    return str(DIM_TO_SPEC[dim])


def _parse_max_nparams(spec_text: str) -> int:
    for line in spec_text.splitlines():
        stripped = line.strip()
        if stripped.startswith('MAX_NPARAMS') and '=' in stripped:
            try:
                return int(stripped.split('=')[-1].strip())
            except ValueError:
                pass
    return 10


# ===========================================================================
# Ground-truth ODE oracle (loaded from strogatz_ode.py)
# ===========================================================================

def load_true_ode(system_name: str) -> Optional[Callable]:
    """Return a callable RHS f(t, x) -> np.ndarray for the named system.

    Matches the NPZ stem (e.g. 'lotka-volterra-simple') to the 'name' field
    in strogatz_ode.equations (e.g. 'Lotka-Volterra simple') by slugifying.
    Returns None if the system is not found or sympy is unavailable.
    """
    try:
        import sympy as sp
        import strogatz_ode
    except ImportError:
        return None

    slug = lambda s: s.lower().replace(' ', '-').replace('_', '-')
    target = slug(system_name)

    eq_dict = None
    for entry in strogatz_ode.equations:
        if slug(entry['name']) == target:
            eq_dict = entry
            break

    if eq_dict is None:
        return None

    dim = eq_dict['dim']
    consts = eq_dict['consts'][0]               # first (and usually only) param set
    eq_string = eq_dict['eq']
    individual_eqs = eq_string.split('|')

    var_symbols   = sp.symbols([f'x_{i}' for i in range(dim)])
    const_symbols = sp.symbols([f'c_{i}' for i in range(len(consts))])
    const_subs    = dict(zip(const_symbols, consts))

    lambdas = []
    for expr_str in individual_eqs:
        expr = sp.sympify(expr_str).subs(const_subs)
        lambdas.append(sp.lambdify(var_symbols, expr, 'numpy'))

    def true_rhs(t: float, x: np.ndarray) -> np.ndarray:
        return np.array([float(f(*x)) for f in lambdas])

    return true_rhs


# ===========================================================================
# Data split helpers
# ===========================================================================

def split_pool(
    t: np.ndarray,
    u: np.ndarray,
    du: np.ndarray,
    n_holdout: int,
    rng: np.random.Generator,
    t_init_max: float = 1.0,
) -> Tuple[dict, dict]:
    """Split the dataset into D_train (first total-n_holdout pts) and val (last n_holdout pts).

    For mdbench data (150 pts over [0, 10]) with n_holdout=30, this gives:
      D_train: indices 0..119  (t in [0, 8], 120 pts — matches oracle-added trajectory size)
      val:     indices 120..149 (t in [8, 10], 30 pts — held-out test portion)
    """
    def _subset(arr_idx):
        return {'t': t[arr_idx], 'u': u[arr_idx], 'du': du[arr_idx],
                'idx': arr_idx}

    n_total = len(t)
    n_train = max(n_total - n_holdout, 1)
    train_idx = np.arange(n_train)
    val_idx   = np.arange(n_train, n_total) if n_train < n_total else np.arange(n_total)
    return _subset(train_idx), _subset(val_idx)



# ===========================================================================
# LLM equation compilation, param optimisation, prediction
# ===========================================================================

def _compile_equation(program_str: str, function_to_evolve: str) -> Optional[Callable]:
    """exec the full program string; return the equation callable or None."""
    namespace: dict = {}
    try:
        exec(compile(program_str, '<llm_eq>', 'exec'), namespace)  # noqa: S102
        fn = namespace.get(function_to_evolve)
        return fn if callable(fn) else None
    except Exception:
        return None


def _optimize_params(
    eq_fn: Callable,
    u: np.ndarray,
    t: np.ndarray,
    du: np.ndarray,
    dim: int,
    max_nparams: int,
) -> np.ndarray:
    """Nelder-Mead fit of free params to minimise MSE on training data."""
    def loss(params: np.ndarray) -> float:
        try:
            args = [u[:, i] for i in range(dim)] + [t, params]
            result = eq_fn(*args)
            if not isinstance(result, tuple):
                result = (result,)
            du_pred = np.stack([np.asarray(r, dtype=float) for r in result], axis=1)
            if du_pred.shape != du.shape or not np.all(np.isfinite(du_pred)):
                return 1e10
            return float(np.mean((du_pred - du) ** 2))
        except Exception:
            return 1e10

    res = minimize(
        loss,
        [1.0] * max_nparams,
        method='Nelder-Mead',
        options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True},
    )
    return res.x


def _predict_at_states(
    eq_fn: Callable,
    u_cand: np.ndarray,
    t_cand: np.ndarray,
    params: np.ndarray,
    dim: int,
) -> Optional[np.ndarray]:
    """Evaluate eq_fn at N candidate states; return (N, dim) array or None."""
    try:
        args = [u_cand[:, i] for i in range(dim)] + [t_cand, params]
        result = eq_fn(*args)
        if not isinstance(result, tuple):
            result = (result,)
        preds = np.stack([np.asarray(r, dtype=float) for r in result], axis=1)
        if preds.shape[1] != dim or not np.all(np.isfinite(preds)):
            return None
        return preds  # (N, dim)
    except Exception:
        return None


# ===========================================================================
# IC acquisition: information gain via inter-equation trajectory divergence
# ===========================================================================

def compute_ic_acquisition(
    compiled_eqs: List[Tuple[Callable, np.ndarray]],
    candidate_u0s: np.ndarray,
    t_query: np.ndarray,
    dim: int,
) -> np.ndarray:
    """Score candidate ICs by median pairwise MSE of predicted derivatives at u0.

    Uses median instead of mean over pairwise MSEs so that a single outlier
    equation (with very bad fit) cannot dominate the score. For the median to
    be large, the MAJORITY of equation pairs must genuinely disagree at u0,
    which is the desired signal for active IC selection.

    Args:
        compiled_eqs:   list of (eq_fn, optimised_params).
        candidate_u0s:  (M, dim) candidate initial conditions.
        t_query:        time grid (only t_query[0] is used as the eval time).
        dim:            state dimensionality.

    Returns:
        (M,) acquisition scores.
    """
    t0 = np.array([float(t_query[0])])
    M  = len(candidate_u0s)
    scores = np.zeros(M)

    for m, u0 in enumerate(candidate_u0s):
        u0_arr = np.asarray(u0, dtype=float).reshape(1, dim)
        predictions: List[np.ndarray] = []

        for eq_fn, params in compiled_eqs:
            pred = _predict_at_states(eq_fn, u0_arr, t0, params, dim)
            if pred is not None:
                predictions.append(pred[0])  # shape (dim,)

        if len(predictions) < 2:
            continue

        # Drop predictions that have blown up — equations exploding at ICs
        # outside training distribution should not influence the score.
        valid = [
            p for p in predictions
            if np.all(np.isfinite(p)) and np.linalg.norm(p) < 1e6
        ]
        if len(valid) < 2:
            continue

        # Normalize by the mean prediction magnitude so the score measures
        # relative disagreement, not absolute derivative scale.
        pred_norm = np.linalg.norm(np.mean(valid, axis=0)) + 1e-8
        normalized = [p / pred_norm for p in valid]

        pairwise_mses = []
        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                pairwise_mses.append(float(np.mean((normalized[i] - normalized[j]) ** 2)))

        scores[m] = float(np.median(pairwise_mses)) if pairwise_mses else 0.0

    return scores


def _ic_domain_bounds(
    train: dict,
    dim: int,
    margin: float = 0.2,
    min_width: float = 1e-2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return expanded per-dimension IC bounds from observed training states."""
    u_train = np.asarray(train['u'], dtype=float)
    lows = np.min(u_train, axis=0)
    highs = np.max(u_train, axis=0)
    widths = highs - lows

    t0_idx = int(np.argmin(train['t']))
    u0_base = u_train[t0_idx].astype(float)
    tiny = widths < min_width
    lows = lows - margin * np.maximum(widths, min_width)
    highs = highs + margin * np.maximum(widths, min_width)

    # If the observed range is degenerate, use a small box around the earliest state.
    lows[tiny] = u0_base[tiny] - min_width
    highs[tiny] = u0_base[tiny] + min_width

    return lows[:dim], highs[:dim]


def _sample_uniform_u0s(
    lows: np.ndarray,
    highs: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return rng.uniform(lows, highs, size=(n, len(lows)))


def select_u0_random(
    compiled_eqs: List[Tuple[Callable, np.ndarray]],
    train: dict,
    t_query: np.ndarray,
    dim: int,
    rng: np.random.Generator,
    budget: int,
) -> Tuple[np.ndarray, float, dict]:
    """Original local Gaussian proposal, kept as a fallback/comparison path."""
    t0_idx = int(np.argmin(train['t']))
    u0_base = train['u'][t0_idx].astype(float)
    noise_scale = np.maximum(train['u'].std(axis=0) * 0.2, 1e-2)
    candidate_u0s = u0_base + rng.normal(0.0, noise_scale, size=(budget, dim))
    ic_scores = compute_ic_acquisition(compiled_eqs, candidate_u0s, t_query, dim)
    best_ic_idx = int(np.argmax(ic_scores))
    return candidate_u0s[best_ic_idx], float(ic_scores[best_ic_idx]), {
        'method': 'random',
        'n_evaluated': int(len(candidate_u0s)),
        'fallback': False,
    }


def select_u0_bo(
    compiled_eqs: List[Tuple[Callable, np.ndarray]],
    train: dict,
    t_query: np.ndarray,
    dim: int,
    rng: np.random.Generator,
    budget: int,
    candidate_pool: int = 256,
    init_points: Optional[int] = None,
    kappa: float = 2.0,
    margin: float = 0.2,
) -> Tuple[np.ndarray, float, dict]:
    """Bayesian optimisation over ICs, using disagreement as the black-box objective."""
    if budget <= 0:
        raise ValueError("BO acquisition requires budget > 0.")
    if init_points is None:
        init_points = min(8, budget)
    init_points = int(np.clip(init_points, 1, budget))
    candidate_pool = max(int(candidate_pool), budget)

    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    except Exception as exc:
        best_u0, best_score, log = select_u0_random(
            compiled_eqs, train, t_query, dim, rng, budget
        )
        log.update({
            'method': 'random',
            'fallback': True,
            'fallback_reason': f'sklearn unavailable: {exc}',
        })
        return best_u0, best_score, log

    lows, highs = _ic_domain_bounds(train, dim, margin=margin)
    candidate_u0s = _sample_uniform_u0s(lows, highs, candidate_pool, rng)
    candidate_unit = (candidate_u0s - lows) / (highs - lows + 1e-12)

    all_indices = np.arange(candidate_pool)
    rng.shuffle(all_indices)
    evaluated = list(all_indices[:init_points])
    remaining = list(all_indices[init_points:])

    scores = {}

    def _evaluate(indices: List[int]) -> None:
        if not indices:
            return
        vals = compute_ic_acquisition(
            compiled_eqs, candidate_u0s[np.array(indices)], t_query, dim
        )
        for idx, val in zip(indices, vals):
            scores[int(idx)] = float(val)

    _evaluate(evaluated)

    while len(scores) < budget and remaining:
        x_train = candidate_unit[np.array(list(scores.keys()))]
        y_train = np.array([scores[i] for i in scores.keys()], dtype=float)

        if len(np.unique(y_train)) <= 1:
            next_idx = int(remaining.pop(0))
        else:
            kernel = (
                ConstantKernel(1.0, (1e-3, 1e3))
                * Matern(length_scale=np.ones(dim), nu=2.5)
                + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2))
            )
            try:
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    normalize_y=True,
                    n_restarts_optimizer=2,
                    random_state=int(rng.integers(0, 2**31 - 1)),
                )
                gp.fit(x_train, y_train)
                x_rem = candidate_unit[np.array(remaining)]
                mu, std = gp.predict(x_rem, return_std=True)
                ucb = mu + kappa * std
                next_pos = int(np.argmax(ucb))
                next_idx = int(remaining.pop(next_pos))
            except Exception:
                next_idx = int(remaining.pop(0))

        _evaluate([next_idx])

    best_idx = max(scores, key=scores.get)
    return candidate_u0s[best_idx], float(scores[best_idx]), {
        'method': 'bo',
        'n_evaluated': int(len(scores)),
        'candidate_pool': int(candidate_pool),
        'init_points': int(init_points),
        'kappa': float(kappa),
        'bounds_low': lows.tolist(),
        'bounds_high': highs.tolist(),
        'fallback': False,
    }


def query_oracle_ic(
    true_rhs: Callable,
    u0: np.ndarray,
    t_query: np.ndarray,
) -> Optional[dict]:
    """Integrate the ground-truth ODE from u0 over t_query.

    Returns dict with 't', 'u', 'du' (all ground truth), or None on failure.
    """
    from scipy.integrate import solve_ivp

    t_start = float(t_query[0])
    t_end   = float(t_query[-1])

    try:
        sol = solve_ivp(
            true_rhs, [t_start, t_end], u0.tolist(),
            t_eval=t_query, method='LSODA',
            rtol=1e-5, atol=1e-7,
            first_step=1e-6, min_step=1e-10,
        )
        if not sol.success or sol.y.shape[1] != len(t_query):
            return None

        u_traj = sol.y.T  # (T, dim)
        if not np.all(np.isfinite(u_traj)):
            return None

        du_traj = np.stack([
            np.asarray(true_rhs(float(t_query[i]), u_traj[i]), dtype=float)
            for i in range(len(t_query))
        ])
        if not np.all(np.isfinite(du_traj)):
            return None

        return {'t': t_query.copy(), 'u': u_traj, 'du': du_traj}
    except Exception:
        return None


# ===========================================================================
# Param pruning: remove near-zero terms and re-index params
# ===========================================================================

def _param_index(node: ast.AST) -> Optional[int]:
    """Return the integer index from a params[i] AST subscript node, or None."""
    if not (isinstance(node, ast.Subscript) and
            isinstance(node.value, ast.Name) and
            node.value.id == 'params'):
        return None
    s = node.slice
    if isinstance(s, ast.Index):   # Python 3.8 wraps in ast.Index
        s = s.value
    if isinstance(s, ast.Constant) and isinstance(s.value, int):
        return s.value
    return None


def prune_zero_params(body: str, params: np.ndarray,
                      rel_threshold: float = 0.01) -> str:
    """Remove terms whose param coefficient is negligible and re-index the rest.

    A param is considered zero if |p| < rel_threshold * max(|params|).
    The zeroed params[i] nodes are replaced with 0.0 in the AST, then
    standard algebraic simplifications (0*x→0, x+0→x, etc.) remove them
    from sums.  Remaining params are re-indexed starting from 0.

    Example
    -------
    body    : params[0]*x0 + params[1]*x1 + params[2]*x2
    params  : [3.5,  0.0001,  -2.1]           # params[1] ≈ 0
    returns : params[0]*x0 + params[1]*x2      # re-indexed: old[2]→new[1]
    """
    max_abs = float(np.max(np.abs(params))) if len(params) else 1.0
    zero_set = {i for i, p in enumerate(params)
                if abs(float(p)) < rel_threshold * (max_abs + 1e-30)}
    if not zero_set:
        return body

    try:
        tree = ast.parse(body, mode='exec')
    except SyntaxError:
        return body

    # Collect non-zero param indices in ascending order → deterministic re-index
    used_nonzero = sorted({_param_index(n) for n in ast.walk(tree)
                           if _param_index(n) is not None} - zero_set)
    old_to_new = {old: new for new, old in enumerate(used_nonzero)}

    class _Transformer(ast.NodeTransformer):
        def visit_Subscript(self, node):
            self.generic_visit(node)
            idx = _param_index(node)
            if idx is None:
                return node
            if idx in zero_set:
                return ast.Constant(value=0.0)
            return ast.Subscript(
                value=ast.Name(id='params', ctx=ast.Load()),
                slice=ast.Constant(value=old_to_new.get(idx, idx)),
                ctx=ast.Load(),
            )

        def visit_BinOp(self, node):
            self.generic_visit(node)
            L, R, op = node.left, node.right, node.op

            def _z(n):
                return isinstance(n, ast.Constant) and n.value == 0.0

            if isinstance(op, ast.Mult) and (_z(L) or _z(R)):
                return ast.Constant(value=0.0)
            if isinstance(op, ast.Add):
                if _z(R): return L
                if _z(L): return R
            if isinstance(op, ast.Sub):
                if _z(R): return L
                if _z(L): return ast.UnaryOp(op=ast.USub(), operand=R)
            return node

        def visit_UnaryOp(self, node):
            self.generic_visit(node)
            if (isinstance(node.op, ast.USub) and
                    isinstance(node.operand, ast.Constant) and
                    node.operand.value == 0.0):
                return ast.Constant(value=0.0)
            return node

    new_tree = _Transformer().visit(tree)
    ast.fix_missing_locations(new_tree)
    try:
        return ast.unparse(new_tree)
    except Exception:
        return body


# ===========================================================================
# Data-aware prompt construction
# ===========================================================================

def build_data_aware_prompt(
    prompt_program: str,
    best_body: str,
    best_nmse: float,
    iteration: int,
    pruned_body: str = '',
) -> str:
    """Prefix an experience-buffer prompt with the best equation found so far.

    If pruned_body is provided it is shown instead of best_body — it has
    near-zero param terms removed and params re-indexed, giving the LLM a
    cleaner skeleton to build on.
    """
    display_body = pruned_body if pruned_body else best_body
    prev_block = ''
    if display_body and best_nmse < float('inf'):
        prev_block = (
            f"\nBest equation found so far (train NMSE={best_nmse:.4e}):\n"
            + '\n'.join('    ' + ln for ln in display_body.splitlines())
            + '\n'
        )

    prompt_prefix = (
        f"Improved ODE right-hand side — active iteration {iteration + 1}."
        + prev_block
    )
    return prompt_prefix + '\n\n' + prompt_program



# ===========================================================================
# Main active learning loop
# ===========================================================================

def run_active(
    system_name: str,
    spec_text: str,
    train: dict,
    test: dict,
    dim: int,
    n_iterations: int,
    samples_per_prompt: int,
    cfg: config_lib.Config,
    timeout: int,
    gamma: float,
    log_path: str,
    rng: np.random.Generator = None,
    n_virtual: int = 20,
    true_rhs: Optional[Callable] = None,
    acq_method: str = 'bo',
    bo_init_points: Optional[int] = None,
    bo_candidate_pool: int = 256,
    bo_kappa: float = 2.0,
    ic_domain_margin: float = 0.2,
):
    """Active LLM-ACES pipeline.

    D_train starts as the first 120 pts (t in [0,8]) of the mdbench data and
    grows each iteration as the oracle adds 120-pt trajectories from new ICs.
    test is the last 30 pts (t in [8,10]) and is FIXED throughout — never added
    to D_train. Both train NMSE and test NMSE are reported after every param fit.
    """
    if rng is None:
        rng = np.random.default_rng()
    if true_rhs is None:
        true_rhs = load_true_ode(system_name)
    if true_rhs is None and n_virtual > 0:
        print("  [INFO] No ground-truth ODE found; IC acquisition disabled.")
        n_virtual = 0

    template = code_manipulation.text_to_program(spec_text)
    function_to_evolve = list(
        code_manipulation.yield_decorated(spec_text, 'equation', 'evolve'))[0]
    function_to_run = list(
        code_manipulation.yield_decorated(spec_text, 'evaluate', 'run'))[0]
    max_nparams = _parse_max_nparams(spec_text)

    llm     = sampler_lib.LocalLLM(samples_per_prompt)
    sandbox = evaluator_lib.LocalSandbox()
    database = buffer_lib.ExperienceBuffer(
        cfg.experience_buffer, template, function_to_evolve
    )

    best_nmse: float = float('inf')
    best_body: str   = ''
    best_pruned_body: str = ''
    best_test_nmse: float = float('inf')
    best_test_body: str   = ''

    os.makedirs(log_path, exist_ok=True)
    results_path = os.path.join(log_path, 'active_aces_results.jsonl')
    open(results_path, 'w').close()

    # val_inputs is rebuilt each iteration from the growing train set (see loop).

    # Dense time grid for IC acquisition over [0, 8] — 120 samples so the
    # oracle trajectory covers the full observed time range.
    t_ic_query = np.linspace(0.0, 8.0, 120)

    print(f"\nActive LLM-ACES  —  system: '{system_name}'")
    print(f"  dim={dim}  max_nparams={max_nparams}")
    print(f"  n_iterations={n_iterations}  samples_per_prompt={samples_per_prompt}")
    print(f"  D_train: {len(train['t'])} pts (grows via oracle queries)  |  test (fixed): {len(test['t'])} pts")
    print(f"  n_virtual={n_virtual}  gamma={gamma}")
    print(f"  acq_method={acq_method}  bo_candidate_pool={bo_candidate_pool}  bo_kappa={bo_kappa}")
    print(f"  experience_buffer.num_islands={cfg.experience_buffer.num_islands}")
    print(f"  Logs -> {results_path}\n")

    seed_inputs = {'data': {'t': train['t'], 'u': train['u'], 'du': train['du']}}
    seed_program = str(template)
    seed_score, seed_ok = sandbox.run(
        program=seed_program,
        function_to_run=function_to_run,
        function_to_evolve=function_to_evolve,
        inputs=seed_inputs,
        test_input='data',
        timeout_seconds=timeout,
    )
    if not seed_ok or seed_score is None:
        raise RuntimeError("Failed to seed the experience buffer from the template equation.")

    seed_fn = copy.deepcopy(template.get_function(function_to_evolve))
    if not evaluator_lib._calls_ancestor(seed_program, function_to_evolve):
        database.register_program(
            seed_fn,
            island_id=None,
            scores_per_test={'data': float(seed_score)},
        )
    seed_nmse = float(-seed_score)
    best_nmse = seed_nmse
    best_body = seed_fn.body.strip()
    best_pruned_body = best_body  # no params fitted yet; keep as-is
    print(f"  Experience buffer seeded with template equation (NMSE={seed_nmse:.4e}).\n")

    for it in range(n_iterations):
        t_iter = time.time()
        print(f"{'='*65}")
        print(f"Iter {it+1}/{n_iterations}  |  D_train={len(train['t'])} pts")

        # ----------------------------------------------------------
        # Step 0: Rebuild sandbox inputs from the current train set.
        # ----------------------------------------------------------
        print(f"  [Step 0] Rebuilding sandbox inputs from D_train ({len(train['t'])} pts)...")
        val_inputs = {'data': {'t': train['t'], 'u': train['u'], 'du': train['du']}}

        # ----------------------------------------------------------
        # Step 1: Build data-aware prompt (stats + best equation; no raw rows).
        # ----------------------------------------------------------
        print(f"  [Step 1] Building prompt from experience buffer...")
        prompt_info = database.get_prompt()
        prompt = build_data_aware_prompt(
            prompt_info.code,
            best_body, best_nmse, iteration=it,
            pruned_body=best_pruned_body,
        )
        print(f"  [Step 1] Prompt built ({len(prompt)} chars, island={prompt_info.island_id}, "
              f"version={prompt_info.version_generated}).")

        # ----------------------------------------------------------
        # Step 2: Query LLM for equation candidates.
        # ----------------------------------------------------------
        print(f"  [Step 2] Querying LLM for {samples_per_prompt} equation candidates...")
        t_llm = time.time()
        try:
            raw_samples = llm.draw_samples(prompt, cfg)
        except Exception as exc:
            print(f"  [Step 2] [WARN] LLM error: {exc}")
            raw_samples = []
        print(f"  [Step 2] LLM returned {len(raw_samples)} samples  ({time.time()-t_llm:.1f}s).")

        # ----------------------------------------------------------
        # Step 3: Parse, sandbox-score, compile, and param-fit each sample.
        # ----------------------------------------------------------
        print(f"  [Step 3] Parsing and scoring {len(raw_samples)} samples...")
        compiled_eqs: List[Tuple[Callable, np.ndarray]] = []
        eq_logs: List[dict] = []

        for si, raw in enumerate(raw_samples):
            print(f"    [Step 3.{si+1}/{len(raw_samples)}] Parsing sample {si}...", end=' ', flush=True)
            try:
                new_fn, program_str = evaluator_lib._sample_to_program(
                    generated_code=raw,
                    version_generated=prompt_info.version_generated,
                    template=template,
                    function_to_evolve=function_to_evolve,
                )
            except Exception:
                new_fn, program_str = None, None

            if program_str is None:
                print("parse failed.")
                eq_logs.append({'si': si, 'ok': False, 'nmse': None, 'body': ''})
                continue

            print("parsed.", end=' ', flush=True)

            # Sandbox: evaluate() in spec fits params and scores NMSE on full train set
            print("sandbox...", end=' ', flush=True)
            score, ok = sandbox.run(
                program=program_str,
                function_to_run=function_to_run,
                function_to_evolve=function_to_evolve,
                inputs=val_inputs,
                test_input='data',
                timeout_seconds=timeout,
            )
            nmse = float(-score) if (ok and score is not None) else None
            body = new_fn.body.strip() if new_fn else ''

            if (ok and score is not None and new_fn is not None
                    and not evaluator_lib._calls_ancestor(program_str, function_to_evolve)):
                database.register_program(
                    new_fn,
                    island_id=prompt_info.island_id,
                    scores_per_test={'data': float(score)},
                )

            if nmse is not None:
                print(f"NMSE={nmse:.4e}.", end=' ', flush=True)
            else:
                print("sandbox failed.", end=' ', flush=True)

            if nmse is not None and nmse < best_nmse:
                best_nmse = nmse
                best_body = body
                best_pruned_body = body  # params not fitted yet; pruned in fit block below
                print(f"[NEW BEST]", end=' ', flush=True)

            eq_logs.append({'si': si, 'ok': nmse is not None, 'nmse': nmse, 'body': body,
                            'test_nmse': None, 'test_rmse': None})

            # Compile + fit params on full accumulated train set, then score on
            # both train and the fixed held-out test set.
            print("compiling...", end=' ', flush=True)
            eq_fn = _compile_equation(program_str, function_to_evolve)
            if eq_fn is not None:
                try:
                    print("fitting params...", end=' ', flush=True)
                    params = _optimize_params(
                        eq_fn, train['u'], train['t'], train['du'], dim, max_nparams
                    )
                    compiled_eqs.append((eq_fn, params))

                    # Train NMSE from fitted params
                    train_pred = _predict_at_states(eq_fn, train['u'], train['t'], params, dim)
                    if train_pred is not None:
                        train_norm = float(np.sum(train['du'] ** 2)) + 1e-10
                        fit_train_nmse = float(np.sum((train_pred - train['du']) ** 2)) / train_norm
                        eq_logs[-1]['fit_train_nmse'] = fit_train_nmse
                        if fit_train_nmse < best_nmse:
                            best_nmse = fit_train_nmse
                            best_body = body
                            best_pruned_body = prune_zero_params(body, params)
                            print(f"train NMSE={fit_train_nmse:.4e} [NEW BEST]", end=' ', flush=True)
                        else:
                            print(f"train NMSE={fit_train_nmse:.4e}", end=' ', flush=True)

                    # Test NMSE (fixed held-out set — never added to D_train)
                    test_pred = _predict_at_states(eq_fn, test['u'], test['t'], params, dim)
                    if test_pred is not None:
                        test_norm = float(np.sum(test['du'] ** 2)) + 1e-10
                        test_nmse = float(np.sum((test_pred - test['du']) ** 2)) / test_norm
                        test_rmse = float(np.sqrt(np.mean((test_pred - test['du']) ** 2)))
                        eq_logs[-1]['test_nmse'] = test_nmse
                        eq_logs[-1]['test_rmse'] = test_rmse
                        if test_nmse < best_test_nmse:
                            best_test_nmse = test_nmse
                            best_test_body = body
                        print(f"test NMSE={test_nmse:.4e}", end=' ', flush=True)
                    else:
                        print("test pred failed.", end=' ', flush=True)
                except Exception as e:
                    print(f"param fit error: {e}.", end=' ', flush=True)
            else:
                print("compile failed.", end=' ', flush=True)
            print()  # newline after each sample's inline status

        n_ok       = sum(1 for e in eq_logs if e['ok'])
        n_compiled = len(compiled_eqs)
        iter_summary = f"  [Step 3] Done: scored {n_ok}/{len(raw_samples)}  |  compiled: {n_compiled}"
        if best_nmse < float('inf'):
            iter_summary += f"  |  best train NMSE: {best_nmse:.4e}"
        if best_test_nmse < float('inf'):
            iter_summary += f"  |  best test NMSE: {best_test_nmse:.4e}"
        print(iter_summary)

        # ----------------------------------------------------------
        # Step 4: IC acquisition — pick the most informative initial condition.
        # ----------------------------------------------------------
        n_ic_added = 0
        acq_log = None
        if n_virtual > 0 and n_compiled >= 2 and true_rhs is not None:
            print(f"  [Step 4] IC acquisition ({acq_method}) — scoring {n_virtual} candidate ICs "
                  f"across {n_compiled} compiled equations...")
            if acq_method == 'bo':
                best_u0, best_ic_score, acq_log = select_u0_bo(
                    compiled_eqs=compiled_eqs,
                    train=train,
                    t_query=t_ic_query,
                    dim=dim,
                    rng=rng,
                    budget=n_virtual,
                    candidate_pool=bo_candidate_pool,
                    init_points=bo_init_points,
                    kappa=bo_kappa,
                    margin=ic_domain_margin,
                )
            else:
                best_u0, best_ic_score, acq_log = select_u0_random(
                    compiled_eqs=compiled_eqs,
                    train=train,
                    t_query=t_ic_query,
                    dim=dim,
                    rng=rng,
                    budget=n_virtual,
                )

            print(f"  [Step 4] Best IC: divergence={best_ic_score:.4e}  "
                  f"evals={acq_log['n_evaluated']}  "
                  f"u0={np.round(best_u0, 3).tolist()}")
            if acq_log.get('fallback'):
                print(f"  [Step 4] Fallback to random: {acq_log.get('fallback_reason', 'unknown')}")

            # Step 5: Oracle query — integrate true ODE from best IC.
            print(f"  [Step 5] Querying oracle: integrating true ODE from best IC over [0, 8]...")
            oracle_data = query_oracle_ic(true_rhs, best_u0, t_ic_query)
            if oracle_data is not None:
                n_ic_added = len(oracle_data['t'])
                train = {
                    't':   np.concatenate([train['t'],   oracle_data['t']]),
                    'u':   np.concatenate([train['u'],   oracle_data['u']]),
                    'du':  np.concatenate([train['du'],  oracle_data['du']]),
                    'idx': np.concatenate([train['idx'], np.full(n_ic_added, -1, dtype=int)]),
                }
                print(f"  [Step 5] Oracle added {n_ic_added} pts → D_train={len(train['t'])} pts total.")
            else:
                print("  [Step 5] Oracle integration failed for best IC.")
        elif n_virtual > 0 and n_compiled < 2:
            print(f"  [Step 4] IC acquisition skipped — need ≥2 compiled equations, got {n_compiled}.")
        elif n_virtual == 0:
            print(f"  [Step 4] IC acquisition disabled (n_virtual=0).")

        # ----------------------------------------------------------
        # Log
        # ----------------------------------------------------------
        iter_log = {
            'iteration':        it + 1,
            'train_size':       int(len(train['t'])),  # val == train (no holdout)
            'n_samples':        len(raw_samples),
            'n_ok':             n_ok,
            'n_compiled':       n_compiled,
            'n_ic_added':       n_ic_added,
            'acquisition':      acq_log,
            'best_nmse_so_far':   best_nmse if best_nmse < float('inf') else None,
            'best_body_so_far':   best_body,
            'best_pruned_so_far': best_pruned_body,
            'equations':        eq_logs,
            'wall_time_s':      round(time.time() - t_iter, 2),
        }
        with open(results_path, 'a') as f:
            f.write(json.dumps(iter_log) + '\n')

    # ------------------------------------------------------------------
    # Final evaluation: fit params on full accumulated train set (original
    # data + all oracle-added trajectories), compute RMSE on same set.
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"Finished.  system='{system_name}'")
    print(f"  Final D_train size : {len(train['t'])} pts")

    final_rmse = None
    if best_body:
        print("  [Final] Compiling best equation...")
        prog = copy.deepcopy(template)
        prog.get_function(function_to_evolve).body = best_body + '\n'
        eq_fn = _compile_equation(str(prog), function_to_evolve)
        if eq_fn is not None:
            print(f"  [Final] Fitting params on full D_train ({len(train['t'])} pts)...")
            params  = _optimize_params(eq_fn, train['u'], train['t'], train['du'], dim, max_nparams)
            print("  [Final] Evaluating on D_train...")
            du_pred = _predict_at_states(eq_fn, train['u'], train['t'], params, dim)
            if du_pred is not None:
                final_rmse = float(np.sqrt(np.mean((du_pred - train['du']) ** 2)))
                norm       = float(np.sum(train['du'] ** 2)) + 1e-10
                final_nmse = float(np.sum((du_pred - train['du']) ** 2)) / norm
                print(f"  Final RMSE (D_train) : {final_rmse:.4e}")
                print(f"  Final NMSE (D_train) : {final_nmse:.4e}")

            # --- Test-set final report (params fitted on train, eval on fixed test) ---
            test_pred = _predict_at_states(eq_fn, test['u'], test['t'], params, dim)
            if True:
                if test_pred is not None:
                    final_test_rmse = float(np.sqrt(np.mean((test_pred - test['du']) ** 2)))
                    test_norm       = float(np.sum(test['du'] ** 2)) + 1e-10
                    final_test_nmse = float(np.sum((test_pred - test['du']) ** 2)) / test_norm
                    print(f"  Final RMSE (D_test)  : {final_test_rmse:.4e}")
                    print(f"  Final NMSE (D_test)  : {final_test_nmse:.4e}")
                else:
                    print("  Final test eval: prediction failed for best equation.")

        print("  Best equation body:")
        for ln in best_body.splitlines():
            print(f"    {ln}")
    else:
        print("  No equation was successfully evaluated.")
    print(f"  Log : {results_path}")

    return final_rmse if final_rmse is not None else best_nmse, best_body


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == '__main__':
    args = parser.parse_args()
    rng  = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = _HERE / data_path
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    raw      = np.load(data_path)
    t_full   = raw['t']                         # (N,)
    u_full   = raw['u']                         # (N, dim)
    du_full  = raw['du']                        # (N, dim)

    if u_full.ndim == 1:
        u_full  = u_full[:, np.newaxis]
    if du_full.ndim == 1:
        du_full = du_full[:, np.newaxis]

    dim          = u_full.shape[1]
    system_name  = data_path.stem
    total_pts    = len(t_full)

    print(f"System   : {system_name}")
    print(f"Data     : t={t_full.shape}  u={u_full.shape}  du={du_full.shape}")

    # ------------------------------------------------------------------
    # Spec
    # ------------------------------------------------------------------
    spec_path = select_spec(dim, args.spec_path, system_name)
    print(f"Spec     : {spec_path}")
    with open(spec_path, encoding='utf-8') as f:
        spec_text = f.read()

    # ------------------------------------------------------------------
    # Split data: test = last 30 raw pts; train init = oracle 20 pts from
    # first IC at t∈[0,1], even-indexed (10 pts), matching active_llm_pysr.
    # ------------------------------------------------------------------
    true_rhs = load_true_ode(system_name)

    n_test = 30
    test_idx = np.arange(total_pts - n_test, total_pts)
    val = {'t': t_full[test_idx], 'u': u_full[test_idx], 'du': du_full[test_idx], 'idx': test_idx}

    t_init_query = np.linspace(0.0, 1.0, 20)
    u0_init = u_full[0]
    init_oracle = query_oracle_ic(true_rhs, u0_init, t_init_query) if true_rhs is not None else None
    if init_oracle is not None:
        all_init_idx = np.arange(20)
        train = {
            't':   init_oracle['t'][all_init_idx[0::2]],
            'u':   init_oracle['u'][all_init_idx[0::2]],
            'du':  init_oracle['du'][all_init_idx[0::2]],
            'idx': np.full(10, -1, dtype=int),
        }
    else:
        pool_idx = np.arange(total_pts - n_test)
        mask = t_full[pool_idx] <= 1.0
        early_idx = pool_idx[mask][:20]
        train_idx = early_idx[0::2]
        train = {'t': t_full[train_idx], 'u': u_full[train_idx], 'du': du_full[train_idx], 'idx': train_idx}

    print(f"Dataset  : D_train={len(train['t'])} pts (oracle init)  |  test={len(val['t'])} pts")
    print(f"Test set : {len(val['t'])} pts from t=[{val['t'].min():.2f}, {val['t'].max():.2f}]  (fixed, never used for training)")
    print()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    cfg      = config_lib.Config(
        use_api=args.use_api,
        api_model=args.api_model,
        temperature=args.temperature,
        experience_buffer=config_lib.ExperienceBufferConfig(
            num_islands=args.num_islands,
        ),
    )
    log_path = args.log_path or str(_HERE / 'logs' / 'active_aces' / system_name)

    print(f"LLM temp : {cfg.temperature}")
    print(f"Islands  : {cfg.experience_buffer.num_islands}")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    if true_rhs is not None:
        print(f"Oracle   : ground-truth ODE loaded for '{system_name}'")
    else:
        print(f"Oracle   : no ground-truth ODE found for '{system_name}' "
              f"(virtual IC sampling disabled)")

    run_active(
        system_name=system_name,
        spec_text=spec_text,
        train=train,
        test=val,
        dim=dim,
        n_iterations=args.n_iterations,
        samples_per_prompt=args.samples_per_prompt,
        cfg=cfg,
        timeout=args.timeout,
        gamma=args.gamma,
        log_path=log_path,
        rng=rng,
        n_virtual=args.n_virtual,
        true_rhs=true_rhs,
        acq_method=args.acq_method,
        bo_init_points=args.bo_init_points,
        bo_candidate_pool=args.bo_candidate_pool,
        bo_kappa=args.bo_kappa,
        ic_domain_margin=args.ic_domain_margin,
    )
