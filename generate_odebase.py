"""
Generate ODEBase NPZ datasets in the same format as the MDBench data/ode/ files.

NPZ arrays per file:
  t       (100,)          time points [0, 1], training window
  u       (100, n_vars)   IC-0 trajectory, training window
  du      (100, n_vars)   IC-0 derivatives, training window
  u_gen   (100, n_vars)   IC-1 trajectory, training window (generalization IC)
  du_gen  (100, n_vars)   IC-1 derivatives, training window
  t_ood   (150,)          time points (1, 10], OOD window
  u_ood   (150, n_vars)   IC-0 trajectory, OOD window
  du_ood  (150, n_vars)   IC-0 derivatives, OOD window

Output layout:
  data/odebase/odebase_vars2_prog<N>/odebase_vars2_prog<N>.npz   (2-D systems)
  data/odebase/odebase_vars3_prog<N>/odebase_vars3_prog<N>.npz   (3-D systems)

Usage:
  python generate_odebase.py                   # saves to data/odebase/
  python generate_odebase.py --save_dir /path  # custom output directory
  python generate_odebase.py --no_snr          # skip noisy variants
"""

import sys
import argparse
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm

# scibench equation definitions are bundled in scripts/scibench/
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from scibench.data.equation_odes_odebase import *   # noqa: F401,F403  registers all classes
from scibench.data.base import EQUATION_CLASS_DICT

# ---------------------------------------------------------------------------
# Time-grid constants (identical to MDBench / generate_ode.py convention)
# ---------------------------------------------------------------------------
T_TRAIN_END = 1.0
N_TRAIN     = 100
N_OOD       = 150

_t_train = np.linspace(0.0, T_TRAIN_END, N_TRAIN)
_t_ood   = np.linspace(T_TRAIN_END, 10.0, N_OOD + 1)[1:]  # exclude duplicate at 1.0
_t_all   = np.concatenate([_t_train, _t_ood])              # 250 points total

SOLVE_CFG = dict(
    t_span     = (0.0, 10.0),
    method     = "LSODA",
    rtol       = 1e-5,
    atol       = 1e-7,
    first_step = 1e-6,
    min_step   = 1e-10,
    t_eval     = _t_all,
)

SNR_LIST = [40, 30, 20, 10]


def sample_ic(eq):
    return np.array([sampler(1).item() for sampler in eq.vars_range_and_types])


def integrate(eq, ic):
    try:
        sol = solve_ivp(eq.np_eq, y0=ic, **SOLVE_CFG)
        if not sol.success or sol.y.shape[1] < N_TRAIN:
            return None
        X  = sol.y.T
        dX = np.array([eq.np_eq(sol.t[i], X[i]) for i in range(len(sol.t))])
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(dX)):
            return None
        return sol.t, X, dX
    except Exception:
        return None


def add_noise(arr, snr):
    sigma = np.sqrt(10 ** (-snr / 10))
    return arr * (1.0 + np.random.randn(*arr.shape) * sigma)


def generate_equation(eq, rng_seed, max_ic_attempts=20):
    np.random.seed(rng_seed)
    for _ in range(max_ic_attempts):
        ic0 = sample_ic(eq)
        r0  = integrate(eq, ic0)
        if r0 is None:
            continue
        t_all, X0, dX0 = r0

        ic1 = sample_ic(eq)
        r1  = integrate(eq, ic1)
        if r1 is None:
            continue
        _, X1, dX1 = r1

        return dict(
            t      = t_all[:N_TRAIN],
            u      = X0[:N_TRAIN],
            du     = dX0[:N_TRAIN],
            u_gen  = X1[:N_TRAIN],
            du_gen = dX1[:N_TRAIN],
            t_ood  = t_all[N_TRAIN:],
            u_ood  = X0[N_TRAIN:],
            du_ood = dX0[N_TRAIN:],
        )
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate ODEBase NPZ datasets")
    parser.add_argument("--save_dir", type=Path, default=Path("data/odebase"),
                        help="Output directory (default: data/odebase/)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_snr", action="store_true",
                        help="Skip noisy SNR variants (saves ~4x disk space)")
    args = parser.parse_args()

    snr_list = [] if args.no_snr else SNR_LIST

    odebase_classes = {
        name: cls for name, cls in EQUATION_CLASS_DICT.items()
        if cls._eq_name.startswith("odebase_vars2_") or cls._eq_name.startswith("odebase_vars3_")
    }
    odebase_classes = dict(sorted(odebase_classes.items(), key=lambda kv: kv[1]._eq_name))

    print(f"Found {len(odebase_classes)} ODEBase equations.")
    print(f"Saving to: {args.save_dir.resolve()}\n")

    failed = []
    for i, (cls_name, cls) in enumerate(tqdm(odebase_classes.items())):
        eq      = cls()
        eq_name = eq._eq_name
        seed    = args.seed + i

        arrays = generate_equation(eq, rng_seed=seed)
        if arrays is None:
            print(f"\n  [SKIP] {eq_name}: all IC attempts failed.")
            failed.append(eq_name)
            continue

        out = args.save_dir / eq_name / f"{eq_name}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **arrays)

        np.random.seed(seed + 10000)
        for snr in snr_list:
            noisy = {**arrays,
                     "u":     add_noise(arrays["u"],     snr),
                     "u_gen": add_noise(arrays["u_gen"], snr),
                     "u_ood": add_noise(arrays["u_ood"], snr)}
            np.savez(args.save_dir / eq_name / f"{eq_name}_snr_{snr}.npz", **noisy)

    print(f"\nDone. {len(odebase_classes) - len(failed)} saved, {len(failed)} failed.")
    if failed:
        print("Failed:", failed)


if __name__ == "__main__":
    main()
