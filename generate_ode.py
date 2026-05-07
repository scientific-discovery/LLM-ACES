
# Adapted from https://github.com/sdascoli/odeformer/tree/main

import re
import argparse
from pathlib import Path
from collections import namedtuple

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from tqdm import tqdm
import strogatz_ode

np.random.seed(0)

OdeDataset = namedtuple('OdeDataset', ['t', 'u', 'du', 'u_gen', 'du_gen', 't_ood', 'u_ood', 'du_ood'])

# Training window: [0, 1] with 100 points (IC 0 and IC 1 both solved here)
# OOD window:      (1, 10] with 150 points (IC 0 only)
T_TRAIN_END = 1.0
N_TRAIN = 100
N_OOD = 150

_t_train_eval = np.linspace(0, T_TRAIN_END, N_TRAIN)
_t_ood_eval   = np.linspace(T_TRAIN_END, 10.0, N_OOD + 1)[1:]  # exclude T_TRAIN_END duplicate

config = {
    "t_span": (0, 10),
    "method": "LSODA",
    "rtol": 1e-5,
    "atol": 1e-7,
    "first_step": 1e-6,
    "t_eval": np.concatenate([_t_train_eval, _t_ood_eval]),  # 250 points total
    "min_step": 1e-10,
}

matplotlib_rc = {
#'text': {'usetex': True},
'font': {'size': '16', 'family': 'serif'},#, 'serif': 'Palatino'},
'figure': {'titlesize': '20'},
'axes': {'titlesize': '22', 'labelsize': '28'},
'xtick': {'labelsize': '22'},
'ytick': {'labelsize': '22'},
'lines': {'linewidth': 3, 'markersize': 10},
'grid': {'color': 'grey', 'linestyle': 'solid', 'linewidth': 0.5},
}

def validate_equations(equations):
    """Validates the equations to make sure they are in the correct format.

    These are just a bunch of basic checks, which would probably all throw errors
    when trying to solve them anyway, but were useful to get the equations right
    in the beginning.
    """
    for eq_dict in equations:
        eq_string = eq_dict['eq']
        dim = eq_dict['dim']
        consts_values = eq_dict['consts']
        init_values = eq_dict['init']
        id = eq_dict['id']
        individual_eqs = eq_string.split('|')
        if len(individual_eqs) != dim:
            print(f"Error in equation {id}: The number of equations does not match the dimension.")

        highest_x_index = max([int(x[2:]) for x in re.findall(r'x_\d+', eq_string)])
        if highest_x_index + 1 != dim:
            pass #print(f"Warning in equation {id}: Found x_{highest_x_index} as highest index, but the dimension is {dim}.")

        const_indices = [int(c[2:]) for c in re.findall(r'c_\d+', eq_string)]
        if len(const_indices) > 0:
            highest_const_index = max(const_indices)
            for j in range(highest_const_index + 1):
                if f'c_{j}' not in eq_string:
                    print(f"Warning in equation {id}: c_{j} not appearing even though c_{highest_const_index} does.")
        for j, consts in enumerate(consts_values):
            if len(set(const_indices)) != len(consts):
                print(f"Warning in equation {id}, constants {j}: The number of constants does not match the number of constants in the equations.")

        for j, init in enumerate(init_values):
            if len(init) != dim:
                print(f"Error in equation {id}, init {j}: The number of initial values does not match the dimension of the equation.")
    print("VALIDATION DONE")


def process_equations(equations):
    """Create sympy expressions for each of the equations (and their different parameter values).
    We directly add the list of expressions to each dictionary.
    """
    validate_equations(equations)
    for eq_dict in equations:
        substituted_fns = create_substituted_functions(eq_dict)
        eq_dict['substituted'] = substituted_fns
    print("PROCESSING DONE")


def create_substituted_functions(eq_dict):
    """For a given equation, create sympy expressions where the different parameter values have been substituted in."""
    eq_string = eq_dict['eq']
    consts_values = eq_dict['consts']
    individual_eqs = eq_string.split('|')
    const_symbols = sp.symbols([f'c_{i}' for i in range(len(consts_values[0]))])
    parsed_eqs = [sp.sympify(eq) for eq in individual_eqs]

    substituted_fns = []
    for consts in consts_values:
        const_subs = dict(zip(const_symbols, consts))
        substituted_fns.append([eq.subs(const_subs) for eq in parsed_eqs])
    return substituted_fns

def solve_equations(equations, config):
    """Solve all equations for a given config.

    We add the solutions to each of the equations dictionary as a list of list of solution dictionaries.
    The list of list represents (number of parameter settings x number of initial conditions).
    """
    for eq_dict in tqdm(equations):
        eq_dict['solutions'] = []
        var_symbols = sp.symbols([f'x_{i}' for i in range(eq_dict['dim'])])
        for i, fns in enumerate(eq_dict['substituted']):
            eq_dict['solutions'].append([])
            callable_fn = lambda t, x: np.array([f(*x) for f in [sp.lambdify(var_symbols, eq, 'numpy') for eq in fns]])
            for initial_conditions in eq_dict['init']:
                sol = solve_ivp(callable_fn, **config, y0=initial_conditions)
                sol_dict = {
                    "success": sol.success,
                    "message": sol.message,
                    "t": sol.t.tolist(),
                    "y": sol.y.tolist(),
                    "nfev": int(sol.nfev),
                    "njev": int(sol.njev),
                    "nlu": int(sol.nlu),
                    "status": int(sol.status),
                }
                if sol.status != 0:
                    print(f"Error in equation {eq_dict['id']}: {eq_dict['eq_description']}, constants {i}, initial conditions {initial_conditions}: {sol.message}")
                sol_dict['consts'] = eq_dict['consts'][i]
                sol_dict['init'] = initial_conditions
                eq_dict['solutions'][i].append(sol_dict)
    print("SOLVING DONE")

def make_dataset(equation) -> OdeDataset:
    '''
    Make a dataset from an equation with three evaluation splits:

    - Train / Reconstruction: IC 0, t in [0, T_TRAIN_END], N_TRAIN points
    - Generalization:         IC 1, t in [0, T_TRAIN_END], N_TRAIN points (never seen during training)
    - OOD:                    IC 0, t in (T_TRAIN_END, 10], N_OOD points

    Both ICs are solved over the full [0, 10] window; only the relevant
    segments are stored.
    '''
    symbols = sp.symbols([f'x_{i}' for i in range(equation['dim'])])
    fns = [sp.lambdify(symbols, expr, 'numpy') for expr in equation['substituted'][0]]

    def _extract(sol_dict):
        t = np.array(sol_dict['t'])
        X = np.array(sol_dict['y']).T
        Y_true = np.zeros_like(X)
        for i, fn in enumerate(fns):
            Y_true[:, i] = fn(*X.T).T
        return t, X, Y_true

    # IC 0: split into train [0, T_TRAIN_END] and OOD (T_TRAIN_END, 10]
    t_all, X_all, Y_all = _extract(equation['solutions'][0][0])
    t_train, u_train, du_train = t_all[:N_TRAIN], X_all[:N_TRAIN], Y_all[:N_TRAIN]
    t_ood,   u_ood,   du_ood   = t_all[N_TRAIN:], X_all[N_TRAIN:], Y_all[N_TRAIN:]

    # IC 1: only keep the training window for generalization evaluation
    _, X_gen_all, Y_gen_all = _extract(equation['solutions'][0][1])
    u_gen, du_gen = X_gen_all[:N_TRAIN], Y_gen_all[:N_TRAIN]

    return OdeDataset(
        t=t_train, u=u_train, du=du_train,
        u_gen=u_gen, du_gen=du_gen,
        t_ood=t_ood, u_ood=u_ood, du_ood=du_ood,
    )

def add_noise(dataset: OdeDataset, snr: float) -> OdeDataset:
    """Add noise to a dataset.

    Noise is applied independently to each trajectory (train, gen, ood).
    True derivatives are never noised.
    """
    sigma2 = 10**(-snr/10)
    def _noisy(arr):
        return arr * (1 + np.random.randn(*arr.shape) * np.sqrt(sigma2))
    return OdeDataset(
        t=dataset.t, u=_noisy(dataset.u), du=dataset.du,
        u_gen=_noisy(dataset.u_gen), du_gen=dataset.du_gen,
        t_ood=dataset.t_ood, u_ood=_noisy(dataset.u_ood), du_ood=dataset.du_ood,
    )

def save_dataset(dataset: OdeDataset, path: Path):
    """Save a dataset to disk."""
    d = dataset._asdict()
    assert d['u'].shape == d['du'].shape
    assert d['u_gen'].shape == d['du_gen'].shape
    assert d['u_ood'].shape == d['du_ood'].shape
    assert d['t'].shape[0] == d['u'].shape[0]
    assert d['t_ood'].shape[0] == d['u_ood'].shape[0]
    np.savez(path, **d)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir',
                        type=Path,
                        default=Path('data/ode'),
                        help='Path to the directory to save the processed ODE equations and solutions'
                        )
    parser.add_argument('--snr',
                        type=list,
                        default=[40, 30, 20, 10],
                        help='Signal-to-noise ratio to add to the trajectories. Default is 0.0.'
                        )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    save_dir = args.save_dir

    equations = strogatz_ode.equations
    process_equations(equations)
    solve_equations(equations, config)
    for equation in equations:
        dataset_name = equation['name'].lower().replace(' ', '-')
        dataset = make_dataset(equation)
        save_path = save_dir / f'{dataset_name}' / f'{dataset_name}.npz'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_dataset(dataset, save_path)
        for snr in args.snr:
            noisy_dataset = add_noise(dataset, snr)
            save_path = save_dir / f'{dataset_name}' /f'{dataset_name}_snr_{snr}.npz'
            save_dataset(noisy_dataset, save_path)
