import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit
from scipy.special import ndtr, ndtri, logsumexp

N_SAMPLES = 100_000
N_RUNS = 10
TEMPERATURES = np.linspace(0.5, 5.0, 10)

L = 4.0
PRESSURE_STEP = 0.01 * L
BASE_SEED = 1234567
OUTPUT_DIR = Path("results")

BASE_PARAMETERS = {"a": 1.0, "b": 0.5, "gamma": 1.0}

RUN_PARAMETER_STUDY = True

PARAMETER_SETS = [
    {"name": "Baseline parameters", "a": 1.0, "b": 0.5, "gamma": 1.0},
    {"name": "Variation of a", "a": 0.7, "b": 0.5, "gamma": 1.0},
    {"name": "Variation of b", "a": 1.0, "b": 1.0, "gamma": 1.0},
    {"name": "Variation of gamma", "a": 1.0, "b": 0.5, "gamma": 2.0},
]

QUANTITIES = ["E", "F", "S", "Cv", "P", "ESS"]
EPS = np.finfo(float).eps

plt.rcParams["figure.max_open_warning"] = 0


@njit
def lcg_uniforms(n, seed):
    result = np.empty(n, dtype=np.float64)
    state = np.uint64(seed)
    modulus = np.uint64(2 ** 32)

    for i in range(n):
        state = (np.uint64(1664525) * state + np.uint64(1013904223)) % modulus
        result[i] = float(state) / float(2 ** 32)

    return result, int(state)


def uniform_matrix(n, seed):
    values, next_seed = lcg_uniforms(n * 6, seed)
    return values.reshape(n, 6), next_seed


def box_muller_normals(n, seed):
    n_pairs = (n + 1) // 2
    u, next_seed = lcg_uniforms(2 * n_pairs, seed)

    u1 = np.maximum(u[0::2], 1e-15)
    u2 = u[1::2]

    radius = np.sqrt(-2.0 * np.log(u1))
    angle = 2.0 * np.pi * u2

    result = np.empty(2 * n_pairs)
    result[0::2] = radius * np.cos(angle)
    result[1::2] = radius * np.sin(angle)

    return result[:n], next_seed


@njit
def rejection_normal_samples(n, seed):
    result = np.empty(n, dtype=np.float64)
    state = np.uint64(seed)
    modulus = np.uint64(2 ** 32)
    count = 0

    while count < n:
        state = (np.uint64(1664525) * state + np.uint64(1013904223)) % modulus
        x = -5.0 + 10.0 * (float(state) / float(2 ** 32))

        state = (np.uint64(1664525) * state + np.uint64(1013904223)) % modulus
        y = 0.5 * (float(state) / float(2 ** 32))

        if y < math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi):
            result[count] = x
            count += 1

    return result, int(state)


def demonstrate_random_generators():
    z_box, _ = box_muller_normals(50_000, BASE_SEED)
    z_rejection, _ = rejection_normal_samples(50_000, BASE_SEED + 1)

    print("\nNORMAL DISTRIBUTION GENERATOR VALIDATION")

    for name, data in [("Box-Muller method", z_box), ("Rejection sampling method", z_rejection)]:
        print(f"\n{name}:")
        print(f"Number of samples: {len(data)}")
        print(f"Mean: {np.mean(data):.6f}")
        print(f"Standard deviation: {np.std(data, ddof=1):.6f}")
        print(f"Minimum: {np.min(data):.6f}")
        print(f"Maximum: {np.max(data):.6f}")

    x = np.linspace(-4, 4, 400)
    gaussian_pdf = np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, data, title in [(axes[0], z_box, "Box-Muller method"),
                            (axes[1], z_rejection, "Rejection sampling method")]:
        ax.hist(data, bins=70, density=True, alpha=0.65, label="Sample distribution")
        ax.plot(x, gaussian_pdf, "k--", label="N(0, 1)")
        ax.set_title(title)
        ax.set_xlabel("z")
        ax.set_ylabel("Probability density")
        ax.legend()
        ax.grid(alpha=0.2)

    fig.tight_layout()
    return fig


def potential_energy(points, a, b, gamma):
    x, y = points[:, :3], points[:, 3:]
    distance2 = np.sum((x - y) ** 2, axis=1)
    interaction = 0.5 * b * distance2 ** (1.0 + gamma / 2.0)
    return a * (np.sum(points ** 2, axis=1) + interaction)


def crude_points(unit_points, temperature, box_size, parameters):
    return unit_points * box_size, 6.0 * np.log(box_size)


def truncated_gaussian_points(unit_points, temperature, box_size, a):
    sigma = np.sqrt(temperature / (2.0 * a))
    normalization = ndtr(box_size / sigma) - 0.5

    probabilities = np.clip(0.5 + unit_points * normalization, EPS, 1.0 - EPS)
    points = sigma * ndtri(probabilities)

    log_q = (-0.5 * np.sum((points / sigma) ** 2, axis=1)
             - 6.0 * np.log(sigma * np.sqrt(2.0 * np.pi) * normalization))

    return points, log_q


def importance_points(unit_points, temperature, box_size, parameters):
    points, log_q = truncated_gaussian_points(unit_points, temperature, box_size, parameters["a"])
    return points, -log_q


SAMPLERS = {"CMC": crude_points, "IS": importance_points}


def log_weights(method, unit_points, temperature, box_size, parameters):
    points, log_factor = SAMPLERS[method](unit_points, temperature, box_size, parameters)
    energy = potential_energy(points, **parameters)
    return energy, -energy / temperature + log_factor


def log_partition(method, unit_points, temperature, box_size, parameters):
    _, log_w = log_weights(method, unit_points, temperature, box_size, parameters)
    return logsumexp(log_w) - np.log(len(log_w))


def calculate_observables(method, unit_points, temperature, box_size, parameters):
    energy, log_w = log_weights(method, unit_points, temperature, box_size, parameters)

    log_sum = logsumexp(log_w)
    log_z = log_sum - np.log(len(log_w))

    weights = np.exp(log_w - log_sum)
    mean_e = weights @ energy
    variance = max(weights @ (energy - mean_e) ** 2, 0.0)

    derivative_l = (
        log_partition(method, unit_points, temperature, box_size + PRESSURE_STEP, parameters)
        - log_partition(method, unit_points, temperature, box_size - PRESSURE_STEP, parameters)
    ) / (2.0 * PRESSURE_STEP)

    return {
        "E": mean_e,
        "F": -temperature * log_z,
        "S": log_z + mean_e / temperature,
        "Cv": variance / temperature ** 2,
        "P": temperature * derivative_l / (3.0 * box_size ** 2),
        "ESS": 1.0 / (weights @ weights),
    }


def run_experiment(parameters, parameter_name, seed):
    rows = []

    print(f"\nPARAMETER SET: {parameter_name}")
    print(f"a = {parameters['a']}, b = {parameters['b']}, gamma = {parameters['gamma']}")

    for temperature in TEMPERATURES:
        print(f"\nTEMPERATURE T = {temperature:.3f}")
        first_row = len(rows)

        for run_index in range(N_RUNS):
            unit_cmc, seed = uniform_matrix(N_SAMPLES, seed)
            unit_is, seed = uniform_matrix(N_SAMPLES, seed)

            for method, unit_points in [("CMC", unit_cmc), ("IS", unit_is)]:
                values = calculate_observables(method, unit_points, temperature, L, parameters)

                rows.append({
                    "Parameters": parameter_name,
                    "T": temperature,
                    "Method": method,
                    "Run": run_index + 1,
                    **values,
                })

                print(f"\n{method} | Run {run_index + 1}/{N_RUNS}")
                print(f"E = {values['E']:.6f} | F = {values['F']:.6f} | S = {values['S']:.6f}")
                print(f"Cv = {values['Cv']:.6f} | P = {values['P']:.6f} | ESS = {values['ESS']:.1f}",
                      flush=True)

        current = pd.DataFrame(rows[first_row:])
        print(f"\nSUMMARY FOR T = {temperature:.3f}")

        for method in ["CMC", "IS"]:
            group = current[current["Method"] == method]
            print(f"\nMethod {method}:")

            for quantity in QUANTITIES:
                mean = group[quantity].mean()
                std = group[quantity].std(ddof=1)
                print(f"{quantity}: {mean:.6f} ± {std:.6f} (SEM = {std / np.sqrt(len(group)):.6f})")

    return pd.DataFrame(rows), seed


def summarize(raw):
    summary = raw.groupby(["Parameters", "T", "Method"], sort=False)[QUANTITIES].agg(["mean", "std"])
    summary.columns = [f"{q}_{s}" for q, s in summary.columns]

    for quantity in QUANTITIES:
        summary[f"{quantity}_sem"] = summary[f"{quantity}_std"] / np.sqrt(N_RUNS)

    return summary.reset_index()


def print_final_table(summary):
    for name, group in summary.groupby("Parameters", sort=False):
        table = pd.DataFrame({"T": group["T"].map("{:.2f}".format), "Method": group["Method"]})

        for quantity in ["E", "F", "S", "Cv", "P"]:
            table[f"{quantity} ± σ"] = [
                f"{m:.5g} ± {s:.2g}"
                for m, s in zip(group[f"{quantity}_mean"], group[f"{quantity}_std"])
            ]

        table["ESS"] = group["ESS_mean"].map("{:.0f}".format)

        print(f"\nFINAL SUMMARY TABLE: {name}")
        print(table.sort_values(["T", "Method"], ascending=[True, False]).to_string(index=False))


def plot_results(summary):
    plot_info = [
        ("E", "Mean Potential Energy"),
        ("F", "Helmholtz Free Energy"),
        ("S", "Entropy"),
        ("Cv", "Heat Capacity"),
        ("P", "Pressure"),
    ]

    names = summary["Parameters"].unique()
    n_cols = 2
    n_rows = -(-len(names) // n_cols)

    for quantity, title in plot_info:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 4.2 * n_rows), sharex=True, squeeze=False)

        for ax, parameter_name in zip(axes.flat, names):
            data_parameter = summary[summary["Parameters"] == parameter_name]

            for method in ["CMC", "IS"]:
                data = data_parameter[data_parameter["Method"] == method].sort_values("T")
                ax.errorbar(data["T"], data[f"{quantity}_mean"], yerr=data[f"{quantity}_std"],
                            marker="o", capsize=4, label=method)

            ax.set_title(parameter_name)
            ax.set_ylabel(quantity)
            ax.grid(alpha=0.3)
            ax.legend()

        for ax in axes.flat[len(names):]:
            ax.set_visible(False)

        for ax in axes[-1]:
            ax.set_xlabel("Temperature T")

        fig.suptitle(f"{title} (error bars = σ over {N_RUNS} runs)")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"{quantity}.png", dpi=150)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("THERMODYNAMICS OF TWO INTERACTING PARTICLES")
    print("Comparison of Crude Monte Carlo and Importance Sampling")
    print(f"\nNumber of samples: {N_SAMPLES}")
    print(f"Number of independent runs: {N_RUNS}")
    print(f"Number of temperatures: {len(TEMPERATURES)}")
    print(f"Box size L: {L}")
    print(f"Temperatures: {TEMPERATURES}")

    demonstrate_random_generators()

    selected_sets = PARAMETER_SETS if RUN_PARAMETER_STUDY else [{"name": "Baseline parameters", **BASE_PARAMETERS}]

    seed = BASE_SEED
    all_raw = []

    for item in selected_sets:
        parameters = {key: item[key] for key in ("a", "b", "gamma")}
        raw, seed = run_experiment(parameters, item["name"], seed)
        all_raw.append(raw)

    raw = pd.concat(all_raw, ignore_index=True)
    summary = summarize(raw)

    raw.to_csv(OUTPUT_DIR / "raw.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    print_final_table(summary)
    plot_results(summary)

    print("\nCalculations completed.")

    plt.show()


if __name__ == "__main__":
    main()