"""Build solver duels from benchmark results and plot their outcomes."""

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import logsumexp

from mdpforge.utils.analysis import _aggregate_rows, _ordered_solver_labels


def _pairwise_outcomes(results, experiment, tolerance):
    """Return per-model counts of first wins, second wins and draws for each pair."""
    if (
        isinstance(tolerance, (bool, np.bool_))
        or not isinstance(tolerance, (int, float, np.integer, np.floating))
        or not np.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ValueError("tolerance must be finite and nonnegative")

    models, solvers, aggregated = _aggregate_rows(
        results, experiment, require_validation=True
    )
    models.sort()  # Keep bootstrap draws reproducible when trial rows are reordered.
    solvers = list(dict.fromkeys([*(experiment or {}).get("solvers", {}), *solvers]))
    ordered, labels = _ordered_solver_labels(solvers, experiment)
    if len(ordered) < 2:
        raise ValueError("A pairwise tournament needs at least two solvers")

    times = np.array(
        [
            [
                aggregated.get((model, solver), {}).get("median_runtime", np.nan)
                for solver in ordered
            ]
            for model in models
        ]
    )
    failures = {
        (row["model"], row["solver"])
        for row in results
        if row["status"] in {"error", "timeout"}
        or row.get("validation_status") == "failed"
    }
    failed = np.array(
        [[(model, solver) in failures for solver in ordered] for model in models]
    )
    first, second = np.triu_indices(len(ordered), k=1)
    first_time, second_time = times[:, first], times[:, second]
    first_valid, second_valid = np.isfinite(first_time), np.isfinite(second_time)
    factor = 1.0 + tolerance
    first_wins = first_valid & (failed[:, second] | (second_time / factor > first_time))
    second_wins = second_valid & (
        failed[:, first] | (first_time / factor > second_time)
    )
    draws = first_valid & second_valid & ~first_wins & ~second_wins
    outcomes = np.stack((first_wins, second_wins, draws), axis=-1)
    return labels, first, second, outcomes


def plot_pairwise_wins(results, experiment=None, *, tolerance=0.05, ax=None, show=True):
    """Plot wins divided by all recorded duels, including draws."""
    labels, first, second, outcomes = _pairwise_outcomes(results, experiment, tolerance)
    counts = outcomes.sum(axis=0)
    n_solvers = len(labels)
    wins = np.zeros((n_solvers, n_solvers), dtype=int)
    draws = np.zeros_like(wins)
    wins[first, second], wins[second, first] = counts[:, 0], counts[:, 1]
    draws[first, second] = draws[second, first] = counts[:, 2]
    totals = wins + wins.T + draws
    rates = np.divide(wins, totals, out=np.full(wins.shape, np.nan), where=totals > 0)

    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(max(6, 0.68 * n_solvers + 2.8), max(4, 0.68 * n_solvers + 1.8))
        )
    else:
        fig = ax.figure
    cmap = plt.get_cmap("Blues").with_extremes(bad="#eef2f6")
    image = ax.imshow(rates, vmin=0, vmax=1, cmap=cmap)
    ax.set_xticks(range(n_solvers), labels, rotation=35, ha="right")
    ax.set_yticks(range(n_solvers), labels)
    ax.set_xlabel("Opponent")
    ax.set_ylabel("Solver")
    ax.set_title(
        f"Pairwise win rates · {100.0 * tolerance:g}% runtime tolerance", loc="left"
    )
    for row, column in np.ndindex(rates.shape):
        if row == column:
            text = "—"
        elif totals[row, column]:
            draw_count = draws[row, column]
            draw_label = "draw" if draw_count == 1 else "draws"
            text = (
                f"{rates[row, column]:.0%}\nn={totals[row, column]}\n"
                f"{draw_count} {draw_label}"
            )
        else:
            text = "—\nn=0"
        ax.text(
            column,
            row,
            text,
            ha="center",
            va="center",
            fontsize=8,
            color="white" if rates[row, column] >= 0.72 else "#102a43",
        )
    fig.colorbar(image, ax=ax, format=PercentFormatter(1), label="Row solver win rate")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def _fit_davidson(first, second, counts, initial):
    """Fit solver log-strengths and a common log draw propensity."""
    totals = counts.sum(axis=1)
    n_solvers = len(initial) - 1
    penalty = 0.25  # Keep estimates finite under complete separation.

    def objective(parameters):
        half_gap = (parameters[first] - parameters[second]) / 2
        logits = np.column_stack(
            (half_gap, -half_gap, np.full(len(first), parameters[-1]))
        )
        log_probabilities = logits - logsumexp(logits, axis=1, keepdims=True)
        loss = -np.sum(counts * log_probabilities) + penalty / 2 * np.dot(
            parameters, parameters
        )

        residuals = totals[:, None] * np.exp(log_probabilities) - counts
        gap_gradient = (residuals[:, 0] - residuals[:, 1]) / 2
        gradient = penalty * parameters
        gradient[:-1] += np.bincount(
            first, weights=gap_gradient, minlength=n_solvers
        ) - np.bincount(second, weights=gap_gradient, minlength=n_solvers)
        gradient[-1] += residuals[:, 2].sum()
        return loss, gradient

    fit = minimize(objective, initial, method="L-BFGS-B", jac=True)
    if not fit.success:
        raise RuntimeError(f"Davidson tournament fit failed: {fit.message}")
    return fit.x


def _bootstrap_ratings(first, second, outcomes, n_solvers, n_bootstrap, seed):
    """Fit the tournament and resample whole models for rating uncertainty."""
    counts = outcomes.sum(axis=0)
    fit = _fit_davidson(first, second, counts, np.zeros(n_solvers + 1))
    strengths = [fit[:-1]]
    n_models = len(outcomes)
    if n_models > 1:
        rng = np.random.default_rng(seed)
        for _ in range(n_bootstrap):
            indices = rng.integers(n_models, size=n_models)
            sample = _fit_davidson(first, second, outcomes[indices].sum(axis=0), fit)
            strengths.append(sample[:-1])

    strengths = np.asarray(strengths)
    centered = strengths - strengths.mean(axis=1, keepdims=True)
    return 1500 + (400 / np.log(10)) * centered


def plot_elo(
    results,
    experiment=None,
    *,
    tolerance=0.05,
    confidence=0.95,
    n_bootstrap=500,
    seed=0,
    ax=None,
    show=True,
):
    """Plot strengths from one Davidson fit and intervals from model resampling."""
    if not isinstance(confidence, (float, np.floating)) or not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one, exclusive")
    if type(n_bootstrap) is not int or n_bootstrap < 2:
        raise ValueError("n_bootstrap must be an integer of at least two")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    labels, first, second, outcomes = _pairwise_outcomes(results, experiment, tolerance)
    played_pairs = outcomes.any(axis=(0, 2))
    if not played_pairs.any():
        raise ValueError("No comparable duels for Elo ratings; use validated results")
    active, indices = np.unique(
        np.r_[first[played_pairs], second[played_pairs]], return_inverse=True
    )
    first, second = np.split(indices, 2)
    n_solvers = len(active)
    adjacency = csr_matrix(
        (np.ones(len(first)), (first, second)), shape=(n_solvers, n_solvers)
    )
    if connected_components(adjacency, directed=False, return_labels=False) != 1:
        raise ValueError("Elo ratings require a connected graph of comparable duels")

    outcomes = outcomes[outcomes.any(axis=(1, 2))][:, played_pairs]
    ratings = _bootstrap_ratings(first, second, outcomes, n_solvers, n_bootstrap, seed)
    order = np.argsort(-ratings[0], kind="stable")
    positions = np.arange(n_solvers)

    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(3.4, 0.36 * n_solvers + 2.2)))
    else:
        fig = ax.figure
    subtitle = (
        f"{len(outcomes)} models with duels · {100.0 * tolerance:g}% runtime tolerance"
    )
    if n_solvers < len(labels):
        subtitle += f" · {len(labels) - n_solvers} solvers without duels"

    ax.axvline(1500, linestyle="--", color="#94a3b8", linewidth=1)
    if len(ratings) > 1:
        tail = (1 - confidence) / 2
        lower, upper = np.quantile(ratings[1:, order], [tail, 1 - tail], axis=0)
        ax.hlines(
            positions,
            lower,
            upper,
            color="#94a3b8",
            linewidth=2,
            label=f"{100 * confidence:g}% bootstrap CI",
        )
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.22), frameon=False)
    else:
        subtitle += "\nBootstrap requires at least two models"

    ax.scatter(ratings[0, order], positions, color="#1d4ed8", zorder=3)
    ax.set_yticks(
        positions, [f"{labels[active[i]]}  ({ratings[0, i]:.0f})" for i in order]
    )
    ax.set_ylim(n_solvers - 0.5, -0.5)
    ax.set_xlabel("Elo-scale rating (mean = 1500)")
    ax.set_title(f"Solver ratings · Bradley–Terry–Davidson\n{subtitle}", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax
