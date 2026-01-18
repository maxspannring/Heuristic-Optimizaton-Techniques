import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_objective_vs_iteration(df):
    iterations = df["iteration"]
    objective = df["objective"]

    # best-so-far curve
    best_so_far = np.minimum.accumulate(objective)

    plt.figure(figsize=(12, 6))
    plt.plot(iterations, objective, label="Objective value", alpha=0.4)
    plt.plot(iterations, best_so_far, label="Best-so-far", linewidth=2)

    plt.xlabel("Iteration")
    plt.ylabel("Objective value")
    plt.title("ALNS convergence behavior")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_objective_vs_time(df):
    time = df["time_sec"].cumsum()
    objective = df["objective"]
    best_so_far = np.minimum.accumulate(objective)

    plt.figure(figsize=(12, 6))
    plt.plot(time, best_so_far, linewidth=2)

    plt.xlabel("Time [s]")
    plt.ylabel("Best objective value")
    plt.title("ALNS convergence over time")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_acceptance_rate(df, window=50):
    """
    Plot rolling acceptance rate over iterations.
    """
    acceptance = df["accepted"].astype(int)
    rolling_rate = acceptance.rolling(window=window, min_periods=1).mean() * 100

    plt.figure(figsize=(12, 6))
    plt.plot(df["iteration"], rolling_rate, linewidth=2)

    plt.xlabel("Iteration")
    plt.ylabel("Acceptance rate [%]")
    plt.title(f"Acceptance rate (rolling window = {window})")
    plt.ylim(0, 100)
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_operator_usage_combined(df, window=50):
    DESTROY_LABELS = {
    0: "Random removal",
    1: "Worst removal",
    2: "Shaw removal",
    3: "Heavy request removal",
    }

    REPAIR_LABELS = {
        0: "Greedy repair",
        1: "Regret-3 repair",
    }

    iterations = df["iteration"]

    destroy_ops = sorted(df["destroy_op"].unique())
    repair_ops = sorted(df["repair_op"].unique())

    plt.figure(figsize=(12, 6))

    # Destroy operators (solid lines)
    for op in destroy_ops:
        usage = (df["destroy_op"] == op).astype(int)
        rolling_usage = usage.rolling(window=window, min_periods=1).mean() * 100

        label = DESTROY_LABELS.get(op, f"Destroy {op}")

        plt.plot(
            iterations,
            rolling_usage,
            linestyle="-",
            linewidth=2,
            label=label
        )

    # Repair operators (dashed lines)
    for op in repair_ops:
        usage = (df["repair_op"] == op).astype(int)
        rolling_usage = usage.rolling(window=window, min_periods=1).mean() * 100

        label = REPAIR_LABELS.get(op, f"Repair {op}")

        plt.plot(
            iterations,
            rolling_usage,
            linestyle="--",
            linewidth=2,
            label=label
        )

    plt.xlabel("Iteration")
    plt.ylabel("Usage frequency [%]")
    plt.title(f"Destroy & repair operator usage (rolling window = {window})")
    plt.ylim(0, 100)
    plt.legend(ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def scaling_plot():
    # Load data
    df = pd.read_csv("src/scaling_results.csv")

    # Aggregate per instance size
    agg = (
        df
        .groupby("n")["runtime_sec"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )

    # === Fit power-law model T(n) = a * n^alpha ===
    n_vals = agg["n"].values
    mean_vals = agg["mean"].values

    log_n = np.log(n_vals)
    log_T = np.log(mean_vals)

    alpha, log_a = np.polyfit(log_n, log_T, 1)
    a = np.exp(log_a)

    print(f"Fitted model: T(n) ≈ {a:.3e} · n^{alpha:.3f}")

    # Generate smooth fit line
    n_fit = np.linspace(n_vals.min(), n_vals.max(), 300)
    T_fit = a * n_fit ** alpha

    # === Plot ===
    plt.figure(figsize=(10, 6))

    # Error bars (mean ± min/max)
    yerr = [
        agg["mean"] - agg["min"],
        agg["max"] - agg["mean"]
    ]

    plt.errorbar(
        agg["n"],
        agg["mean"],
        yerr=yerr,
        fmt="o",
        capsize=4,
        label="Measured runtime (mean ± min/max)"
    )

    # Fitted scaling curve
    plt.plot(
        n_fit,
        T_fit,
        "--",
        linewidth=2,
        label=fr"Fit: $T(n) \propto n^{{{alpha:.2f}}}$"
    )

    # plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Instance size n")
    plt.ylabel("Runtime (seconds)")
    plt.title("ALNS Runtime Scaling with Instance Size")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.show()



def plot_algorithm_comparison(csv_path):
    """
    Creates two side-by-side plots:
    1) Objective value (mean ± min/max)
    2) Runtime in seconds (mean ± min/max)

    X-axis: Algorithm (BeamSearch, ACO, ALNS)
    """

    # -------------------------
    # Load and sanitize data
    # -------------------------
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Ensure consistent ordering
    algorithms = ["BeamSearch", "ACO", "ALNS"]

    # -------------------------
    # Aggregate statistics
    # -------------------------
    agg = (
        df
        .groupby("algorithm")
        .agg(
            objective_mean=("objective", "mean"),
            objective_min=("objective", "min"),
            objective_max=("objective", "max"),
            runtime_mean=("runtime_sec", "mean"),
            runtime_min=("runtime_sec", "min"),
            runtime_max=("runtime_sec", "max"),
        )
        .reindex(algorithms)
        .reset_index()
    )

    x = np.arange(len(algorithms))

    # -------------------------
    # Create figure
    # -------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ==========================================================
    # Plot 1: Objective score
    # ==========================================================
    obj_yerr = [
        agg["objective_mean"] - agg["objective_min"],
        agg["objective_max"] - agg["objective_mean"],
    ]

    axes[0].errorbar(
        x,
        agg["objective_mean"],
        yerr=obj_yerr,
        fmt="o",
        capsize=5
    )

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(algorithms)
    axes[0].set_ylabel("Objective value")
    axes[0].set_title("Objective score per algorithm")
    axes[0].grid(True)

    # ==========================================================
    # Plot 2: Runtime
    # ==========================================================
    time_yerr = [
        agg["runtime_mean"] - agg["runtime_min"],
        agg["runtime_max"] - agg["runtime_mean"],
    ]

    axes[1].errorbar(
        x,
        agg["runtime_mean"],
        yerr=time_yerr,
        fmt="o",
        capsize=5
    )

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(algorithms)
    axes[1].set_ylabel("Runtime (seconds)")
    axes[1].set_title("Runtime per algorithm")
    axes[1].grid(True)

    # -------------------------
    # Layout
    # -------------------------
    plt.tight_layout()
    plt.show()



def plot_algorithm_boxplots(csv_path):
    """
    Creates two side-by-side boxplots:
    1) Objective values per algorithm
    2) Runtime per algorithm

    X-axis: BeamSearch, ACO, ALNS
    """

    # -------------------------
    # Load data
    # -------------------------
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    algorithms = ["BeamSearch", "ACO", "ALNS"]

    # -------------------------
    # Prepare data for boxplots
    # -------------------------
    objective_data = [
        df[df["algorithm"] == algo]["objective"].values
        for algo in algorithms
    ]

    runtime_data = [
        df[df["algorithm"] == algo]["runtime_sec"].values
        for algo in algorithms
    ]

    # -------------------------
    # Create figure
    # -------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ==================================================
    # Boxplot 1: Objective values
    # ==================================================
    axes[0].boxplot(
        objective_data,
        labels=algorithms,
        showfliers=True
    )
    axes[0].set_title("Objective value per algorithm")
    axes[0].set_ylabel("Objective value")
    axes[0].grid(True)

    # ==================================================
    # Boxplot 2: Runtime
    # ==================================================
    axes[1].boxplot(
        runtime_data,
        labels=algorithms,
        showfliers=True
    )
    axes[1].set_title("Runtime per algorithm")
    axes[1].set_ylabel("Runtime (seconds)")
    axes[1].grid(True)

    # -------------------------
    # Layout
    # -------------------------
    plt.tight_layout()
    plt.show()




def scaling_plot2():
    # Load data
    df = pd.read_csv("src/scaling_results_aco_preliminary.csv")

    # Aggregate per instance size
    agg = (
        df
        .groupby("n")["runtime_sec"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )

    # === Fit power-law model T(n) = a * n^alpha ===
    n_vals = agg["n"].values
    mean_vals = agg["mean"].values

    log_n = np.log(n_vals)
    log_T = np.log(mean_vals)

    alpha, log_a = np.polyfit(log_n, log_T, 1)
    a = np.exp(log_a)

    print(f"Fitted model: T(n) ≈ {a:.3e} · n^{alpha:.3f}")

    # Generate smooth fit line
    n_fit = np.linspace(n_vals.min(), n_vals.max(), 300)
    T_fit = a * n_fit ** alpha

    # === Plot ===
    plt.figure(figsize=(10, 6))

    # Error bars (mean ± min/max)
    yerr = [
        agg["mean"] - agg["min"],
        agg["max"] - agg["mean"]
    ]

    plt.errorbar(
        agg["n"],
        agg["mean"],
        yerr=yerr,
        fmt="o",
        capsize=4,
        label="Measured runtime (mean ± min/max)"
    )

    # Fitted scaling curve
    plt.plot(
        n_fit,
        T_fit,
        "--",
        linewidth=2,
        label=fr"Fit: $T(n) \propto n^{{{alpha:.2f}}}$"
    )

    # plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Instance size n")
    plt.ylabel("Runtime (seconds)")
    plt.title("ALNS Runtime Scaling with Instance Size")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.show()
