"""Reproducible Monte Carlo convergence experiment for Option Pricing Lab.

Place beside option_pricing_app.py and svi.py, then run:
    python research_mc.py

The one-step exact GBM transition has the same terminal-price distribution as
the app's 64-step paths. A European call depends only on that terminal price.
"""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np
import plotly.graph_objects as go

from option_pricing_app import black_scholes_call, simulate_call


SPOT = 100.0
STRIKE = 100.0
MATURITY = 1.0
RATE = 0.05
DIVIDEND_YIELD = 0.0
VOLATILITY = 0.20
PATH_COUNTS = (1_000, 10_000, 100_000)


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20,
                        help="Different random seeds per path count (default: 20)")
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("research_results"))
    args = parser.parse_args()
    if args.repeats < 2 or args.seed_start < 0:
        parser.error("--repeats must be at least 2 and --seed-start nonnegative")

    args.output.mkdir(parents=True, exist_ok=True)
    bs_price = float(black_scholes_call(
        SPOT, STRIKE, MATURITY, RATE, DIVIDEND_YIELD, VOLATILITY))
    runs = []
    summary = []

    for n_paths in PATH_COUNTS:
        group = []
        for seed in range(args.seed_start, args.seed_start + args.repeats):
            start = perf_counter()
            _, mc_price, mc_se = simulate_call(
                SPOT, STRIKE, MATURITY, RATE, DIVIDEND_YIELD, VOLATILITY,
                n_paths=n_paths, n_steps=1, seed=seed)
            elapsed_ms = (perf_counter() - start) * 1000
            row = {
                "paths": n_paths,
                "seed": seed,
                "bs_price": bs_price,
                "mc_price": mc_price,
                "absolute_error": abs(mc_price - bs_price),
                "mc_standard_error": mc_se,
                "elapsed_ms": elapsed_ms,
            }
            runs.append(row)
            group.append(row)

        summary.append({
            "paths": n_paths,
            "repeats": args.repeats,
            "bs_price": bs_price,
            "mean_absolute_error": float(np.mean([r["absolute_error"] for r in group])),
            "mean_mc_standard_error": float(np.mean([r["mc_standard_error"] for r in group])),
            "mean_elapsed_ms": float(np.mean([r["elapsed_ms"] for r in group])),
        })

    write_csv(args.output / "runs.csv", runs, tuple(runs[0]))
    write_csv(args.output / "summary.csv", summary, tuple(summary[0]))

    fig = go.Figure()
    fig.add_scatter(x=[r["paths"] for r in summary],
                    y=[r["mean_absolute_error"] for r in summary],
                    mode="lines+markers", name="Mean absolute pricing error")
    fig.add_scatter(x=[r["paths"] for r in summary],
                    y=[r["mean_mc_standard_error"] for r in summary],
                    mode="lines+markers", name="Mean reported MC standard error")
    fig.update_layout(title="European call: Monte Carlo convergence",
                      xaxis_title="Simulated paths (log scale)",
                      yaxis_title="Price error ($)", xaxis_type="log",
                      template="plotly_white")
    fig.write_html(args.output / "convergence.html", include_plotlyjs=True)

    print("Fixed assumptions: S=K=$100, T=1 year, r=5%, q=0%, volatility=20%")
    print(f"Black-Scholes benchmark: ${bs_price:.4f}")
    print("Paths    Mean abs. error    Mean MC SE    Mean runtime")
    for row in summary:
        print(f"{row['paths']:>6,}    ${row['mean_absolute_error']:.4f}"
              f"             ${row['mean_mc_standard_error']:.4f}"
              f"        {row['mean_elapsed_ms']:.2f} ms")
    print(f"Saved runs.csv, summary.csv and convergence.html in {args.output}/")
    print("This is a numerical experiment under one assumed scenario, not market validation.")


if __name__ == "__main__":
    main()
