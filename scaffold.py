"""
Long Short-Term Memory from Scratch in PyTorch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Long Short-Term Memory from Scratch in PyTorch (Hochreiter and Schmidhuber, 1997).

Story: measure how a plain RNN's gradient dies with the lag; build the constant
error carousel and check its gradient is exactly one; wrap it in the paper's input
and output gates and measure the slower decay; add the forget gate, stack the gates,
verify the cell against nn.LSTM weight for weight, and measure what the forget bias
does to the horizon; then run the paper's benchmarks, the adding problem and the
temporal order task, LSTM against RNN under the same budget.
"""
import math
import torch
import torch.nn as nn


def main() -> None:
    torch.manual_seed(0)
    lags = [0, 10, 20, 40]

    # ---- 1. The problem ----
    p = init_rnn(2, 8, seed=0, scale=0.5)
    torch.manual_seed(1)
    X = torch.randn(4, 60, 2)
    norms = gradient_vs_lag(lambda Z: rnn_forward(Z, p), X, lags)
    rate = decay_rate(lags, norms)
    print("plain RNN, gradient of the last state w.r.t. the input at lag: " + "  ".join(f"{l}: {n:.2e}" for l, n in zip(lags, norms)))
    print(f"  per-step factor {math.exp(rate):.3f}, half-life {half_life(rate):.1f} steps: the paper's vanishing gradient")

    # ---- 2. The 1997 cell ----
    print(f"\nconstant error carousel: gradient at lag 99 = {carousel_gain(100, 3, 99).tolist()}, constant at every lag: {is_constant_error(100, 3)}")
    cell = LSTM1997Cell(2, 8)
    rnn_rate, cell_rate = compare_decay(p, cell, X, lags)
    print(f"1997 cell (input + output gate, no forget gate): decay rate {cell_rate:+.4f} per step vs RNN {rnn_rate:+.4f}; carousel wins: {carousel_wins((rnn_rate, cell_rate))}")

    # ---- 3. The modern LSTM ----
    ours, ref = LSTM(4, 16), nn.LSTM(4, 16, batch_first=True)
    copy_from_torch(ours, ref)
    Xs = torch.randn(2, 30, 4)
    print(f"\nfrom-scratch LSTM vs nn.LSTM with copied weights: max |difference| {max_abs_diff(ours, ref, Xs):.2e} over 30 steps; {param_count(ours.cell)} parameters")
    Xh = torch.randn(1, 100, 4) * 0.1
    biases = [0.0, 2.0, 4.0, 6.0]
    hz = horizon(ours, Xh, biases)
    print("gradient reaching the first of 100 inputs by forget bias: " + "  ".join(f"b={b:g}: {n:.2e}" for b, n in zip(biases, hz)) + f"  -> smallest bias with a usable gradient: {bias_for_horizon(ours, Xh, biases, 1e-3)}")

    # ---- 4. The paper's benchmarks ----
    print("\nthe paper's benchmarks, LSTM vs plain RNN, identical data, optimizer, clipping and budget:")
    for line in paper_table(50, 1000, 200, 500):
        print("  " + line)
    print("  the RNN cannot carry a value across the lag; the LSTM's carousel and gates can")


if __name__ == "__main__":
    main()

