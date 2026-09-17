# Long Short-Term Memory from Scratch in PyTorch

Hochreiter and Schmidhuber's 1997 paper, rebuilt in PyTorch. First reproduce the problem: measure with autograd how the gradient of a plain recurrent network dies with the time lag. Then build the paper's answer piece by piece: the constant error carousel whose gradient is exactly one at any lag, the original memory cell with an input gate and an output gate and no forget gate, and the modern cell with a forget gate, verified weight for weight against nn.LSTM. Measure how the forget-gate bias sets the gradient horizon, then run the paper's own benchmarks, the adding problem and the temporal order task, and watch the LSTM learn across lags where the plain RNN never leaves the baseline.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** rnn_forward
- [x] **2.** gradient_vs_lag
- [x] **3.** carousel
- [x] **4.** lstm1997_cell
- [x] **5.** cell_gradient_horizon
- [x] **6.** lstm_cell
- [x] **7.** lstm_layer_vs_torch
- [x] **8.** forget_bias_horizon
- [x] **9.** adding_problem
- [x] **10.** train_and_compare
- [x] **11.** temporal_order
- [x] **12.** results_table

## Results

```
plain RNN, gradient of the last state w.r.t. the input at lag: 0: 1.45e+00  10: 7.35e-02  20: 2.61e-03  40: 1.97e-06
  per-step factor 0.712, half-life 2.0 steps: the paper's vanishing gradient

constant error carousel: gradient at lag 99 = [1.0, 1.0, 1.0], constant at every lag: True
1997 cell (input + output gate, no forget gate): decay rate +0.0035 per step vs RNN -0.3399; carousel wins: True

from-scratch LSTM vs nn.LSTM with copied weights: max |difference| 7.45e-08 over 30 steps; 1408 parameters
gradient reaching the first of 100 inputs by forget bias: b=0: 9.90e-23  b=2: 6.37e-04  b=4: 1.55e+00  b=6: 4.50e+00  -> smallest bias with a usable gradient: 4.0

the paper's benchmarks, LSTM vs plain RNN, identical data, optimizer, clipping and budget:
  adding problem T=50: baseline 0.1694 lstm 0.0046 rnn 0.1694
  temporal order T=200: chance 0.25 lstm 0.880 rnn 0.485
  the RNN cannot carry a value across the lag; the LSTM's carousel and gates can
```
