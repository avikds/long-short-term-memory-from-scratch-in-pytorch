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

---

Built on Deep-ML.
