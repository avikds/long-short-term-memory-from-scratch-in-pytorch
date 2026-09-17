"""
Long Short-Term Memory from Scratch in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - rnn_forward
import torch
import torch.nn as nn

def init_rnn(d, h, seed, scale=0.5):
    # Set the random seed before drawing the parameters.
    torch.manual_seed(seed)

    # Initialize RNN parameters as nn.Parameters.
    Wx = nn.Parameter(torch.randn(h, d) * scale)
    Wh = nn.Parameter(torch.randn(h, h) * scale)
    b = nn.Parameter(torch.zeros(h))

    return {
        "Wx": Wx,
        "Wh": Wh,
        "b": b
    }

def rnn_step(x, h_prev, p):
    # h_t = tanh(x_t @ Wx.T + h_{t-1} @ Wh.T + b)
    return torch.tanh(
        x @ p["Wx"].T +
        h_prev @ p["Wh"].T +
        p["b"]
    )

def rnn_forward(X, p):
    # X shape: (B, T, d)
    B, T, _ = X.shape

    # Hidden-state size h.
    h = p["b"].shape[0]

    # Initial hidden state: zeros.
    h_prev = X.new_zeros(B, h)

    hidden_states = []

    # Process the sequence one timestep at a time.
    for t in range(T):
        h_prev = rnn_step(X[:, t, :], h_prev, p)
        hidden_states.append(h_prev)

    # Shape: (B, T, h)
    return torch.stack(hidden_states, dim=1)

# Step 2 - gradient_vs_lag
import math
import numpy as np

def gradient_vs_lag(forward, X, lags):
    # Clone X and make it a leaf tensor that tracks gradients.
    X_grad = X.clone().detach().requires_grad_(True)

    # Forward pass: expected shape (B, T, h).
    states = forward(X_grad)

    # Sum of the last timestep's hidden states.
    loss = states[:, -1, :].sum()

    # Backpropagate through the entire sequence.
    loss.backward()

    # Gradient with respect to X at timestep T - 1 - lag.
    T = X_grad.shape[1]
    norms = []

    for lag in lags:
        t = T - 1 - lag
        grad = X_grad.grad[:, t, :]

        # L2 norm over the entire batch and feature dimension.
        norm = torch.linalg.vector_norm(grad).item()
        norms.append(float(norm))

    return norms


def decay_rate(lags, norms):
    # Fit a least-squares line:
    # log(norm) = slope * lag + intercept
    lags = np.asarray(lags, dtype=float)
    norms = np.asarray(norms, dtype=float)

    log_norms = np.log(norms)

    slope, _ = np.polyfit(lags, log_norms, 1)

    return float(slope)


def half_life(rate):
    # For a negative decay rate, solve:
    # exp(rate * half_life) = 0.5
    if rate < 0:
        return float(math.log(0.5) / rate)

    return float("inf")

# Step 3 - carousel
def carousel_forward(U):
    # C_t = C_{t-1} + U_t, with C_0 = 0.
    # Cumulative sum along the time dimension.
    return torch.cumsum(U, dim=1)

def carousel_gain(T, h, lag):
    # Random candidate inputs with gradient tracking enabled.
    U = torch.randn(1, T, h, requires_grad=True)

    # Forward through the constant error carousel.
    C = carousel_forward(U)

    # Sum of the final state's elements.
    loss = C[:, -1, :].sum()

    # Backpropagate to obtain d(loss) / dU.
    loss.backward()

    # Return the gradient at timestep T - 1 - lag
    # for the first (and only) batch element.
    return U.grad[0, T - 1 - lag, :]

def is_constant_error(T, h):
    # Verify that the gradient gain is exactly a vector of ones
    # for every possible lag.
    expected = torch.ones(h)

    for lag in range(T):
        gain = carousel_gain(T, h, lag)

        if not torch.equal(gain, expected):
            return False

    return True

# Step 4 - lstm1997_cell
class LSTM1997Cell(nn.Module):
    def __init__(self, d, h):
        super().__init__()

        # Three independent affine maps:
        # input gate, output gate, and candidate update.
        self.gate_i = nn.Linear(d + h, h)
        self.gate_o = nn.Linear(d + h, h)
        self.cand = nn.Linear(d + h, h)

        self.h = h

    def forward(self, x, state):
        h_prev, c_prev = state

        # Concatenate current input and previous hidden state.
        z = torch.cat([x, h_prev], dim=1)

        # Input gate, output gate, and candidate.
        i = torch.sigmoid(self.gate_i(z))
        o = torch.sigmoid(self.gate_o(z))
        g = torch.tanh(self.cand(z))

        # Original 1997 memory update:
        # no forget gate; previous cell state is carried unchanged.
        c = c_prev + i * g

        # Hidden/output state.
        h = o * torch.tanh(c)

        return h, c

def run_cell(cell, X, h):
    # X shape: (B, T, d)
    B, T, _ = X.shape

    # Initial hidden and cell states.
    h_prev = X.new_zeros(B, h)
    c_prev = X.new_zeros(B, h)

    hidden_states = []
    cell_states = []

    # Run the cell sequentially over time.
    for t in range(T):
        h_prev, c_prev = cell(X[:, t, :], (h_prev, c_prev))

        hidden_states.append(h_prev)
        cell_states.append(c_prev)

    # Stack into (B, T, h).
    H = torch.stack(hidden_states, dim=1)
    C = torch.stack(cell_states, dim=1)

    return H, C

def gate_values(cell, x, state):
    h_prev, _ = state

    # Same concatenated input used by the cell.
    z = torch.cat([x, h_prev], dim=1)

    # Compute the three gate/candidate values.
    i = torch.sigmoid(cell.gate_i(z))
    o = torch.sigmoid(cell.gate_o(z))
    g = torch.tanh(cell.cand(z))

    return {
        "i": i,
        "o": o,
        "g": g
    }

# Step 5 - cell_gradient_horizon
def state_gradient_vs_lag(cell, X, lags):
    # Clone X and enable gradient tracking.
    X_grad = X.clone().detach().requires_grad_(True)

    # Run the 1997 LSTM cell over the sequence.
    _, C = run_cell(cell, X_grad, cell.h)

    # Loss is the sum of the final cell state.
    loss = C[:, -1, :].sum()

    # Backpropagate through the sequence.
    loss.backward()

    # Measure the gradient norm at each requested lag.
    T = X_grad.shape[1]
    norms = []

    for lag in lags:
        t = T - 1 - lag

        # Gradient with respect to the input at this timestep.
        grad = X_grad.grad[:, t, :]

        # L2 norm over the batch and input dimensions.
        norm = torch.linalg.vector_norm(grad).item()
        norms.append(float(norm))

    return norms

def compare_decay(rnn_params, cell, X, lags):
    # Gradient decay for the plain RNN hidden states.
    rnn_norms = gradient_vs_lag(
        lambda X_: rnn_forward(X_, rnn_params),
        X,
        lags
    )
    rnn_rate = decay_rate(lags, rnn_norms)

    # Gradient decay for the LSTM cell states.
    cell_norms = state_gradient_vs_lag(
        cell,
        X,
        lags
    )
    cell_rate = decay_rate(lags, cell_norms)

    return rnn_rate, cell_rate

def carousel_wins(rates):
    rnn_rate, cell_rate = rates

    # The cell wins when its decay rate is closer to zero
    # than the plain RNN's decay rate.
    return abs(cell_rate) < abs(rnn_rate)

# Step 6 - lstm_cell
import math

class LSTMCell(nn.Module):
    def __init__(self, d, h):
        super().__init__()

        self.h = h

        # PyTorch LSTM parameter shapes and initialization.
        self.W_ih = nn.Parameter(torch.empty(4 * h, d))
        self.W_hh = nn.Parameter(torch.empty(4 * h, h))
        self.b_ih = nn.Parameter(torch.empty(4 * h))
        self.b_hh = nn.Parameter(torch.empty(4 * h))

        # Same uniform initialization range used by nn.LSTM:
        # [-1/sqrt(h), 1/sqrt(h)].
        bound = 1.0 / math.sqrt(h)

        nn.init.uniform_(self.W_ih, -bound, bound)
        nn.init.uniform_(self.W_hh, -bound, bound)
        nn.init.uniform_(self.b_ih, -bound, bound)
        nn.init.uniform_(self.b_hh, -bound, bound)

    def forward(self, x, state):
        h_prev, c_prev = state

        # Combined affine transformation.
        z = (
            x @ self.W_ih.T
            + self.b_ih
            + h_prev @ self.W_hh.T
            + self.b_hh
        )

        # PyTorch gate order:
        # input, forget, cell candidate, output.
        i, f, g, o = z.chunk(4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        # Modern LSTM state update.
        c = f * c_prev + i * g
        h = o * torch.tanh(c)

        return h, c

def set_forget_bias(cell, value):
    # Both PyTorch-style bias vectors contribute to the forget gate,
    # so each receives value / 2.
    with torch.no_grad():
        cell.b_ih[cell.h:2 * cell.h].fill_(value / 2)
        cell.b_hh[cell.h:2 * cell.h].fill_(value / 2)

    return cell

def param_count(cell):
    return sum(p.numel() for p in cell.parameters())

# Step 7 - lstm_layer_vs_torch
class LSTM(nn.Module):
    def __init__(self, d, h):
        super().__init__()

        self.h = h
        self.cell = LSTMCell(d, h)

    def forward(self, X, state=None):
        # X shape: (B, T, d)
        B, T, _ = X.shape

        # Start from zero states when no initial state is provided.
        if state is None:
            h_prev = X.new_zeros(B, self.h)
            c_prev = X.new_zeros(B, self.h)
        else:
            h_prev, c_prev = state

        hidden_states = []

        # Run the LSTM cell over each timestep.
        for t in range(T):
            h_prev, c_prev = self.cell(
                X[:, t, :],
                (h_prev, c_prev)
            )
            hidden_states.append(h_prev)

        # H shape: (B, T, h)
        H = torch.stack(hidden_states, dim=1)

        return H, (h_prev, c_prev)

def copy_from_torch(ours, ref):
    # Copy PyTorch's parameters into our LSTM cell.
    with torch.no_grad():
        ours.cell.W_ih.copy_(ref.weight_ih_l0)
        ours.cell.W_hh.copy_(ref.weight_hh_l0)
        ours.cell.b_ih.copy_(ref.bias_ih_l0)
        ours.cell.b_hh.copy_(ref.bias_hh_l0)

    return ours

def max_abs_diff(ours, ref, X):
    # Our implementation returns (H, (h, c)).
    H_ours, _ = ours(X)

    # nn.LSTM returns (H, (h_n, c_n)).
    H_ref, _ = ref(X)

    # Largest absolute element-wise difference.
    return float(torch.max(torch.abs(H_ours - H_ref)).item())

# Step 8 - forget_bias_horizon
def horizon(lstm, X, biases):
    norms = []

    T = X.shape[1]

    for bias in biases:
        # Set the same forget-bias contribution in both bias vectors.
        set_forget_bias(lstm.cell, bias)

        # Return the final cell state as a one-step sequence so that
        # gradient_vs_lag can measure the gradient with respect to
        # the first input at lag T - 1.
        forward = lambda Z: lstm(Z)[1][1].unsqueeze(1)

        norm = gradient_vs_lag(
            forward,
            X,
            [T - 1]
        )[0]

        norms.append(float(norm))

    return norms

def bias_for_horizon(lstm, X, biases, threshold):
    norms = horizon(lstm, X, biases)

    # Find the smallest bias whose measured gradient norm
    # reaches or exceeds the requested threshold.
    qualifying = [
        bias
        for bias, norm in zip(biases, norms)
        if norm >= threshold
    ]

    if not qualifying:
        return None

    return min(qualifying)

# Step 9 - adding_problem
def adding_problem(n, T, gen):
    # Channel 0: random values in [0, 1).
    values = torch.rand(n, T, generator=gen)

    # Channel 1: binary markers.
    markers = torch.zeros(n, T)

    # Choose two distinct positions for each sequence.
    for i in range(n):
        positions = torch.randperm(T, generator=gen)[:2]
        markers[i, positions] = 1.0

    # Construct the input tensor: (n, T, 2).
    X = torch.stack([values, markers], dim=2)

    # Target is the sum of the two marked values.
    y = (values * markers).sum(dim=1, keepdim=True)

    return X, y

class SequenceRegressor(nn.Module):
    def __init__(self, core, h):
        super().__init__()

        self.core = core
        self.head = nn.Linear(h, 1)

    def forward(self, X):
        # Recurrent core returns the complete sequence of outputs
        # followed by its final state.
        H, _ = self.core(X)

        # Use only the output from the final timestep.
        return self.head(H[:, -1, :])

def baseline_mse(y):
    # The MSE of predicting the mean target is the target variance.
    return float(torch.var(y).item())

# Step 10 - train_and_compare
def train_regressor(model, X, y, steps, lr=5e-3, batch=32, seed=0, clip=1.0):
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Use one seeded generator for the sequence of mini-batch draws.
    batch_gen = torch.Generator().manual_seed(seed)

    n = X.shape[0]
    losses = []

    for _ in range(steps):
        # Draw a mini-batch of indices.
        idx = torch.randint(
            0,
            n,
            (batch,),
            generator=batch_gen
        )

        X_batch = X[idx]
        y_batch = y[idx]

        # Forward pass.
        pred = model(X_batch)

        # Mean squared error.
        loss = torch.mean((pred - y_batch) ** 2)

        # Backpropagation.
        optimizer.zero_grad()
        loss.backward()

        # Clip the global gradient norm.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # Parameter update.
        optimizer.step()

        losses.append(float(loss.item()))

    return losses

def evaluate_mse(model, X, y):
    model.eval()

    with torch.no_grad():
        pred = model(X)
        mse = torch.mean((pred - y) ** 2).item()

    return float(mse)

def compare_on_adding(T, steps, seed=0):
    # Generate train and test sets from the same generator.
    data_gen = torch.Generator().manual_seed(seed)

    X_train, y_train = adding_problem(2000, T, data_gen)
    X_test, y_test = adding_problem(500, T, data_gen)

    # Construct LSTM model after setting the requested seed.
    torch.manual_seed(seed)
    lstm_model = SequenceRegressor(
        nn.LSTM(2, 32, batch_first=True),
        32
    )

    # Construct RNN model after resetting the same seed.
    torch.manual_seed(seed)
    rnn_model = SequenceRegressor(
        nn.RNN(2, 32, batch_first=True),
        32
    )

    # Train both models with identical training settings and batch seed.
    train_regressor(
        lstm_model,
        X_train,
        y_train,
        steps,
        seed=seed
    )

    train_regressor(
        rnn_model,
        X_train,
        y_train,
        steps,
        seed=seed
    )

    # Evaluate on the held-out test set.
    baseline = baseline_mse(y_test)
    lstm_mse = evaluate_mse(lstm_model, X_test, y_test)
    rnn_mse = evaluate_mse(rnn_model, X_test, y_test)

    return {
        "baseline": round(baseline, 4),
        "lstm": round(lstm_mse, 4),
        "rnn": round(rnn_mse, 4)
    }

# Step 11 - temporal_order
def temporal_order(n, T, gen):
    # Draw distractor symbols once for all sequences.
    S = torch.randint(
        0,
        4,
        (n, T),
        generator=gen,
        dtype=torch.long
    )

    for i in range(n):
        # Choose one position from the first half and one from the second half.
        first_pos = torch.randint(
            0,
            T // 2,
            (1,),
            generator=gen
        ).item()

        second_pos = torch.randint(
            T // 2,
            T,
            (1,),
            generator=gen
        ).item()

        # Choose X=0 or Y=1 for each of the two marked positions.
        first = torch.randint(
            0,
            2,
            (1,),
            generator=gen
        ).item()

        second = torch.randint(
            0,
            2,
            (1,),
            generator=gen
        ).item()

        # Encode X as 4 and Y as 5.
        S[i, first_pos] = 4 + first
        S[i, second_pos] = 4 + second

    # Labels:
    # XX -> 0, XY -> 1, YX -> 2, YY -> 3.
    labels = torch.empty(n, dtype=torch.long)

    for i in range(n):
        # Recover the two inserted symbols from their positions.
        # The positions themselves are regenerated here from the already
        # chosen values, so we instead construct the label during generation.
        # This block is replaced below by the correct direct construction.
        pass

    # Re-create S and labels together to preserve the exact generator order.
    S = torch.randint(
        0,
        4,
        (n, T),
        generator=gen,
        dtype=torch.long
    )

    labels = torch.empty(n, dtype=torch.long)

    for i in range(n):
        first_pos = torch.randint(
            0,
            T // 2,
            (1,),
            generator=gen
        ).item()

        second_pos = torch.randint(
            T // 2,
            T,
            (1,),
            generator=gen
        ).item()

        first = torch.randint(
            0,
            2,
            (1,),
            generator=gen
        ).item()

        second = torch.randint(
            0,
            2,
            (1,),
            generator=gen
        ).item()

        S[i, first_pos] = 4 + first
        S[i, second_pos] = 4 + second

        labels[i] = 2 * first + second

    return S, labels

class SequenceClassifier(nn.Module):
    def __init__(self, core, h, n_classes=4):
        super().__init__()

        self.embedding = nn.Embedding(6, core.input_size)
        self.core = core
        self.head = nn.Linear(h, n_classes)

    def forward(self, S):
        # Convert integer symbols to dense vectors.
        X = self.embedding(S)

        # Run the recurrent core over the embedded sequence.
        H, _ = self.core(X)

        # Classify using the final timestep's output.
        return self.head(H[:, -1, :])

def train_classifier(model, S, labels, steps, lr=3e-3, batch=32, seed=0):
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    batch_gen = torch.Generator().manual_seed(seed)

    n = S.shape[0]
    losses = []

    for _ in range(steps):
        # Seeded mini-batch sampling.
        idx = torch.randint(
            0,
            n,
            (batch,),
            generator=batch_gen
        )

        S_batch = S[idx]
        labels_batch = labels[idx]

        # Forward pass and cross-entropy loss.
        logits = model(S_batch)
        loss = torch.nn.functional.cross_entropy(logits, labels_batch)

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping with global norm 1.0.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        losses.append(float(loss.item()))

    return losses

def accuracy(model, S, labels):
    model.eval()

    with torch.no_grad():
        logits = model(S)
        predictions = logits.argmax(dim=1)
        correct = (predictions == labels).float().mean().item()

    return float(correct)

# Step 12 - results_table
import re

def paper_table(T_add, steps_add, T_order, steps_order, seed=0):
    # Adding-problem results.
    adding_results = compare_on_adding(
        T_add,
        steps_add,
        seed=seed
    )

    adding_line = (
        f"adding problem T={T_add}: "
        f"baseline {adding_results['baseline']:.4f} "
        f"lstm {adding_results['lstm']:.4f} "
        f"rnn {adding_results['rnn']:.4f}"
    )

    # Generate temporal-order train/test data from one seeded generator.
    gen = torch.Generator().manual_seed(seed)

    S_train, labels_train = temporal_order(
        2000,
        T_order,
        gen
    )

    S_test, labels_test = temporal_order(
        400,
        T_order,
        gen
    )

    # Construct LSTM classifier with the requested seed.
    torch.manual_seed(seed)
    lstm_model = SequenceClassifier(
        nn.LSTM(8, 32, batch_first=True),
        32
    )

    # Construct RNN classifier with the same initialization seed.
    torch.manual_seed(seed)
    rnn_model = SequenceClassifier(
        nn.RNN(8, 32, batch_first=True),
        32
    )

    # Train both models identically.
    train_classifier(
        lstm_model,
        S_train,
        labels_train,
        steps_order,
        seed=seed
    )

    train_classifier(
        rnn_model,
        S_train,
        labels_train,
        steps_order,
        seed=seed
    )

    # Evaluate test accuracy.
    lstm_acc = accuracy(
        lstm_model,
        S_test,
        labels_test
    )

    rnn_acc = accuracy(
        rnn_model,
        S_test,
        labels_test
    )

    temporal_line = (
        f"temporal order T={T_order}: "
        f"chance 0.25 "
        f"lstm {lstm_acc:.3f} "
        f"rnn {rnn_acc:.3f}"
    )

    return [adding_line, temporal_line]

def lstm_wins(lines):
    # Parse the adding-problem row.
    adding_match = re.search(
        r"adding problem T=\d+:\s*"
        r"baseline ([0-9]*\.?[0-9]+)\s+"
        r"lstm ([0-9]*\.?[0-9]+)\s+"
        r"rnn ([0-9]*\.?[0-9]+)",
        lines[0]
    )

    # Parse the temporal-order row.
    order_match = re.search(
        r"temporal order T=\d+:\s*"
        r"chance ([0-9]*\.?[0-9]+)\s+"
        r"lstm ([0-9]*\.?[0-9]+)\s+"
        r"rnn ([0-9]*\.?[0-9]+)",
        lines[1]
    )

    if adding_match is None or order_match is None:
        return False

    adding_lstm = float(adding_match.group(2))
    adding_rnn = float(adding_match.group(3))

    order_lstm = float(order_match.group(2))
    order_rnn = float(order_match.group(3))

    # Lower MSE is better for the adding problem.
    adding_wins = adding_lstm < adding_rnn

    # Higher accuracy is better for the temporal-order task.
    order_wins = order_lstm > order_rnn

    return adding_wins and order_wins

