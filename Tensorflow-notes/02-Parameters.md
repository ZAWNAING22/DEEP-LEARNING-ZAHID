## Critical Parameters to Learn (And *How* to Learn Them)

This is a **great question** because most students memorize parameters without understanding what they *do*. Here's the systematic approach.

---

## Part 1: The MUST-KNOW Parameters (By Category)

### Category A: Model Architecture Parameters

| Parameter | Where Used | What It Controls | Default | Why Important |
|-----------|-----------|------------------|---------|----------------|
| `units` | `Dense(units=64)` | Number of neurons | None (required) | Model capacity |
| `activation` | `Dense(64, activation='relu')` | Non-linearity | None | Learning dynamics |
| `input_shape` | First layer only | Input dimensions | None | Data compatibility |
| `kernel_initializer` | Any layer | Starting weights | `'glorot_uniform'` | Convergence speed |

**How to learn these:**
```python
# Experiment: Change units and observe
for units in [16, 64, 256]:
    model = tf.keras.Sequential([tf.keras.layers.Dense(units, activation='relu')])
    # Train and plot loss
    # Question: At what point does more units NOT help?
```

---

### Category B: Training Parameters

| Parameter | Where Used | What It Controls | Default | Why Important |
|-----------|-----------|------------------|---------|----------------|
| `optimizer` | `model.compile()` | Weight update rule | `'rmsprop'` (Keras) | Learning speed/stability |
| `learning_rate` | Inside optimizer | Step size per update | Depends (0.001 for Adam) | **Most important parameter** |
| `loss` | `model.compile()` | Error measurement | None | Guides all learning |
| `metrics` | `model.compile()` | Human-readable tracking | None | Progress monitoring |

**How to learn these:**
```python
# The LR experiment (most critical skill)
learning_rates = [0.0001, 0.001, 0.01, 0.1]
histories = {}

for lr in learning_rates:
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mse')
    histories[lr] = model.fit(X_train, y_train, epochs=20, verbose=0)

# Plot loss curves for each LR
# Observe: Too low → slow convergence. Too high → loss explodes.
```

---

### Category C: Training Loop Parameters

| Parameter | Where Used | What It Controls | Default | Why Important |
|-----------|-----------|------------------|---------|----------------|
| `epochs` | `model.fit()` | Full dataset passes | 1 | Under/overfitting |
| `batch_size` | `model.fit()` | Samples per update | 32 | Speed vs stability tradeoff |
| `validation_split` | `model.fit()` | Holdout for monitoring | 0 | Overfitting detection |
| `shuffle` | `model.fit()` | Order randomization | True | Prevents order learning |
| `verbose` | `model.fit()` | Progress printing | 1 | Debugging |

**How to learn these:**
```python
# Batch size experiment
batch_sizes = [1, 16, 64, 256, 1024]

for batch_size in batch_sizes:
    print(f"Training with batch size: {batch_size}")
    %time model.fit(X_train, y_train, batch_size=batch_size, epochs=5, verbose=0)
    # Question: Which batch size is fastest? Most stable?
```

---

### Category D: Regularization Parameters

| Parameter | Where Used | What It Controls | Default | Why Important |
|-----------|-----------|------------------|---------|----------------|
| `dropout_rate` | `Dropout(rate=0.5)` | Neurons to zero | 0 | Overfitting prevention |
| `l1_reg` / `l2_reg` | `kernel_regularizer` | Weight penalty | None | Simplicity constraint |
| `patience` | `EarlyStopping(patience=5)` | Wait epochs before stopping | 1 | Training efficiency |

**How to learn these:**
```python
# Dropout comparison
with_dropout = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Add this
    tf.keras.layers.Dense(1)
])

without_dropout = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Train both on small dataset (to encourage overfitting)
# Observe: With dropout trains slower but validates better
```

---

### Category E: CNN-Specific Parameters

| Parameter | Where Used | What It Controls | Default | Why Important |
|-----------|-----------|------------------|---------|----------------|
| `filters` | `Conv2D(filters=32)` | Number of feature maps | None | Pattern detection capacity |
| `kernel_size` | `Conv2D(kernel_size=3)` | Filter window size | 3 | Receptive field |
| `strides` | `Conv2D(strides=1)` | Movement step | 1 | Downsampling |
| `padding` | `Conv2D(padding='same')` | Edge handling | `'valid'` | Output size |
| `pool_size` | `MaxPooling2D(pool_size=2)` | Window for pooling | 2 | Spatial reduction |

**How to learn these:**
```python
# Filter visualization experiment
layer = tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu')
# Train on MNIST, extract weights, visualize first layer filters
# Question: Why do early layers look like edge detectors?
```

---

## Part 2: HOW to Learn Parameters (The Method)

### Strategy 1: The "Change One Thing" Rule

**Never change multiple parameters at once.**

```python
# BAD (don't do this)
model.compile(optimizer='adam', learning_rate=0.01)  # Changed both

# GOOD (do this)
model.compile(optimizer='adam', learning_rate=0.001)  # Baseline
model.compile(optimizer='adam', learning_rate=0.01)   # Change LR only
model.compile(optimizer='sgd', learning_rate=0.001)   # Change optimizer only
```

**Why this matters:** You need to know *which change* caused the difference.

---

### Strategy 2: Extreme Testing

Test each parameter at its extremes to see what breaks.

| Parameter | Low Extreme | High Extreme | What Breaks |
|-----------|-------------|--------------|--------------|
| `learning_rate` | 1e-7 | 1.0 | Slow convergence / Loss NaN |
| `batch_size` | 1 | 10000 | Noisy gradients / Out of memory |
| `units` | 1 | 4096 | Underfitting / Overfitting |
| `dropout` | 0.0 | 0.95 | No effect / Underfitting |
| `epochs` | 1 | 1000 | Underfitting / Overfitting |

**Learning exercise:** Push each parameter until the model fails. Then dial back.

---

### Strategy 3: The Baseline + Grid Search

**Step 1:** Create a working baseline
```python
baseline = {
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'batch_size': 32,
    'units': 64,
    'dropout': 0.0,
    'epochs': 20
}
# Get baseline loss/accuracy
```

**Step 2:** Test one parameter at a time
```python
experiments = [
    {'learning_rate': 0.0001},
    {'learning_rate': 0.01},
    {'batch_size': 64},
    {'batch_size': 128},
    {'units': 128},
    {'units': 256},
    {'dropout': 0.3},
    {'dropout': 0.5},
]
```

**Step 3:** Record results in a table

| Parameter | Value | Val Loss | Val Acc | Better than baseline? |
|-----------|-------|----------|---------|----------------------|
| Baseline | - | 0.35 | 0.87 | - |
| LR | 0.0001 | 0.42 | 0.82 | No |
| LR | 0.01 | 0.38 | 0.85 | No |
| Batch size | 64 | 0.33 | 0.88 | **Yes** |
| Units | 128 | 0.34 | 0.88 | Yes |

---

### Strategy 4: Parameter Interaction Experiments

Some parameters interact. Learn these pairs:

| Parameter Pair | Interaction |
|----------------|-------------|
| `batch_size` + `learning_rate` | Larger batches often need larger LR |
| `dropout` + `epochs` | Dropout needs more epochs to converge |
| `units` + `regularization` | More units need stronger regularization |

**Experiment:**
```python
# Test batch_size + learning_rate together
combinations = [
    (32, 0.001),   # standard
    (64, 0.002),   # double batch, double LR
    (128, 0.004),  # 4x batch, 4x LR
]
```

---

## Part 3: Learning Workflow (Week by Week)

### Week 1-2: Architecture Parameters
```python
# Focus on: units, activation, input_shape
# Exercises:
1. Change units [1, 10, 100, 1000] on same problem
2. Compare relu vs tanh vs sigmoid
3. Add/remove layers
```

### Week 3-4: Optimization Parameters
```python
# Focus on: optimizer, learning_rate, batch_size
# Exercises:
1. Compare Adam vs SGD vs RMSprop
2. Find the "goldilocks" LR for your dataset
3. Plot loss vs batch size
```

### Week 5-6: Regularization Parameters
```python
# Focus on: dropout, L2, early_stopping
# Exercises:
1. Overfit a small dataset, then fix with dropout
2. Compare L1 vs L2 regularization
3. Set patience=2,5,10 and observe stopping point
```

### Week 7-8: CNN Parameters (if relevant)
```python
# Focus on: filters, kernel_size, strides, padding
# Exercises:
1. Visualize increasing filters (32→64→128)
2. Compare kernel_size 3 vs 5 vs 7
3. Understand 'same' vs 'valid' padding
```

---

## Part 4: Quick Reference Card (Print This)

```text
┌─────────────────────────────────────────────────────────────┐
│              MOST IMPORTANT PARAMETERS (90% of time)        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  learning_rate    → Try: 0.001 first, then adjust          │
│  units            → Start: input_dim * 2, then experiment  │
│  batch_size       → Start: 32, then power of 2 [16,64,128]  │
│  dropout          → Start: 0.0, add 0.3-0.5 if overfitting  │
│  epochs           → Start: 50 + EarlyStopping              │
│  optimizer        → Start: 'adam', switch if stuck         │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│              DEFAULT VALUES THAT JUST WORK                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  activation       → 'relu' for hidden, 'softmax' for output │
│  loss             → 'mse' for regression,                  │
│                    'categorical_crossentropy' for class     │
│  kernel_initializer → 'he_normal' with relu               │
│  padding          → 'same' for conv layers                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Part 5: The Ultimate Learning Exercise

Build this **parameter explorer** template:

```python
def experiment(parameter_name, values, build_model, X, y):
    """
    Test how changing one parameter affects performance
    """
    results = {}
    
    for value in values:
        print(f"\nTesting {parameter_name} = {value}")
        
        # Build model with current parameter
        model = build_model(**{parameter_name: value})
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Train briefly (for speed)
        history = model.fit(X, y, epochs=10, validation_split=0.2, verbose=0)
        
        # Store final validation loss
        results[value] = history.history['val_loss'][-1]
        
    # Plot results
    plt.plot(values, list(results.values()), 'o-')
    plt.xlabel(parameter_name)
    plt.ylabel('Validation Loss')
    plt.title(f'Effect of {parameter_name} on Performance')
    plt.show()
    
    return results

# Use it:
experiment('learning_rate', [0.0001, 0.001, 0.01, 0.1], 
           lambda lr: tf.keras.Sequential([
               tf.keras.layers.Dense(64, activation='relu'),
               tf.keras.layers.Dense(1)
           ]), X_train, y_train)
```

**This one function will teach you more about parameters than reading any documentation.**

---

## Summary: What To Learn In Order

| Priority | Parameter Category | Why | Time to learn |
|----------|-------------------|-----|---------------|
| 1 | `learning_rate` | Single most impactful | 2 days |
| 2 | `units`, `layers` | Model capacity | 3 days |
| 3 | `batch_size`, `epochs` | Training dynamics | 2 days |
| 4 | `optimizer` | Convergence speed | 1 day |
| 5 | `dropout`, `regularization` | Fix overfitting | 2 days |
| 6 | CNN params (if needed) | Architecture specific | 3 days |

**Golden rule:** Theory tells you parameters exist. Experimenting tells you what they *do*. Do both.
```text
┌─────────────────────────────────────────────────────────┐
│            REAL-WORLD DEEP LEARNING                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │ Parameters  │    │    Data     │    │ Deployment  │ │
│  │ (25%)       │    │ Pipeline    │    │ (20%)       │ │
│  │ • LR        │    │ (30%)       │    │ • Serving   │ │
│  │ • batch     │    │ • tf.data   │    │ • API       │ │
│  │ • units     │    │ • caching   │    │ • latency   │ │
│  └─────────────┘    │ • prefetch  │    └─────────────┘ │
│                     └─────────────┘                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │ Evaluation  │    │ Debugging   │    │    Git      │ │
│  │ (15%)       │    │ (15%)       │    │ (5%)        │ │
│  │ • metrics   │    │ • gradient  │    │ • version   │ │
│  │ • confusion │    │   checking  │    │ • CI/CD     │ │
│  │ • ROC       │    │ • failure   │    │             │ │
│  └─────────────┘    │   analysis  │    └─────────────┘ │
│                     └─────────────┘                    │
└─────────────────────────────────────────────────────────┘
```
