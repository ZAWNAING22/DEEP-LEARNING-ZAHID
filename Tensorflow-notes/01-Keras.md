# STAGE 2 — KERAS (The Real TensorFlow You Use)

In modern TensorFlow, most students mainly use:

```text id="drswsh"
tf.keras
```

Keras is a **high-level API** that makes deep learning easier.

Instead of manually writing all neural network mathematics, you describe the model using layers.

---

# BIG IDEA OF STAGE 2

You learn the complete workflow:

```text id="wkr8uo"
Build Model
↓
Compile Model
↓
Train Model
↓
Evaluate
↓
Predict
```

This workflow is used almost everywhere.

---

# COMPLETE SIMPLE EXAMPLE

We will predict:

```text id="g0a2my"
y = 2x
```

using a neural network.

---

# FULL CODE

```python id="r1kxsp"
import tensorflow as tf
import numpy as np

# Training data
X = np.array([1,2,3,4,5], dtype=float)
y = np.array([2,4,6,8,10], dtype=float)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1])
])

# Compile model
model.compile(
    optimizer='sgd',
    loss='mean_squared_error'
)

# Train model
model.fit(X, y, epochs=500)

# Predict
print(model.predict([6.0]))
```

---

# STEP-BY-STEP EXPLANATION

---

# 1. IMPORT LIBRARIES

```python id="1eyxjr"
import tensorflow as tf
import numpy as np
```

## Why?

### TensorFlow

For neural networks.

### NumPy

For arrays/data handling.

---

# 2. CREATE DATASET

```python id="f0tnl9"
X = np.array([1,2,3,4,5], dtype=float)
y = np.array([2,4,6,8,10], dtype=float)
```

---

# WHAT IS X?

Input features.

```text id="6iywuv"
1 → 2
2 → 4
3 → 6
```

The model learns:
output = 2 × input

---

# PARAMETER: dtype=float

Means:

```text id="w25kvo"
Use decimal numbers
```

Deep learning usually uses floating-point values.

---

# 3. BUILD MODEL

```python id="2qqm9h"
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1])
])
```

This is the MOST important Stage 2 concept.

---

# WHAT IS Sequential()?

```text id="j5kcc4"
A linear stack of layers
```

Data flows layer by layer.

Visualization:

```text id="w3frjz"
Input → Layer1 → Layer2 → Output
```

---

# WHAT IS Dense()?

A fully connected neural layer.

Every neuron connects to every input.

---

# PARAMETER: Dense(1)

Means:

```text id="m6jlwm"
1 neuron in this layer
```

Since our output is one value:

```text id="6j6m3e"
y = 2x
```

we need 1 output neuron.

---

# PARAMETER: input_shape=[1]

Means:

```text id="rz5a6y"
Each input has 1 feature
```

Example:

```text id="e4e8a8"
[1]
[2]
[3]
```

Each sample contains one number.

---

# VISUALIZATION

```text id="d1i5yk"
Input(x)
   ↓
Neuron
   ↓
Output(y)
```

---

# HOW THE NEURON WORKS

Neural network computes:

genui{"math_block_widget_always_prefetch_v2":{"content":"y = wx + b"}}

Where:

* `x` = input
* `w` = learned weight
* `b` = learned bias

TensorFlow automatically learns:

* best weight
* best bias

---

# 4. COMPILE MODEL

```python id="81bglv"
model.compile(
    optimizer='sgd',
    loss='mean_squared_error'
)
```

This prepares training.

---

# PARAMETER: optimizer

Optimizer updates weights to reduce error.

---

# optimizer='sgd'

SGD = Stochastic Gradient Descent.

Idea:

```text id="gw4jzh"
Move weights slowly toward better values
```

Like walking downhill toward minimum error.

---

# COMMON OPTIMIZERS

| Optimizer | Usage        |
| --------- | ------------ |
| SGD       | Basic/simple |
| Adam      | Most popular |
| RMSprop   | RNN/LSTM     |
| Adagrad   | Sparse data  |

Most modern projects use:

```python id="mowgrj"
optimizer='adam'
```

---

# PARAMETER: loss

Measures prediction error.

---

# mean_squared_error

Formula:

MSE = \frac{1}{n}\sum (y-\hat{y})^2

Used for:

* regression problems

Example:

* house prices
* temperature prediction
* rainfall prediction

---

# COMMON LOSS FUNCTIONS

| Loss                     | Used For                   |
| ------------------------ | -------------------------- |
| mean_squared_error       | Regression                 |
| binary_crossentropy      | Binary classification      |
| categorical_crossentropy | Multi-class classification |

---

# 5. TRAIN MODEL

```python id="r0s59d"
model.fit(X, y, epochs=500)
```

This is where learning happens.

---

# WHAT DOES fit() DO?

Internally:

```text id="mz4v7h"
1. Predict output
2. Calculate loss
3. Backpropagation
4. Update weights
5. Repeat
```

---

# PARAMETER: epochs

Means:

```text id="u2f2b8"
How many times the model sees the entire dataset
```

Example:

```text id="5z0n4w"
epochs=500
```

means:
training repeats 500 times.

---

# WHY MULTIPLE EPOCHS?

At first:

* predictions are bad

Gradually:

* weights improve
* loss decreases

---

# VISUALIZATION

```text id="y09mlz"
Epoch 1   → bad prediction
Epoch 50  → better
Epoch 500 → good prediction
```

---

# OTHER IMPORTANT fit() PARAMETERS

---

# batch_size

Example:

```python id="g5q7c7"
model.fit(X, y, epochs=10, batch_size=2)
```

Means:
train using 2 samples at a time.

---

# WHY BATCHES?

Large datasets cannot fit into memory at once.

---

# validation_split

Example:

```python id="a6tft4"
model.fit(X, y, validation_split=0.2)
```

Means:
20% data used for validation.

Purpose:
check overfitting.

---

# verbose

```python id="4x7p8z"
verbose=1
```

Shows training progress.

```python id="gw6sk0"
verbose=0
```

Silent mode.

---

# 6. PREDICTION

```python id="2e8khk"
model.predict([6.0])
```

Model predicts output for unseen data.

Expected output:

```text id="z7mjlwm"
≈ 12
```

because:
6 × 2 = 12

---

# REAL-WORLD USAGE

---

# REGRESSION

Predict continuous values.

Examples:

* temperature
* stock price
* rainfall
* crop yield

Usually:

* Dense layers
* MSE loss

---

# CLASSIFICATION

Predict categories.

Examples:

* spam/not spam
* disease/no disease
* land cover types

Usually:

* softmax
* crossentropy loss

---

# IMAGE AI

Using CNN layers.

Example:

```python id="ib0p2u"
Conv2D()
MaxPooling2D()
```

Applications:

* medical imaging
* object detection
* satellite classification

---

# GIS + REMOTE SENSING USAGE

You may later use TensorFlow for:

---

# LAND COVER CLASSIFICATION

Input:
satellite image tensors

Output:

* forest
* water
* urban
* agriculture

---

# FLOOD DETECTION

CNN learns flood patterns from imagery.

---

# WILDFIRE DETECTION

Using thermal/spectral bands.

---

# MOST IMPORTANT PARAMETERS SUMMARY

| Parameter            | Meaning               |
| -------------------- | --------------------- |
| Dense(64)            | 64 neurons            |
| activation='relu'    | activation function   |
| optimizer='adam'     | weight update method  |
| loss='mse'           | error calculation     |
| epochs=10            | training repetitions  |
| batch_size=32        | samples per update    |
| validation_split=0.2 | validation percentage |

---

# ACTIVATION FUNCTIONS

Very important.

---

# RELU

Most common.

f(x)=\max(0,x)

Used in hidden layers.

---

# SIGMOID

Output between 0 and 1.

Used for binary classification.

---

# SOFTMAX

Used for multi-class classification.

Example:

* cat
* dog
* horse

---

# STUDENT PRACTICE PLAN

Practice in this order:

---

# Exercise 1

Build:

* 1 neuron model

Predict:
`y = 3x`

---

# Exercise 2

Change:

* epochs
* optimizer

Observe:

* prediction quality

---

# Exercise 3

Add hidden layer:

```python id="z3e5y7"
Dense(8, activation='relu')
```

---

# Exercise 4

Use real dataset:

* student scores
* house prices

---

# MOST IMPORTANT THING TO UNDERSTAND

Stage 2 teaches:

```text id="js2kfa"
How to build/train neural networks using Keras
```

This becomes the foundation for:

* CNNs
* RNNs
* Transformers
* GIS AI
* Computer vision
* NLP
* Remote sensing AI
