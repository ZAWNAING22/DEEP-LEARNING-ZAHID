# ANN (Artificial Neural Network) — Full Beginner Workflow

ANN is the simplest deep learning model and usually the **first Stage 3 topic** students learn.

It is mainly used for:

* Numerical/tabular data
* Regression
* Basic classification

Examples:

* House price prediction
* Student marks prediction
* Diabetes prediction
* Salary prediction

---

# STEP 1 — High-Level Understanding

An ANN tries to learn patterns from data using layers of neurons.

Basic idea:

```text id="0xgbhu"
Input
↓
Hidden Layer(s)
↓
Output
```

Example:

Predict exam marks from:

* study hours
* attendance
* assignments

---

# STEP 2 — ANN Pipeline

Typical ANN flow:

```text id="b4v6wj"
Input Features
↓
Dense Layer
↓
Activation Function
↓
Dense Layer
↓
Prediction
```

---

# STEP 3 — TensorFlow/Keras Implementation

Simple ANN example:

```python
import tensorflow as tf
import numpy as np

# Input data
X = np.array([[1], [2], [3], [4]], dtype=float)

# Output data
y = np.array([[2], [4], [6], [8]], dtype=float)

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
model.fit(X, y, epochs=100)

# Predict
print(model.predict([[5]]))
```

This model learns:

genui{"math_block_widget_always_prefetch_v2":{"content":"y = 2x"}}

---

# STEP 4 — Understand Important ANN Parameters

## 1. Dense Layer

```python
Dense(64)
```

Means:

* 64 neurons in the layer

More neurons:

* more learning capacity
* more computation

---

## 2. Activation Function

Example:

```python
activation='relu'
```

Common activations:

| Activation | Usage                      |
| ---------- | -------------------------- |
| relu       | Most hidden layers         |
| sigmoid    | Binary classification      |
| softmax    | Multi-class classification |

Example:

```python
Dense(64, activation='relu')
```

---

## 3. Optimizer

Controls learning updates.

Example:

```python
optimizer='adam'
```

Common:

* SGD
* Adam
* RMSprop

Adam is most commonly used.

---

## 4. Loss Function

Measures prediction error.

Examples:

| Problem Type          | Loss                     |
| --------------------- | ------------------------ |
| Regression            | mse                      |
| Binary classification | binary_crossentropy      |
| Multi-class           | categorical_crossentropy |

---

## 5. Epochs

```python
epochs=10
```

Means:

* entire dataset passes through model 10 times

More epochs:

* better learning
* possible overfitting

---

## 6. Batch Size

Example:

```python
batch_size=32
```

Data processed in groups.

---

# STEP 5 — Practice by Modifying

This is the most important learning stage.

Try changing:

| Change               | Observe           |
| -------------------- | ----------------- |
| epochs 10 → 100      | better learning   |
| neurons 8 → 64       | model capacity    |
| relu → sigmoid       | output behavior   |
| optimizer SGD → Adam | convergence speed |

---

# REAL Beginner ANN Example

## Student Marks Prediction

Inputs:

* study hours
* attendance

Output:

* exam score

Example dataset:

| Hours | Attendance | Score |
| ----- | ---------- | ----- |
| 2     | 70         | 50    |
| 5     | 90         | 85    |

---

# ANN Architecture Example

```text id="om08g8"
2 Inputs
↓
Dense(16, relu)
↓
Dense(8, relu)
↓
Dense(1)
```

---

# Important ANN Concepts

## Forward Propagation

Data moves:

```text id="mjlwmz"
Input → Hidden Layers → Output
```

---

## Backpropagation

Model updates weights using error.

Very important exam topic.

---

## Overfitting

Model memorizes training data.

Symptoms:

* high training accuracy
* poor test accuracy

Solutions:

* more data
* dropout
* early stopping

---

# ANN vs CNN

| ANN               | CNN                      |
| ----------------- | ------------------------ |
| Tabular data      | Images                   |
| Dense connections | Convolution filters      |
| Simpler           | More powerful for vision |

---

# Best ANN Practice Datasets

## Regression

* House prices
* Salary prediction

## Classification

* Diabetes
* Titanic survival
* Heart disease

---

# Your Goal While Practising ANN

You should become comfortable with:

```text id="u0okv7"
1. Building model
2. Compiling
3. Training
4. Evaluating
5. Predicting
6. Changing parameters
```

---

# Most Important TensorFlow ANN Workflow

Memorize this structure:

```python
# 1. Import
import tensorflow as tf

# 2. Load data

# 3. Build model
model = tf.keras.Sequential()

# 4. Compile
model.compile()

# 5. Train
model.fit()

# 6. Evaluate
model.evaluate()

# 7. Predict
model.predict()
```

This exact workflow is used in almost every TensorFlow project.
