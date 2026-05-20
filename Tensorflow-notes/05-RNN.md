# RNN (Recurrent Neural Network)

RNN is a neural network designed for **sequential data**.

Used for:

* Text prediction
* Language translation
* Chatbots
* Speech recognition
* Time-series prediction
* Stock prediction

Unlike ANN/CNN, RNN remembers previous information.

---

# WHY RNN EXISTS

ANN assumes inputs are independent.

But many real problems are sequential.

Example sentence:

```text id="x4e8r1"
"I am learning deep _____"
```

To predict the next word:

* the model must remember previous words.

That is why RNN was created.

---

# STEP 1 — High-Level Understanding

RNN processes data step-by-step while keeping memory.

Basic idea:

```text id="jlwm9k"
Previous information
↓
Memory state
↓
Current prediction
```

---

# SEQUENTIAL DATA EXAMPLES

| Data Type   | Example             |
| ----------- | ------------------- |
| Text        | sentence prediction |
| Time-series | stock prices        |
| Speech      | audio processing    |
| Sensor data | IoT signals         |

---

# STEP 2 — RNN PIPELINE

Basic RNN flow:

```text id="gkm4t8"
Input sequence
↓
RNN/LSTM layer
↓
Dense layer
↓
Prediction
```

Example text sequence:

```text id="z6c9t2"
"I love deep"
↓
Predict:
"learning"
```

---

# HOW RNN WORKS

RNN processes sequence one timestep at a time.

Example:

```text id="c4o2rf"
Word 1 → memory updated
Word 2 → memory updated
Word 3 → prediction
```

The hidden state carries information forward.

---

# SIMPLE RNN EQUATION

RNN uses previous hidden state:

Where:

* (x_t) = current input
* (h_{t-1}) = previous memory
* (h_t) = current hidden state

You do NOT need to memorize deeply initially.
Just understand:

* RNN remembers previous information.

---

# STEP 3 — TensorFlow/Keras Implementation

Simple RNN example:

```python id="7tq9v1"
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(64),
    tf.keras.layers.Dense(1)
])
```

---

# IMPORTANT RNN PARAMETERS

## 1. Units

```python id="0jlwmx"
SimpleRNN(64)
```

Means:

* 64 memory units

More units:

* more learning capacity
* slower training

---

## 2. return_sequences

Example:

```python id="1jz8qf"
LSTM(64, return_sequences=True)
```

Controls whether layer outputs:

* single output
  OR
* full sequence

Important for stacked RNNs.

---

## 3. Input Shape

Very important.

Example:

```python id="v0y5s1"
input_shape=(timesteps, features)
```

Example:

```python id="r2o7zt"
input_shape=(10,1)
```

Means:

* 10 timesteps
* 1 feature each step

---

# BIG PROBLEM OF BASIC RNN

RNN struggles with long sequences.

Problem:

* vanishing gradients

It forgets old information.

---

# STEP 4 — LSTM (Long Short-Term Memory)

LSTM is improved RNN.

MOST important sequence model for beginners.

It solves memory problems.

---

# WHY LSTM IS BETTER

LSTM can remember important information longer.

Used for:

* long text
* speech
* forecasting

---

# LSTM PIPELINE

```text id="jlwm72"
Sequence
↓
LSTM memory cells
↓
Dense
↓
Prediction
```

---

# SIMPLE LSTM EXAMPLE

```python id="m9x2lf"
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(
        64,
        input_shape=(10,1)
    ),

    tf.keras.layers.Dense(1)
])
```

---

# IMPORTANT LSTM PARAMETERS

## 1. Units

```python id="n7v5pw"
LSTM(64)
```

64 memory cells.

---

## 2. return_sequences

Needed when stacking layers.

Example:

```python id="jlwm55"
LSTM(64, return_sequences=True)
```

---

## 3. Dropout

Helps prevent overfitting.

```python id="v6s8zt"
LSTM(64, dropout=0.2)
```

---

# COMPLETE LSTM EXAMPLE

## Time-Series Prediction

Example:
Predict next value from sequence.

---

# TensorFlow Example

```python id="6w2l9q"
import tensorflow as tf
import numpy as np

# Example sequence data
X = np.array([
    [[1],[2],[3]],
    [[2],[3],[4]],
    [[3],[4],[5]]
])

y = np.array([4,5,6])

# Build model
model = tf.keras.Sequential([

    tf.keras.layers.LSTM(
        64,
        input_shape=(3,1)
    ),

    tf.keras.layers.Dense(1)
])

# Compile
model.compile(
    optimizer='adam',
    loss='mse'
)

# Train
model.fit(X, y, epochs=100)

# Predict
print(model.predict(np.array([[[4],[5],[6]]])))
```

---

# UNDERSTANDING SHAPES

This is VERY important.

Input shape:

```python id="jlwm20"
(3,1)
```

Means:

```text id="jlwm21"
3 timesteps
1 feature
```

Example sequence:

```text id="jlwm22"
[1,2,3]
```

timesteps:

* 1
* 2
* 3

---

# RNN vs CNN vs ANN

| Model    | Best For                    |
| -------- | --------------------------- |
| ANN      | Tabular data                |
| CNN      | Images                      |
| RNN/LSTM | Sequential/time-series/text |

---

# COMMON RNN/LSTM APPLICATIONS

## NLP

* next word prediction
* translation
* chatbots

## Finance

* stock forecasting

## Speech

* voice recognition

## IoT

* sensor sequence analysis

---

# COMMON RNN/LSTM PROBLEMS

## 1. Vanishing Gradients

Model forgets older information.

Solved partly using:

* LSTM
* GRU

---

## 2. Overfitting

Solutions:

* dropout
* more data
* regularization

---

# IMPORTANT PARAMETERS TO EXPERIMENT WITH

| Parameter  | Effect            |
| ---------- | ----------------- |
| units      | memory capacity   |
| timesteps  | sequence length   |
| epochs     | learning duration |
| dropout    | regularization    |
| batch size | speed/memory      |

---

# Beginner RNN Practice Progression

## LEVEL 1

Simple sequence prediction

Example:

```text id="jlwm66"
[1,2,3] → 4
```

---

## LEVEL 2

Time-series forecasting

Example:

* temperature prediction

---

## LEVEL 3

Text prediction

Example:

* next word prediction

---

## LEVEL 4

Sentiment analysis

Example:

* positive/negative review classification

---

# Most Important RNN/LSTM Workflow

Memorize:

```text id="jlwm73"
Prepare sequences
↓
Reshape data
↓
LSTM/RNN layer
↓
Dense layer
↓
Train
↓
Predict
```

This is the standard beginner sequence-modeling pipeline in TensorFlow.
