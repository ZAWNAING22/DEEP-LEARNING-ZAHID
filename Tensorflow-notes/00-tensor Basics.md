# STAGE 1 — TensorFlow Basics

Stage 1 is about understanding the **foundation** of TensorFlow.

Before neural networks, CNNs, or AI models, TensorFlow works with:

```text id="0cgdzv"
Tensors
```

Everything in TensorFlow becomes a tensor.

---

# WHAT IS A TENSOR?

A tensor is basically:

```text id="4ww8wx"
A multidimensional array
```

You already know simpler versions:

| Type         | Example          |
| ------------ | ---------------- |
| Scalar (0D)  | `5`              |
| Vector (1D)  | `[1,2,3]`        |
| Matrix (2D)  | `[[1,2],[3,4]]`  |
| Tensor (3D+) | image/video data |

---

# WHY TENSORS ARE USED

Because Deep Learning needs:

* fast matrix calculations
* GPU acceleration
* huge multidimensional data

Examples:

* Images
* Satellite rasters
* Audio
* Video
* Time series

For your GIS/remote sensing work:
a satellite image itself becomes a tensor.

Example:

```text id="w61v31"
Height × Width × Bands
```

Like:

```text id="9m8jif"
256 × 256 × 13
```

for Sentinel-2 multispectral imagery.

---

# EXAMPLE 1 — SCALAR TENSOR

```python id="mz3v2q"
import tensorflow as tf

x = tf.constant(5)

print(x)
```

Output:

```text id="6q5v0y"
tf.Tensor(5, shape=(), dtype=int32)
```

---

# UNDERSTAND THE OUTPUT

## Value

```text id="38gdmf"
5
```

## Shape

```text id="hq0wd5"
()
```

Means:
0-dimensional tensor (scalar).

---

# EXAMPLE 2 — VECTOR TENSOR

```python id="dr5v8f"
x = tf.constant([1,2,3,4])

print(x)
```

Output:

```text id="8l9u9x"
shape=(4,)
```

Meaning:

* 1-dimensional tensor
* contains 4 values

---

# EXAMPLE 3 — MATRIX TENSOR

```python id="ehc6k2"
x = tf.constant([
    [1,2],
    [3,4]
])

print(x)
```

Output:

```text id="hzjj74"
shape=(2,2)
```

Meaning:

* 2 rows
* 2 columns

This is VERY important because neural networks use matrix multiplication heavily.

---

# REAL USAGE IN DEEP LEARNING

Suppose you have:

```text id="n6bwt8"
1000 images
```

Each image:

```text id="5cf2ot"
28 × 28 pixels
```

TensorFlow stores them as:

```text id="t89gn4"
(1000, 28, 28)
```

Meaning:

```text id="e6iw59"
(Number of images, height, width)
```

If RGB images:

```text id="1d9odg"
(1000, 28, 28, 3)
```

where:

* 3 = RGB channels

---

# WHY SHAPE IS IMPORTANT

Deep learning models require exact dimensions.

Example:

```text id="9m2a3s"
Dense layer expects vectors
CNN expects images
RNN expects sequences
```

So understanding tensor shape is critical.

---

# TENSOR OPERATIONS

TensorFlow is basically optimized math.

---

# ADDITION

```python id="j6g6c0"
a = tf.constant([1,2,3])
b = tf.constant([4,5,6])

print(a + b)
```

Output:

```text id="x75ygn"
[5 7 9]
```

---

# MULTIPLICATION

```python id="qjlwmw"
print(a * b)
```

Output:

```text id="6muj0f"
[4 10 18]
```

Element-wise multiplication.

---

# MATRIX MULTIPLICATION

VERY important in neural networks.

```python id="e4pb29"
A = tf.constant([
    [1,2],
    [3,4]
])

B = tf.constant([
    [5,6],
    [7,8]
])

print(tf.matmul(A,B))
```

This powers:

* neural layers
* weight calculations
* backpropagation

---

# WHY MATRIX MULTIPLICATION MATTERS

Neural network formula:

genui{"math_block_widget_always_prefetch_v2":{"content":"y = Wx + b"}}

Where:

* `x` = input tensor
* `W` = weight matrix
* `b` = bias
* `y` = output

TensorFlow performs these calculations efficiently on GPU.

---

# TENSOR DATA TYPES

TensorFlow tensors have types:

| Type    | Meaning         |
| ------- | --------------- |
| int32   | integers        |
| float32 | decimal numbers |
| bool    | true/false      |

Example:

```python id="7axdhq"
x = tf.constant([1.0,2.0])

print(x.dtype)
```

Output:

```text id="10k2kp"
float32
```

---

# GPU USAGE

TensorFlow is designed for GPU acceleration.

---

# CHECK GPU

```python id="cr4f54"
print(tf.config.list_physical_devices('GPU'))
```

If GPU exists:

```text id="e9dtfs"
[PhysicalDevice(name='/physical_device:GPU:0')]
```

---

# WHY GPU IS IMPORTANT

CPU:

* sequential processing

GPU:

* thousands of parallel operations

Deep learning requires millions of matrix operations.

---

# SIMPLE VISUALIZATION

```text id="nbo03v"
Normal Python:
slow math

TensorFlow:
optimized tensor math on CPU/GPU
```

---

# REAL-WORLD USAGE

---

# IMAGE PROCESSING

Image tensor:

```text id="jz1ylh"
(height, width, channels)
```

Example:

```text id="f9mjlwm"
(224,224,3)
```

Used in:

* CNNs
* object detection
* medical imaging

---

# GIS / REMOTE SENSING

Satellite raster:

```text id="3hq0tv"
(height, width, spectral bands)
```

Example:

```text id="1qwwcv"
(256,256,13)
```

Used in:

* land cover classification
* flood detection
* wildfire mapping

---

# NLP (TEXT AI)

Sentence tensor:

```text id="6rqjv4"
(sequence_length, embedding_dimension)
```

Used in:

* chatbots
* translation
* transformers

---

# STAGE 1 GOAL

By the end of Stage 1 you should understand:

✅ What tensors are
✅ Tensor dimensions/shapes
✅ Tensor operations
✅ Matrix multiplication
✅ Why GPU matters
✅ Why tensors are the foundation of DL

---

# BEST PRACTICE EXERCISES

Practice these:

---

## Exercise 1

Create:

* scalar
* vector
* matrix tensor

Print:

* shape
* dtype

---

## Exercise 2

Perform:

* addition
* subtraction
* multiplication

---

## Exercise 3

Do matrix multiplication with:

```text id="e6c9l6"
2×2 matrices
```

---

## Exercise 4

Load a grayscale image and check shape.

Expected:

```text id="q7pqns"
(height,width)
```

---

# MOST IMPORTANT THING TO REMEMBER

TensorFlow is basically:

```text id="rrq9lj"
Highly optimized tensor mathematics
```

Deep learning is built on top of that foundation.
