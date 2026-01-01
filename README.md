# mlp-classification-from-scratch
# MLP Classification from Scratch (Java)

Implementation of a **Multilayer Perceptron (MLP)** classifier in **Java**, trained with **backpropagation** and **mini-batch gradient descent** (no ML frameworks).

The model is trained on a synthetic 2D dataset with **4 classes** and evaluated on a separate test set.

---

## Problem Setup (Dataset)

- Input space: points (x1, x2) in the square **[0, 2] × [0, 2]**
- Total samples: **8000**
  - Training: **4000**
  - Test: **4000**
- Labels: **4 categories**
- CSV format:
  - Header + columns: `x1,x2,label`

---

## Model

- Inputs: **D = 2**
- Output classes: **K = 4**
- Hidden layers: **3 hidden layers**
  - H1, H2, H3 are configurable in the code
- Hidden activations (configurable):
  - `tanh`, `ReLU`, (and logistic option)
- Output layer:
  - logistic units per class
- Loss:
  - sum of squared errors (SSE)

---

## Training

- Backpropagation + mini-batch gradient descent
- Random weight initialization in (-1, 1)
- Prints training/test related info and exports test predictions.

---

## How to Run

### 1) Put datasets in `data/`
Make sure you have:
- `data/train_T.csv`
- `data/test_T.csv`

### 2) Compile
`javac PT_MLP.java`

### 3) Run
java PT_MLP


The program produces:

test_results_T.csv (test predictions / results)

### 4) Experiments (What to Try)

Change hidden layer sizes: H1, H2, H3

Change hidden activations:

HIDDEN_ACT_1, HIDDEN_ACT_2, HIDDEN_ACT_3

Change batch size and learning rate (BATCH_SIZE, ETA)

Compare generalization performance on the test set
