# Introduction to Machine Learning — Lecture Notes (CS 4810, Fall 2024)

## Lecture 1: What is Machine Learning?

Machine learning (ML) is a subfield of artificial intelligence that gives computers the ability to learn from data without being explicitly programmed. The term was coined by Arthur Samuel in 1959. Tom Mitchell (1997) provided a more formal definition: "A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E."

### Types of Machine Learning

**1. Supervised Learning**
The algorithm learns from labeled training data — input-output pairs where the correct answer is known. The goal is to learn a mapping function f(x) = y.
- **Classification:** Output is a category (e.g., spam vs. not spam, cat vs. dog)
- **Regression:** Output is a continuous value (e.g., house price, temperature)
- Common algorithms: Linear Regression, Logistic Regression, Decision Trees, Random Forests, Support Vector Machines (SVM), k-Nearest Neighbors (k-NN), Neural Networks

**2. Unsupervised Learning**
The algorithm learns from unlabeled data — no correct answers are provided. The goal is to find structure or patterns in the data.
- **Clustering:** Group similar data points (e.g., customer segmentation)
- **Dimensionality Reduction:** Reduce the number of features while preserving important information (e.g., PCA)
- **Anomaly Detection:** Identify unusual data points
- Common algorithms: k-Means Clustering, Hierarchical Clustering, DBSCAN, Principal Component Analysis (PCA), Autoencoders

**3. Reinforcement Learning**
An agent learns by interacting with an environment. It receives rewards or penalties for actions and learns to maximize cumulative reward over time.
- Key concepts: agent, environment, state, action, reward, policy
- Applications: game playing (AlphaGo), robotics, autonomous driving
- Common algorithms: Q-Learning, Deep Q-Network (DQN), Policy Gradient, Proximal Policy Optimization (PPO)

---

## Lecture 2: The Bias-Variance Tradeoff

### Underfitting vs. Overfitting
- **Underfitting (High Bias):** The model is too simple to capture the underlying pattern. Training error and test error are both high.
- **Overfitting (High Variance):** The model is too complex and memorizes noise in the training data. Training error is low but test error is high.

The goal is to find the "sweet spot" — a model complex enough to capture the true pattern but simple enough to generalize to unseen data.

### Mathematical Decomposition
The expected prediction error can be decomposed as:

Error = Bias² + Variance + Irreducible Noise

- **Bias:** Error from simplifying assumptions in the model
- **Variance:** Error from sensitivity to small fluctuations in the training data
- **Irreducible Noise:** Error inherent in the data itself (cannot be reduced by any model)

### Strategies to Manage the Tradeoff
- **Cross-validation:** Use k-fold cross-validation (typically k=5 or k=10) to estimate how well a model generalizes
- **Regularization:** Add a penalty term to the loss function to discourage overly complex models (L1/Lasso, L2/Ridge)
- **Ensemble methods:** Combine multiple models to reduce variance (bagging) or bias (boosting)
- **More data:** Increasing training set size generally reduces variance

---

## Lecture 3: Evaluation Metrics

### Classification Metrics

**Confusion Matrix:**
|  | Predicted Positive | Predicted Negative |
|---|---|---|
| Actual Positive | True Positive (TP) | False Negative (FN) |
| Actual Negative | False Positive (FP) | True Negative (TN) |

**Key Metrics:**
- **Accuracy** = (TP + TN) / (TP + TN + FP + FN) — overall correctness. Misleading on imbalanced datasets.
- **Precision** = TP / (TP + FP) — of all predicted positives, how many are actually positive? Important when false positives are costly (e.g., spam filter).
- **Recall (Sensitivity)** = TP / (TP + FN) — of all actual positives, how many did we find? Important when false negatives are costly (e.g., cancer detection).
- **F1 Score** = 2 × (Precision × Recall) / (Precision + Recall) — harmonic mean of precision and recall. Useful when you want a single metric balancing both.
- **AUC-ROC:** Area Under the Receiver Operating Characteristic curve. Measures the model's ability to distinguish between classes across all thresholds. AUC = 0.5 means random guessing; AUC = 1.0 means perfect classification.

### Regression Metrics
- **Mean Absolute Error (MAE):** Average absolute difference between predicted and actual values. Easy to interpret.
- **Mean Squared Error (MSE):** Average squared difference. Penalizes large errors more heavily.
- **Root Mean Squared Error (RMSE):** Square root of MSE. Same units as the target variable.
- **R² (Coefficient of Determination):** Proportion of variance in the target explained by the model. R² = 1 is perfect; R² = 0 means the model is no better than predicting the mean.

---

## Lecture 4: Decision Trees and Random Forests

### Decision Trees
A decision tree splits data into subsets based on feature values, creating a tree-like structure of if-then rules.

**Splitting Criteria:**
- **Gini Impurity:** Measures the probability of incorrectly classifying a randomly chosen element. Gini = 0 means the node is pure (all same class).
- **Entropy / Information Gain:** Measures the expected reduction in uncertainty. Entropy = 0 means the node is pure.

**Advantages:** Easy to interpret, handles both numerical and categorical data, no feature scaling needed.
**Disadvantages:** Prone to overfitting (especially deep trees), unstable (small data changes → very different tree).

### Random Forests
An ensemble of decision trees, each trained on a random subset of data (bagging) and a random subset of features.

**Key hyperparameters:**
- `n_estimators`: Number of trees (typically 100–500)
- `max_depth`: Maximum depth per tree (controls overfitting)
- `min_samples_split`: Minimum samples required to split a node
- `max_features`: Number of features to consider at each split (default: sqrt(n_features) for classification, n_features/3 for regression)

**Why Random Forests work:** By averaging many uncorrelated trees, variance is reduced without increasing bias. The "wisdom of crowds" effect — individual trees may overfit, but the ensemble generalizes well.

---

## Recommended Textbooks
1. **"Introduction to Statistical Learning" (ISLR)** by James, Witten, Hastie, Tibshirani — free PDF available online, excellent for beginners
2. **"Pattern Recognition and Machine Learning"** by Christopher Bishop — more mathematical, graduate level
3. **"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"** by Aurélien Géron — practical, Python-focused

