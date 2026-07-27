# Theoretical Foundations of Unigram Tokenization: Discrete Pruning vs. $L_1$ Sparse Proximal Optimization

## Abstract

This document outlines the theoretical, mathematical, and algorithmic foundations unifying **Discrete Pruning** ($\Delta_i = f_i \cdot \delta\text{loss}_i$) and **$L_1$ Sparse Proximal Optimization** ($\mathbb{E}[c_i] - \lambda$) in SentencePiece Unigram Language Model tokenization. We present formal proofs showing that Soft EM lattice expectations $\mathbb{E}[c_i]$ implicitly integrate alternative segmentation distortions ($\delta\text{loss}_i$), establish the sigmoidal mapping between Log-space and Linear probability space, and demonstrate how quantile-based annealing (the 25% rule) guarantees rank-order equivalence across both pruning strategies.

---

## 1. Problem Formulation & Dual Optimization

### 1.1 Discrete Combinatorial Optimization (0-1 Knapsack)
The goal of subword vocabulary selection is to find an optimal subword dictionary $V \subset \mathcal{C}^*$ of size $|V| \le K$ that minimizes total corpus negative log-likelihood $\mathcal{L}(V)$:

$$\min_{V \subset \mathcal{C}^*, \; |V| \le K} \; \mathcal{L}(V) = \sum_{d \in \mathcal{D}} -\log P(d \mid V)$$

In Discrete Pruning, the marginal benefit of retaining a candidate subword $w_i$ is evaluated by its Discrete Partial Derivative (First-Order Taylor Expansion of leave-one-out loss change):

$$\Delta_i = \mathcal{L}(V \setminus \{w_i\}) - \mathcal{L}(V) \approx f_i \cdot \delta\text{loss}_i$$

where:
- $f_i$: Hard Viterbi frequency of $w_i$ on the 1-best segmentation paths.
- $\delta\text{loss}_i = \log p(w_i) - \sum_{k} \log p(w'_{i,k})$: Per-occurrence log-likelihood distortion (irreplacability) when $w_i$ is replaced by alternative subword sequence $w'$.

### 1.2 Continuous $L_1$ Proximal Optimization
In $L_1$ Sparse Unigram Optimization, the discrete knapsack budget is relaxed into a continuous regularized objective:

$$\min_{\mathbf{w} \ge 0} \; \mathcal{J}(\mathbf{w}) = \text{Loss}(\mathbf{w}) + \lambda \|\mathbf{w}\|_1 = \text{Loss}(\mathbf{w}) + \lambda \sum_{i} \mathbb{E}[c_i]$$

where $\lambda$ represents the universal **Shadow Price (Marginal Rate of Substitution)** in units of $[\text{Nats}/\text{Token}]$.

During M-steps, pruning is enforced via proximal soft-thresholding:

$$w_i^{(t+1)} = \max\left(0, \; \mathbb{E}[c_i] - \lambda\right)$$

### 1.3 Comparative Structural Analysis: Discrete Hard Truncation vs. $L_1$ Soft Proximal Shrinkage
- **Role of Post-Cut E-Step in Discrete Pruning**: After dropping the bottom 25% low-utility candidates ($V^{(t+1)} = V^{(t)} \setminus \{\text{bottom 25\%}\}$), running the E-step (Forward-Backward) re-allocates 100% of the lost lattice probability mass from eliminated edges onto the surviving 75% subwords. This dynamic screening (deep annealing) prevents greedy algorithm local minima.
- **Absence of $\lambda$ Parameter Attenuation in Discrete Pruning**: Unlike $L_1$ Sparse Pruning (which subtracts $\lambda$ from expected counts, introducing temporary parameter shrinkage bias), Discrete Pruning performs pure unbiased Maximum Likelihood Estimation (MLE, $\hat{p}_i = c_i / \sum c_j$) on surviving candidates without subtracting $\lambda$.

---

## 2. Hard Viterbi Count ($f_i$) vs. Soft Lattice Expectation ($\mathbb{E}[c_i]$)

### 2.1 E-Step Marginal Posterior Integration
In Forward-Backward (Soft EM), the marginal posterior probability of selecting subword $w_i$ on lattice edge $e = (u, v)$ for document $d$ is:

$$P(e \in \text{Path} \mid d) = \frac{\alpha_u \cdot P(w_i) \cdot \beta_v}{Z(d)}$$

where $Z(d) = \sum_{\pi \in \Omega(d)} \prod_{e \in \pi} P(e)$ is the total lattice partition function (marginal likelihood).

### 2.2 Implicit Integration of Alternative Path Distortions ($\delta\text{loss}_i$)
Because Forward-Backward integrates over all competing parallel lattice paths:
1. **High Irreplacability ($\delta\text{loss}_i \gg 0$)**: Alternative paths bypassing $w_i$ have negligible probability. Probability mass concentrates heavily on $w_i$ ($\alpha_u \cdot \beta_v \approx Z(d)$), pushing $P(w_i \mid d) \to 1.0$ and boosting $\mathbb{E}[c_i]$.
2. **Low Irreplacability ($\delta\text{loss}_i \approx 0$)**: Alternative paths bypassing $w_i$ carry significant probability mass. Probability splits across parallel edges, suppressing $P(w_i \mid d)$ and reducing $\mathbb{E}[c_i]$.

### 2.3 Sparsity Self-Reinforcement Dynamics (Iterative Feedback Loop)
The interplay between M-step proximal soft-thresholding and E-step lattice re-decoding forms a self-reinforcing positive feedback loop:
1. **M-Step Shrinkage**: Subtracting $\lambda$ in $\hat{c}_i^{(t+1)} = \max(0, \mathbb{E}[c_i]^{(t)} - \lambda)$ penalizes weak candidates ($\mathbb{E}[c_i] \approx \lambda$) much more severely in relative percentage terms, lowering their prior probability $P^{(t+1)}(w_i)$.
2. **E-Step Lattice Rerouting**: In the subsequent E-step, the reduced prior $P^{(t+1)}(w_i)$ decreases edge weights for $w_i$. Parallel paths composed of stronger subwords capture the released probability mass, further suppressing $P^{(t+1)}(w_i \mid d)$ and causing $\mathbb{E}[c_i]^{(t+1)}$ to shrink even faster.
3. **Accelerated Convergence to Zero**: This iterative feedback cascade accelerates weak candidates towards exact zero counts ($\mathbb{E}[c_i] < \lambda \implies \hat{c}_i = 0$), naturally separating high-utility subwords from noise fragments within very few EM iterations.

---

## 3. Log-Space ($\delta\text{loss}_i$) vs. Linear Probability Space ($P(w_i)$)

### 3.1 The Sigmoidal Mapping
Let $L_1 = -\log p(w_i)$ and $L_2 = -\sum \log p(w'_i)$. The log-space difference is $\delta\text{loss}_i = L_2 - L_1$.
The Softmax probability of choosing $w_i$ over alternative path $w'_i$ in linear space is:

$$P(w_i) = \frac{e^{-L_1}}{e^{-L_1} + e^{-L_2}} = \frac{1}{1 + e^{-(L_2 - L_1)}} = \sigma(\delta\text{loss}_i)$$

where $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the **Sigmoid (Logistic) Function**.

### 3.2 Rank Monotonicity under Quantile Annealing (The 25% Rule)
Near the pruning threshold (bottom 25% quantile where $\delta\text{loss}_i \approx 0 \sim 1.5$ Nats), the Sigmoid function operates in its linear Taylor expansion regime:

$$\sigma(x) \approx 0.5 + 0.25x$$

Because the transformation between Log-space ($\delta\text{loss}_i$) and Linear-space ($P(w_i)$) is strictly monotonic, ordinal ranking is preserved. Quantile-based pruning (dropping the bottom 25% at each epoch) normalizes scale differences, producing virtually identical vocabulary deletion sets.

---

## 4. Marginal Loss Difference Against ALL Alternative Segmentations

### 4.1 Inequality Formulation & Lower Bound Guarantee
If we evaluate loss distortion against **ALL other alternative segmentations** simultaneously:

Let $Z(d) = Z_{+i}(d) + Z_{-i}(d)$, where $Z_{-i}(d)$ is the partition function of all paths excluding $w_i$.
The exact marginal loss difference across all paths is:

$$\Delta \mathcal{L}_i^{\text{ALL}}(d) = \log Z(d) - \log Z_{-i}(d) = -\log\left(1 - \frac{Z_{+i}(d)}{Z(d)}\right) = -\log(1 - P(w_i \mid d))$$

Because $f(x) = -\log(1 - x)$ is strictly convex for $x \in [0, 1)$, its 1st-order Taylor expansion at $x = 0$ provides a strict lower bound via tangential convexity:

$$-\log(1 - x) \ge x \quad \forall x \in [0, 1)$$

Summing over the entire corpus $\mathcal{D}$:

$$\Delta \mathcal{L}_i^{\text{ALL}} = \sum_{d \in \mathcal{D}} -\log(1 - P(w_i \mid d)) \;\ge\; \sum_{d \in \mathcal{D}} P(w_i \mid d) = \mathbb{E}[c_i]$$

$$\mathbf{\Delta \mathcal{L}_i^{\text{ALL}} \ge \mathbb{E}[c_i]}$$

**Theorem (Conservative Lower Bound Guarantee)**: *The Soft EM Expected Count $\mathbb{E}[c_i]$ is a guaranteed strict lower bound for the true marginal loss difference $\Delta \mathcal{L}_i^{\text{ALL}}$ across all alternative lattice segmentations.*

### 4.2 Large Probability Behavior ($P(w_i \mid d) \to 1$)
When subword $w_i$ is highly irreplaceable in document $d$ ($x = P(w_i \mid d) \to 1$):
1. **Divergence of True Penalty**: While the 1st-order linear term saturates at $\lim_{x \to 1^-} x = 1$, the exact loss penalty diverges to infinity: $\lim_{x \to 1^-} -\log(1 - x) = +\infty$.
2. **Underestimation in High-Utility Regime**: For irreplaceable pieces (e.g., $x = 0.9$), $\mathbb{E}[c_i] = 0.9$ underestimates the exact loss penalty ($-\log(0.1) \approx 2.30$ Nats).
3. **Robustness of Pruning Decisions**: Crucially, both $f(x) = x$ and $g(x) = -\log(1 - x)$ are strictly monotonically increasing on $[0, 1)$. High-probability pieces ($x \approx 1$) remain far above the deletion boundary (bottom 25% quantile). In the boundary region where deletion decisions are actually made ($x \ll 0.1$), the Taylor inequality $-\log(1 - x) \approx x$ holds with near-exact precision.

---

## 5. Numerical Toy Example Verification

Consider a corpus of 100 occurrences of string `"ab"` with candidate subword `"ab"` ($-\log p(\text{ab}) = 2.30$):

| Scenario | Alternative Cost | $\delta\text{loss}_{\text{ab}}$ | Discrete Score $\Delta_{\text{ab}}$ | Soft EM $P(\text{ab})$ | $L_1$ Expectation $\mathbb{E}[c_{\text{ab}}]$ | Final Action |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Scenario A (Irreplacable)** | $9.20$ Nats | $6.90$ Nats | **$690.0$** (High) | $0.999$ | **$99.9$** (High) | **Retained in both** |
| **Scenario B (Replaceable)** | $2.40$ Nats | $0.10$ Nats | **$10.0$** (Low) | $0.524$ | **$52.4$** (Low) | **Pruned in both** |

---

## 6. Post-Lasso Debiased Refit (Unpenalized MLE)

During $L_1$ sparse iterations, proximal soft-thresholding $\max(0, \mathbb{E}[c_i] - \lambda)$ introduces parameter attenuation (L1 shrinkage bias). Furthermore, Bayesian Dirichlet priors ($\psi(x)$ Digamma) penalize small counts.

In Step 5 (**Post-Lasso Refit**), we freeze the active vocabulary $V^*$ and re-estimate probabilities via unpenalized MLE (`ToLogProb`):

$$\hat{p}_i = \frac{c_i}{\sum_{j \in V^*} c_j}$$

This debiasing step eliminates parameter attenuation and Dirichlet prior bias, ensuring exact Maximum Likelihood estimation over the optimal active vocabulary $V^*$.
