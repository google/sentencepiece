# Sparse L1 Unigram (`--use_sparse_pruning`): Vocabulary Selection via Continuous L1 Regularization

SentencePiece 0.2.3 adds **Sparse Pruning (`--use_sparse_pruning=true`)** as an optional vocabulary selection algorithm for the Unigram model (`default = false`; in 0.3.0, this flag is planned to be removed and Sparse Pruning will become the only behavior). Unlike the discrete heuristic pruning of the current default Unigram training (**Discrete Pruning**), Sparse Pruning uses a continuous **L1-regularized EM (Expectation-Maximization)** procedure. This mode is required when [Auto-character Coverage](auto_character_coverage.md) is used with the Unigram model.

---

## 1. Background: Limitations of the Current Default Method (Discrete Pruning)

### 1.1 The Unigram Model and the Vocabulary Selection Problem

The Unigram model ([Kudo, 2018](https://aclanthology.org/P18-1007/)) assigns an occurrence probability $P(x)$ to each subword $x \in V$ in a vocabulary $V$, and defines the probability of a segmentation path $\pi = (x_1, \ldots, x_n)$ of a sentence $d$ as $P(\pi) = \prod_k P(x_k)$. The training objective is the negative log marginal likelihood over the corpus $\mathcal{D}$:

$$
\mathcal{L}(V) = \sum_{d \in \mathcal{D}} -\log \sum_{\pi \in \Omega(d)} P(\pi)
$$

Vocabulary selection is the problem of solving $\min_{|V| \le K} \mathcal{L}(V)$ under a vocabulary size budget $K$. Once $V$ is fixed, $P(x)$ can be estimated with EM; the difficulty lies in the discrete choice of $V$. Because searching over all substrings in the corpus is not practical, SentencePiece uses a **Suffix Array** to extract substrings with frequency at least 2 as the initial candidate (seed) vocabulary, and selects $V$ of size $K$ from it by pruning.

### 1.2 The Current Default Method: Discrete Pruning

The current default implementation (`--use_sparse_pruning=false`, `TrainDiscretePruning` / `PruneSentencePieces`) starts from the seed vocabulary and repeats the following two steps until the vocabulary reaches $K$:

1. **Fix the vocabulary $V$ and run EM** for a few iterations (`--num_sub_iterations`) to estimate $P(x)$.
2. **Estimate a heuristic "removal cost" for each subword $x_i$** and remove the bottom 25% (`--shrinking_factor=0.75`).

Because computing the exact leave-one-out loss $\Delta\mathcal{L}_i = \mathcal{L}(V \setminus \{x_i\}) - \mathcal{L}(V)$ for every candidate is expensive, step 2 relies on the approximation

$$
\Delta_i \approx f_i \cdot \bigl(\log P(x_i) - \textstyle\sum_k \log P(x'_{i,k})\bigr)
$$

where $f_i$ is the number of occurrences of $x_i$ when the corpus is segmented with **Viterbi (1-best)** decoding, and $(x'_{i,1}, x'_{i,2}, \ldots)$ is the **second-best** segmentation of the string $x_i$ without using $x_i$ itself.

### 1.3 Limitations

**(a) Theoretical grounding**

- The removal cost depends on the 1-best frequency $f_i$. A subword whose probability mass is spread over several competing paths in the lattice may be useful, yet receive $f_i = 0$ and be removed (1-best bias).
- The approximation assumes that every occurrence of $x_i$ in the corpus is replaced by the same local second-best sequence $x'_i$, which does not match how probability mass is redistributed across the sentence lattice.
- Although the target problem is formulated as minimizing negative log-likelihood under a discrete budget ($\min_{|V| \le K} \mathcal{L}(V)$), the vocabulary size constraint $|V| \le K$ is handled by a separate heuristic pruning step rather than continuous optimization of a single objective. In particular, **sequence length (token count)** does not appear in $\mathcal{L}(V)$.
- The initial character set selected by `--character_coverage` is excluded from pruning, so the budget allocation between characters and subwords is not optimized.

**(b) Implementation complexity**

- A 2-best lattice search (`Lattice::NBest(2)`) must be run for every subword string to obtain `alternatives[i]`.
- An **additional pass** over the whole corpus with Viterbi decoding is needed to count $f_i$, separate from the Forward-Backward pass of EM.
- Special cases such as "no second-best path exists" or "the 1-best path is already split" require dedicated branches such as `always_keep`.

---

## 2. Formulation of Sparse L1 Unigram

### 2.1 Change of Variables to Unnormalized Frequencies

Applying L1 regularization directly to normalized probabilities $P(x)$ has no effect, because the probability constraint $\sum_x P(x) = 1$ makes the L1 norm $\sum_x P(x) = 1$ constant. We therefore represent each subword $x_i$ by its **non-negative unnormalized expected corpus frequency** $w_i \ge 0$ (measured on the corpus token-count scale so that $w_i = \mathbb{E}[c_i]$ when $\lambda = 0$) and define $P_{\mathbf{w}}(x_i) = w_i / \|\mathbf{w}\|_1$ with $\|\mathbf{w}\|_1 = \sum_j w_j$.

### 2.2 Objective

The discrete size constraint $|V| \le K$ is relaxed into the following **continuous L1-regularized objective**:

$$
\min_{\mathbf{w} \ge 0} \; \mathcal{J}(\mathbf{w}) = \mathcal{L}(\mathbf{w}) + \lambda \|\mathbf{w}\|_1
$$

where $\lambda > 0$ is the regularization parameter. A subword with $w_i = 0$ drops out of the vocabulary, so $\|\mathbf{w}\|_0 = |V|$, and the L1 term is the convex (Lasso-type) relaxation of the L0 constraint $|V| \le K$.

### 2.3 L1-Regularized EM and the Proximal Update

The objective is optimized with EM.

- **E-step**: Run Forward-Backward under the current $P_{\mathbf{w}}$ and compute the **expected count** $\mathbb{E}[c_i] = \sum_d \mathbb{E}[c_i \mid d]$ of each subword (identical to standard Unigram).
- **M-step**: Apply the proximal operator of the L1 term (soft-thresholding):

$$
w_i^{(t+1)} = \mathrm{prox}_{\lambda\|\cdot\|_1}\bigl(\mathbb{E}[c_i]\bigr) = \max\bigl(0,\; \mathbb{E}[c_i] - \lambda\bigr)
$$

While standard maximum-likelihood estimation sets $w_i \leftarrow \mathbb{E}[c_i]$, the L1 version **subtracts the same $\lambda$ from every subword and sets negative values to 0 (= removal)**. The E-step is unchanged, and the M-step adds this subtraction.

### 2.4 Choosing $\lambda$: Adjust $\lambda$ so that $K$ weights remain non-zero

Rather than fixing $\lambda$, the implementation (`TrainSparsePruning`) **iteratively adjusts $\lambda$ so that the vocabulary size reaches $K$**:

1. Run the E-step to obtain $\mathbb{E}[c_i]$.
2. While the number of active subwords is sufficiently larger than $K$, set $\lambda^{(t)}$ to the threshold that removes 25% of the active subwords, i.e., the lower 25% quantile of $\mathbb{E}[c_i]$. $\lambda$ is updated monotonically ($\lambda^{(t)} \leftarrow \max(\lambda^{(t-1)}, \text{quantile})$).
3. Apply the M-step $w_i \leftarrow \max(0, \mathbb{E}[c_i] - \lambda^{(t)})$ and normalize to probabilities.
4. Repeat until the number of active subwords reaches $K$.
5. Finally, run one ordinary EM step with $\lambda = 0$ on the surviving vocabulary to remove the shrinkage bias introduced by L1 (`--post_l1_debias=true`, i.e., a post-Lasso / debiased refit).

This is the standard Lagrangian-relaxation / dual-ascent view of the constrained problem $\min \mathcal{L}(\mathbf{w})$ s.t. $\|\mathbf{w}\|_0 \le K$: replace the constraint by a multiplier $\lambda$ and **adjust the multiplier until the constraint is met**. Raising $\lambda$ gradually in 25% steps rather than all at once follows the same schedule as `shrinking_factor=0.75` in standard Unigram, and can be seen as a discretized **regularization path** along which the solution is tracked while EM re-estimates the parameters.

**Related known techniques**

- Variable selection by L1 regularization (Lasso): [Tibshirani (1996)](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x)
- Soft-thresholding as the proximal operator of L1: [Parikh & Boyd (2014)](https://doi.org/10.1561/2400000003)
- Re-estimation without penalty after selection (post-Lasso): [Belloni & Chernozhukov (2013)](https://doi.org/10.3150/11-BEJ410)

Sparse Pruning combines these ingredients with Unigram EM, whose latent variables are segmentation paths.

---

## 3. Theoretical Properties

Introducing L1 regularization gives the pruning step the following four interpretations.

### 3.1 $\|\mathbf{w}\|_1$ is the expected token count

On the expected-count scale $w_i = \mathbb{E}[c_i]$ (before subtracting $\lambda$),

$$
\|\mathbf{w}\|_1 = \sum_i \mathbb{E}[c_i] = \mathbb{E}[\text{total number of tokens in the corpus}]
$$

The objective $\mathcal{J} = \mathcal{L} + \lambda \|\mathbf{w}\|_1$ is therefore a **weighted sum of the negative log-likelihood and the expected total number of tokens needed to encode the corpus**, including sequence length (token count) directly in the objective.

### 3.2 $\mathbb{E}[c_i]$ is a lower bound on the leave-one-out loss

Splitting $Z(d) = \sum_\pi P(\pi)$ into paths that contain $x_i$ and paths that do not, the increase in loss on sentence $d$ when $x_i$ is removed from the vocabulary (with other weights held fixed) is

$$
\Delta\mathcal{L}_i(d) = \log Z(d) - \log Z_{-i}(d) = -\log\bigl(1 - P(x_i \mid d)\bigr)
$$

Since $-\log(1-p) \ge p$ for $0 \le p < 1$, summing over the corpus gives

$$
\Delta\mathcal{L}_i = \sum_d -\log\bigl(1 - P(x_i \mid d)\bigr) \;\ge\; \sum_d P(x_i \mid d) = \mathbb{E}[c_i]
$$

That is, **the expected count $\mathbb{E}[c_i]$, already available from the E-step, is a lower bound on the loss increase caused by removing $x_i$.**

- Unlike $\Delta_i$, which relies on 1-best frequencies and a second-best alternative, $\mathbb{E}[c_i]$ accounts for all paths and has no 1-best bias.
- Both $-\log(1-p)$ and $p$ are monotonically increasing, so the ranking within each sentence is preserved. In the low-frequency region where pruning decisions are made ($P(x_i \mid d) \ll 1$), $-\log(1-p) \approx p$ and the bound is tight.
- When $P(x_i \mid d) \to 1$ (there is no segmentation of $d$ that avoids $x_i$), the loss diverges while the bound saturates at 1. This property is used when pruning characters (Auto-char, Section 4).
- No 2-best search or extra Viterbi pass is required; **the pruning criterion is already computed in the E-step**, which simplifies the implementation.

### 3.3 Subtracting $\lambda$ acts as a rich-get-richer regularizer

Subtracting the same $\lambda$ from every subword in the M-step is a penalty that is **non-uniform in relative terms**:

- High-frequency subwords ($\mathbb{E}[c_i] \gg \lambda$): the relative reduction is small.
- Low-frequency subwords ($\mathbb{E}[c_i] \approx \lambda$): a large part of the mass is removed, or the weight becomes 0.

In the next E-step, paths containing the shrunk low-frequency subwords receive lower posterior probability, and the mass moves to paths containing higher-frequency subwords. Through this positive feedback (rich-get-richer), competing redundant fragments are eliminated across iterations. This is **the same mechanism by which Lasso shrinkage drives solutions toward sparsity**, operating inside EM.

The same mechanism can be read as a penalty on the number of splits. Representing a string with a single token $x$ subtracts $\lambda$ once; splitting it into two tokens $(y, z)$ subtracts $\lambda$ once for each token, i.e., twice in total. **Each additional split costs an extra $\lambda$ on the expected-count scale**, so among segmentations with similar probabilities, the one with fewer tokens is preferred.

> **Note**:
> This shrinkage is a regularizer for vocabulary selection and need not remain in the final probability estimates. After the vocabulary is fixed, `--post_l1_debias=true` (default) re-estimates with $\lambda = 0$, restoring ordinary maximum-likelihood estimates for the surviving vocabulary.

### 3.4 $\lambda$ is a shadow price (Lagrange multiplier)

In terms of units, $\mathcal{L}$ is measured in nats and $\|\mathbf{w}\|_1$ in tokens, so

$$
\lambda = \frac{\Delta \mathcal{L}}{\Delta(\text{number of tokens})} \quad [\text{nats/token}]
$$

In optimization terminology, $\lambda$ is the **Lagrange multiplier** of the constraint $\|\mathbf{w}\|_0 \le K$ (relaxed to $\|\mathbf{w}\|_1$); in economics the same quantity is called a **shadow price**: the log-likelihood cost of saving one token by using one vocabulary slot.

- A subword with $\mathbb{E}[c_i] \ge \lambda$ is judged to be worth its vocabulary slot (in terms of tokens saved) and survives.
- The iteration in Section 2.4 can be interpreted as gradually raising this price until **demand (the number of surviving subwords) equals supply (the budget $K$)**, i.e., finding the equilibrium price $\lambda^*$.

---

## 4. Relation to Auto-character Coverage

[Auto-character Coverage](auto_character_coverage.md) (`--auto_character_coverage=true`) does not pre-select characters with `--character_coverage`, puts every character in the corpus into the seed vocabulary, prunes unnecessary characters under the same criterion as subwords, and leaves the pruned characters to byte fallback at inference time (`--byte_fallback=true`). For Unigram, this mode requires `--use_sparse_pruning=true`.

In both Discrete Pruning's 2-best comparison ($\Delta_i = f_i \cdot \delta\mathrm{loss}_i$) and the exact leave-one-out loss ($\Delta\mathcal{L}_i = -\sum_d \log(1 - P(x_i \mid d))$), evaluating the removal of a single character $c$ is problematic when no shorter alternative path exists in the training lattice: the alternative path probability is $0$ ($P(c \mid d) = 1$), so the loss increase diverges to $+\infty$ (which is why Discrete Pruning unconditionally excludes single characters from pruning).

Using the lower bound $\mathbb{E}[c_i]$ from Section 3.2 resolves this naturally because $\mathbb{E}[c_i]$ handles both cases with a single formula:

1. **When alternative paths exist ($P(x_i \mid d)$ is small, as for typical subwords)**: Since $-\log(1 - u) \approx u$, the expected count $\mathbb{E}[c_i]$ behaves directly as an approximation to the leave-one-out log-likelihood loss.
2. **When no alternative path exists ($P(c \mid d) \to 1$, as for single-character tokens)**: Instead of diverging like $-\log(1 - u) \to \infty$, each occurrence saturates at $u = 1$, and the corpus sum $\mathbb{E}[c]$ reduces to the expected number of times character $c$ is used as a single-character token—which is proportional to the number of extra byte fallback tokens that would be emitted at inference time if $c$ were removed.

Thus, without introducing any special branch or ad-hoc byte probabilities in the training lattice, the single rule $\mathbb{E}[c_i] < \lambda$ simultaneously prunes subwords with small leave-one-out loss and leaves rare characters (used fewer than $\lambda$ times as single-character tokens) to byte fallback.

---

## 5. Benchmark Summary (13 Languages, 390 MB Wikipedia Corpus)

To isolate the effect of Sparse Pruning (`--use_sparse_pruning=true`) from character selection, we evaluate `Unigram Discrete`, `Unigram Sparse`, and `BPE` under the same fixed character coverage (`--character_coverage=0.9995`, 5,940 single characters) on a 390 MB Wikipedia corpus (`wikimedia/wikipedia`, 30 MB per language across 13 languages: `en`, `de`, `fr`, `vi`, `ru`, `ar`, `he`, `hi`, `th`, `ja`, `zh`, `ko`, `el`) and a separate 13 MB evaluation set (1 MB per language). All models use `normalization_rule_name=identity`, `--split_by_whitespace=true`, `--split_by_unicode_script=false`, `--split_by_number=false`, and `--byte_fallback=true` (see [auto_character_coverage.md Appendix A](auto_character_coverage.md#appendix-experimental-setup) for full training flags).

For benchmark results combining Sparse Pruning with automatic character selection (`--auto_character_coverage=true --use_sparse_pruning=true`, including comparisons against Byte-Level BPE), see [Auto-character Coverage §4](auto_character_coverage.md#4-benchmark-summary-13-languages-390mb-wikipedia-corpus).

### 5.1 Overall Performance by Vocabulary Size

All models share the same 5,940 single characters (`--character_coverage=0.9995`), 256 byte tokens, and 3 control tokens; the remaining $K - 6,199$ slots are multi-character subwords (1,801 / 9,801 / 25,801 / 57,801 / 121,801 at $K = 8\text{k} \dots 128\text{k}$).

| Vocab Size | Model | Total Tokens (↓) | Overall BPT (↑) | Corpus Passes (E-step + Prune) |
|---:|---|---:|---:|---:|
| **8,000** | Unigram Discrete | 4,911,865 | 2.7762 | 45 (30 + 15) |
| | Unigram Sparse | 4,854,101 | 2.8092 | **20 (20 + 0)** |
| | **BPE** | **4,757,263** | **2.8664** | — |
| **16,000** | Unigram Discrete | 3,701,860 | 3.6836 | 39 (26 + 13) |
| | **Unigram Sparse** | **3,642,149** | **3.7440** | **17 (17 + 0)** |
| | BPE | 3,650,479 | 3.7355 | — |
| **32,000** | Unigram Discrete | 3,136,294 | 4.3479 | 33 (22 + 11) |
| | **Unigram Sparse** | **3,078,441** | **4.4296** | **15 (15 + 0)** |
| | BPE | 3,105,444 | 4.3911 | — |
| **64,000** | Unigram Discrete | 2,731,505 | 4.9922 | 24 (16 + 8) |
| | **Unigram Sparse** | **2,693,741** | **5.0622** | **13 (13 + 0)** |
| | BPE | 2,717,130 | 5.0186 | — |
| **128,000** | Unigram Discrete | 2,431,350 | 5.6085 | 18 (12 + 6) |
| | **Unigram Sparse** | **2,399,733** | **5.6824** | **10 (10 + 0)** |
| | BPE | 2,415,118 | 5.6462 | — |

Key observations:
- **Direct comparison (`Unigram Discrete` vs `Unigram Sparse`)**: Holding the character vocabulary fixed at 5,940 single characters and changing only `--use_sparse_pruning=false` $\to$ `true` improves overall BPT across all vocabulary sizes by **+1.19% to +1.88%** (reducing token count on the 13 MB evaluation set by 31k–60k tokens).
- **Comparison against BPE**: Under Discrete Pruning, `Unigram Discrete` produces lower BPT than `BPE` at every vocabulary size. With Sparse Pruning, `Unigram Sparse` exceeds `BPE` from 16,000 upward (`3.7440` vs `3.7355` at 16k; `4.4296` vs `4.3911` at 32k; `5.0622` vs `5.0186` at 64k; `5.6824` vs `5.6462` at 128k).
- **Computational cost**: Discrete Pruning executes two E-steps (`--num_sub_iterations=2`) plus one Viterbi corpus segmentation pass (`PruneSentencePieces`) and 2-best lattice searches per outer stage. Sparse Pruning executes one E-step per outer stage (plus one refit E-step at the end) with no Viterbi pruning pass or 2-best search, reducing total corpus passes by roughly half (e.g., 33 $\to$ 15 passes at $K = 32,000$).

### 5.2 Per-Language BPT at Vocabulary Size 32,000

| Language | Script Family | Unigram Discrete | Unigram Sparse | BPE |
|---|---|---:|---:|---:|
| **th** (Thai) | Abugida (Thai) | **7.6930** | 7.6209 | 7.6615 |
| **hi** (Hindi) | Abugida (Devanagari) | 6.9042 | **7.0455** | 6.9782 |
| **ru** (Russian) | Cyrillic | 4.9693 | **5.0806** | 5.0566 |
| **el** (Greek) | Greek | 4.9912 | **5.0895** | 5.0605 |
| **ar** (Arabic) | Abjad (Arabic) | 4.7515 | **4.8679** | 4.7355 |
| **he** (Hebrew) | Abjad (Hebrew) | 4.5965 | **4.7431** | 4.6214 |
| **vi** (Vietnamese) | Latin + diacritics | 4.3439 | 4.4109 | **4.4208** |
| **ja** (Japanese) | Ideographic + Syllabic | 4.0335 | **4.0456** | 4.0078 |
| **ko** (Korean) | Syllabic (Hangul) | 3.8206 | **4.0252** | 4.0054 |
| **de** (German) | Latin (compounds) | 3.6314 | **3.6577** | 3.6053 |
| **fr** (French) | Latin | 3.4634 | **3.5081** | 3.4720 |
| **zh** (Chinese) | Ideographic (CJK) | 3.4903 | 3.5108 | **3.5215** |
| **en** (English) | Basic Latin | 3.3877 | **3.4889** | 3.4577 |

- At $K = 32,000$, `Unigram Sparse` improves BPT over `Unigram Discrete` in 12 of the 13 languages, with the largest gains in Korean (`+5.35%`), Hebrew (`+3.19%`), English (`+2.99%`), Arabic (`+2.45%`), and Russian (`+2.24%`), and achieves the highest BPT among the three models in 10 of the 13 languages.

---

## 6. Usage

```bash
spm_train \
  --input=corpus.txt \
  --model_prefix=spm_unigram_sparse \
  --model_type=unigram \
  --vocab_size=32000 \
  --use_sparse_pruning=true \
  --post_l1_debias=true      # default: true
```

Combined with Auto-char:

```bash
spm_train \
  --input=corpus.txt \
  --model_prefix=spm_unigram_autochar \
  --model_type=unigram \
  --vocab_size=32000 \
  --use_sparse_pruning=true \
  --auto_character_coverage=true \
  --byte_fallback=true
```

| Flag | Default | Description |
|---|---|---|
| `--use_sparse_pruning` | `false` | Perform Unigram vocabulary selection with L1-regularized EM (Sparse Pruning). Currently an experimental `spm_train` CLI-only flag; in 0.3.0 this flag will be removed and Sparse Pruning will become the only behavior. |
| `--post_l1_debias` | `true` | After the vocabulary is fixed, re-estimate with $\lambda = 0$ to remove the L1 shrinkage bias. Currently a `spm_train` CLI-only flag. |

The training log reports, for each epoch, `active` (number of active subwords), `lambda`, `obj` (per-sentence negative log-likelihood), `|w|` (Viterbi 1-best token count across the corpus), and `bytes/tok` (corpus bytes divided by `|w|`).

---

## 7. Summary of Differences

| Aspect | Discrete Pruning (current default) | Sparse Pruning (L1) |
|---|---|---|
| Pruning criterion | $\Delta_i \approx f_i \cdot \delta\mathrm{loss}_i$ (1-best frequency × 2-best alternative) | $\mathbb{E}[c_i]$ (expected count over all paths; lower bound on the leave-one-out loss) |
| Objective formulation | Discrete $\min_{\lvert V \rvert \le K} \mathcal{L}(V)$ approximated by alternating EM and heuristic pruning | Continuous $\mathcal{L} + \lambda\|\mathbf{w}\|_1$ optimized consistently |
| Token count | Absent from the objective | Explicit as $\|\mathbf{w}\|_1$ = expected token count |
| Probability re-estimation | $w_i \leftarrow \mathbb{E}[c_i]$ (with digamma update) | $w_i \leftarrow \max(0, \mathbb{E}[c_i] - \lambda)$ (debiased at the end) |
| Extra computation | 2-best search + Viterbi re-segmentation pass | None (already computed in the E-step) |
| Characters | Initial character set excluded from pruning | Same criterion as subwords (Auto-char) |
