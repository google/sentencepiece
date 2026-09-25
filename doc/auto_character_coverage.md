# Auto-character Coverage (`--auto_character_coverage`): Global Optimization of Character and Subword Budget

SentencePiece 0.2.3 introduces **Auto-character Coverage mode (`--auto_character_coverage=true`)** for both BPE and Unigram models (default is `false`). This mode automatically determines the allocation between single characters and multi-character subwords within a given vocabulary budget via mathematical optimization. Just like Byte-Level BPE, **lossless round-trip is guaranteed**: Auto-character coverage retains a byte fallback tier (`--byte_fallback=true`) for anything outside the learned character set, decomposing unseen codepoints into UTF-8 byte tokens (`<0x00>`–`<0xFF>`) rather than producing `<unk>`.

---

## 1. Introduction

### 1.1 Unicode-Based Subwords vs. Byte-Level BPE (BBPE)

Subword tokenization methods generally fall into two categories: **Unicode character-based** and **byte-level (Byte-Level BPE: BBPE)**.

- **Byte-Level BPE (BBPE)**:
  Uses the 256 raw bytes (`0x00`–`0xFF`) as initial atomic units. Because the base vocabulary is fixed to 256 bytes, no initial character selection is needed, and out-of-vocabulary (OOV) tokens do not occur, making it widely used in Large Language Models (LLMs). On the other hand, because byte sequences are merged without respecting UTF-8 character boundaries, text is frequently split into malformed UTF-8 byte fragments that cannot be independently decoded into valid characters ([Wang et al., 2020](https://arxiv.org/abs/1909.03341); [Jang et al., 2024](https://arxiv.org/abs/2410.23684); [Land & Arnett, 2025](https://arxiv.org/abs/2505.24689)).
- **Unicode Character-Based Methods (SentencePiece + byte-fallback)**:
  Treats Unicode codepoints appearing in the training corpus as candidate atomic units. Because a large multilingual corpus contains tens of thousands of unique Unicode characters, registering all of them within a limited vocabulary budget (e.g., 8,000 to 128,000 pieces) is impossible; instead, high-frequency characters are selected using a heuristic criterion to serve as atomic units (unregistered characters are handled by decomposing them into UTF-8 byte tokens `<0x00>`–`<0xFF>` via `--byte_fallback=true`, guaranteeing lossless round-trip without `<unk>`).

### 1.2 Limitations of Fixed `character_coverage` (0.9995 Heuristic)

Previously, SentencePiece determined the initial character set using a cumulative frequency threshold (default `--character_coverage=0.9995`). While simple, this approach has two limitations:

1. **Lack of Theoretical Objective**:
   The threshold of `0.9995` is an empirical heuristic and does not optimize corpus compression efficiency (Bytes Per Token: BPT) or language model log-likelihood.
2. **Subword Budget Starvation at Small Vocabularies**:
   In multilingual or CJK corpora, reaching 99.95% character coverage often requires 5,000 to 6,000+ single characters. At smaller vocabulary budgets (such as 8,000), single characters consume most of the vocabulary, leaving only around 2,000 slots for subword merges and reducing compression efficiency.

### 1.3 Empirical Findings: Auto-char vs. Byte-Level BPE

`--auto_character_coverage=true` formulates the trade-off between "registering a single character to avoid byte fallback" and "registering a merged subword" on a unified token-reduction scale, optimizing the vocabulary allocation globally.

On a 13-language, 390MB Wikipedia corpus with matched normalization and whitespace-only pretokenization, the following characteristics were observed relative to Byte-Level BPE (see §4 for detailed numbers and per-language breakdowns):

1. **Compression (BPT) on par with Byte-Level BPE, clearly better than fixed 0.9995**:
   Overall BPT is slightly ahead for Byte-Level BPE at 8k, whereas Auto-char is ahead at 16k and above, with the difference remaining within ±1.3% across all vocabulary sizes. Against the fixed `--character_coverage=0.9995` baseline, Auto-char avoids single-character budget starvation and achieves substantially better compression at small vocabularies.
2. **Zero invalid UTF-8 subwords in the vocabulary**:
   Because BBPE merges raw bytes without character boundary constraints, incomplete multi-byte UTF-8 fragments are registered as vocabulary entries. In contrast, because Auto-char constructs subwords at the Unicode character level, **invalid multi-byte UTF-8 fragments in the subword vocabulary are 0 (0.0%)**; out-of-vocabulary characters are handled via explicit single-byte fallback tokens (`<0x00>`–`<0xFF>`).
3. **Trade-offs depending on script and vocabulary size**:
   - **Scripts without whitespace word delimiters (e.g., Thai)**: Because pretokenization units are long, optimizing the character/subword allocation yields a clear advantage for Auto-char across all vocabulary sizes.
   - **CJK and Hangul at small vocabularies**: At small vocabularies, delegating low-frequency Hanzi and Hangul syllables to `byte_fallback` lowers single-language BPT relative to Byte-Level BPE, but the gap largely disappears at 32k–64k and above.
   - **Byte fallback at large vocabularies**: The byte fallback rate drops rapidly as the vocabulary grows and becomes negligible at 128k.

---

## 2. Algorithm: Auto-char + BPE

### 2.1 Why Naive UTF-8 Filtering is Insufficient

A straightforward way to prevent incomplete UTF-8 sequences in BPE would be to start from raw bytes and constrain or filter merge rules so that only valid UTF-8 sequences survive. However, assembling 3- or 4-byte Unicode characters from raw bytes requires multiple intermediate merge steps per character. This consumes the merge budget on character reconstruction rather than forming multi-character words or morphemes, resulting in poor compression.

### 2.2 Global Optimization over the BPE Merge Trajectory

SentencePiece takes a two-phase approach: it first runs standard character-level BPE over all Unicode characters to extract candidate merge rules, and then performs a global optimization along the BPE merge trajectory to determine the optimal number of merges $k$ and the set of single characters to retain vs. delegate to `byte_fallback`.

#### Step 1: Extract Candidate Merge Sequence Over All Unicode Characters
Let $\mathcal{C}$ be the set of all Unicode characters in the corpus. We run standard BPE starting from $\mathcal{C}$ to extract an ordered sequence of candidate multi-character subwords (hereafter **merge tokens**):

$$\mathcal{M} = (s_1, s_2, \dots, s_M)$$

where $f_i$ is the corpus frequency (the number of tokens saved each time merge rule $s_i$ is applied).

#### Step 2: Select Top $k$ Merge Tokens and Merge Gain
To preserve bottom-up merge reachability (ensuring that no selected token requires an omitted intermediate subword), we select a contiguous prefix of $k$ merge tokens $\mathcal{S}_k = \{s_1, s_2, \dots, s_k\}$ ($0 \le k \le M$).
The token reduction gain from these $k$ merge tokens is:

$$G_{\text{merge}}(k) = \sum_{i=1}^{k} f_i$$

#### Step 3: Character Gain and Full Character Closure Constraint
If a Unicode character $c \in \mathcal{C}$ is not in the vocabulary, SentencePiece decomposes it via `byte_fallback` into $b(c)$ byte tokens (`<0x..>`), where $b(c) \in \{1, 2, 3, 4\}$ is the UTF-8 byte length of $c$. Including $c$ in the vocabulary represents it as 1 token, saving $b(c) - 1$ tokens per occurrence.
Thus, the token reduction gain of registering character $c$ with corpus frequency $N(c)$ is:

$$g(c) = N(c) \cdot (b(c) - 1)$$

Let $\mathcal{C}_{\text{sub}}(k) = \bigcup_{i=1}^{k} \text{chars}(s_i)$ denote the set of single characters appearing in the selected $k$ merge tokens $\mathcal{S}_k$.
To ensure that every subword in $\mathcal{S}_k$ can be constructed during tokenization (**Full Character Closure Constraint**), all characters in $\mathcal{C}_{\text{sub}}(k)$ must be included in the initial character set.
Let $R(k) = |\mathcal{C}_{\text{sub}}(k)|$ be the number of these required constituent characters. Their total character gain is:

$$G_{\text{sub-char}}(k) = \sum_{c \in \mathcal{C}_{\text{sub}}(k)} g(c)$$

#### Step 4: Remaining Budget and Standalone Characters
Let $V$ be the target vocabulary size and $N_{\text{res}}$ be the number of reserved tokens (control symbols such as `<s>`, `</s>`, `<unk>`, plus the 256 byte tokens `<0x00>`–`<0xFF>`). The usable vocabulary budget is $W = V - N_{\text{res}}$.

After allocating $k$ merge tokens and $R(k)$ required constituent characters, the remaining budget for additional single characters is:

$$N_{\text{rem}}(k) = W - k - R(k)$$

(Values of $k$ where $k + R(k) > W$ exceed the vocabulary budget and are excluded.)

From the remaining unused characters $\mathcal{C}_{\text{avail}}(k) = \mathcal{C} \setminus \mathcal{C}_{\text{sub}}(k)$, we select the top $N_{\text{rem}}(k)$ standalone characters $\mathcal{C}_{\text{standalone}}(k)$ ordered by gain $g(c)$ descending. Characters not selected in $\mathcal{C}_{\text{sub}}(k) \cup \mathcal{C}_{\text{standalone}}(k)$ are omitted from the vocabulary and handled via `byte_fallback`.
The gain from the selected standalone characters is:

$$G_{\text{standalone}}(k) = \sum_{c \in \mathcal{C}_{\text{standalone}}(k)} g(c)$$

#### Step 5: Global Objective Maximization (Equivalence to Total Token Minimization)
Let $T_{\text{bytes}} = \sum_{c \in \mathcal{C}} N(c) \cdot b(c)$ be the baseline total token count when the entire corpus is encoded purely as raw UTF-8 byte tokens (`<0x00>`–`<0xFF>`). By the Full Character Closure Constraint, every character appearing in the first $k$ merge tokens $\mathcal{S}_k$ is included in the vocabulary ($\mathcal{C}_{\text{sub}}(k)$), so delegating unselected characters to `byte_fallback` does not alter any merge frequency $f_1, \dots, f_k$. Consequently, the exact corpus token count $T(k)$ at prefix length $k$ satisfies:

$$T(k) = T_{\text{bytes}} - J(k)$$

where $J(k)$ is the total token reduction gain relative to raw UTF-8 bytes:

$$
\begin{aligned}
J(k) &= G_{\text{merge}}(k) + G_{\text{sub-char}}(k) + G_{\text{standalone}}(k) \\
     &= \sum_{i=1}^{k} f_i + \sum_{c \in \mathcal{C}_{\text{sub}}(k)} g(c) + \sum_{c \in \mathcal{C}_{\text{standalone}}(k)} g(c)
\end{aligned}
$$

We evaluate $J(k)$ for all feasible $k \in \{0, 1, \dots, M\}$ ($k + R(k) \le W$) and choose the global optimum $k^*$ that maximizes $J(k)$ (equivalently, minimizes total tokens $T(k)$):

$$k^* = \mathop{\mathrm{arg\,max}}_{0 \le k \le M} J(k)$$

### 2.3 Fast Global Search via Fenwick Trees

Evaluating the sum of the top $N_{\text{rem}}(k)$ unused character gains naively at each step $k = 0 \dots M$ would take $O(M \cdot |\mathcal{C}|)$ time.
To perform this efficiently, SentencePiece pre-sorts all characters by gain $g(c)$ and maintains character availability and gain sums using two **Fenwick Trees (Binary Indexed Trees)** with **Binary Lifting**. Each update and top-$N_{\text{rem}}(k)$ query executes in $O(\log |\mathcal{C}|)$, reducing the total search complexity to $O(M \log |\mathcal{C}|)$.

---

## 3. Algorithm: Auto-char + Unigram

In the Unigram model, `--auto_character_coverage=true` is supported in combination with the new Sparse Pruning training algorithm (`--use_sparse_pruning=true`), enabling automatic allocation between single characters and subwords.

Whereas standard Unigram training unconditionally protects the initial character set selected by `--character_coverage=0.9995` from pruning, Auto-char with Sparse Pruning evaluates and prunes both single characters and subwords jointly under a unified log-likelihood objective. Detailed descriptions of the new Sparse Pruning Unigram training algorithm and its formulation with Auto-char are TBA (To Be Announced).

---

## 4. Benchmark Summary (13 Languages, 390MB Wikipedia Corpus)

Trained on a 390.88 MB Wikipedia corpus (`wikimedia/wikipedia`) covering 13 languages (`en`, `de`, `fr`, `vi`, `ru`, `ar`, `he`, `hi`, `th`, `ja`, `zh`, `ko`, `el`) and evaluated on independent 1 MB hold-out texts per language (13 MB total). The `Byte-Level BPE` baseline was trained with the HuggingFace Tokenizers library. All models use `normalization=identity` and **matched whitespace-only pretokenization** (see Appendix A.3 for details).

### 4.1 Overall Performance by Vocabulary Size

| Vocab Size | Model | Single Chars | Subwords | Total Tokens (↓) | Overall BPT (↑) | Invalid UTF-8 Subwords in Vocab* | Encoding Invalid Token Rate |
|---:|---|---:|---:|---:|---:|---:|---:|
| **8,000** | BPE auto-char | 2,291 | 5,450 | 4,101,439 | 3.3258 | **0 (0.0%)** | 5.24% (byte fallbacks) |
| | Unigram auto-char | 1,744 | 5,997 | 4,136,929 | 3.2972 | **0 (0.0%)** | 8.36% (byte fallbacks) |
| | BPE 0.9995 + byte-fallback | 5,901 | 1,840 | 4,711,067 | 2.8954 | **0 (0.0%)** | 0.31% |
| | Unigram 0.9995 + byte-fallback | 5,901 | 1,840 | 4,861,595 | 2.8058 | **0 (0.0%)** | **0.30%** |
| | **Byte-Level BPE** | 344 | 7,653 | **4,086,842** | **3.3377** | 773 (9.7%) | 9.31% |
| **16,000** | BPE auto-char | 3,262 | 12,479 | 3,508,589 | 3.8877 | **0 (0.0%)** | 2.73% (byte fallbacks) |
| | **Unigram auto-char** | 2,529 | 13,212 | **3,505,591** | **3.8911** | **0 (0.0%)** | 4.85% (byte fallbacks) |
| | BPE 0.9995 + byte-fallback | 5,901 | 9,840 | 3,598,841 | 3.7902 | **0 (0.0%)** | 0.41% |
| | Unigram 0.9995 + byte-fallback | 5,901 | 9,840 | 3,646,237 | 3.7410 | **0 (0.0%)** | **0.40%** |
| | Byte-Level BPE | 349 | 15,647 | 3,511,535 | 3.8845 | 928 (5.8%) | 5.28% |
| **32,000** | BPE auto-char | 4,271 | 27,470 | 3,040,301 | 4.4866 | **0 (0.0%)** | 1.40% (byte fallbacks) |
| | **Unigram auto-char** | 3,428 | 28,313 | **3,015,920** | **4.5228** | **0 (0.0%)** | 2.56% (byte fallbacks) |
| | BPE 0.9995 + byte-fallback | 5,901 | 25,840 | 3,054,422 | 4.4658 | **0 (0.0%)** | **0.48%** |
| | Unigram 0.9995 + byte-fallback | 5,901 | 25,840 | 3,080,387 | 4.4282 | **0 (0.0%)** | **0.48%** |
| | Byte-Level BPE | 356 | 31,639 | 3,046,815 | 4.4770 | 1,096 (3.4%) | 2.96% |
| **64,000** | BPE auto-char | 5,415 | 58,326 | 2,662,962 | 5.1223 | **0 (0.0%)** | 0.75% (byte fallbacks) |
| | **Unigram auto-char** | 4,382 | 59,359 | **2,643,222** | **5.1606** | **0 (0.0%)** | 1.34% (byte fallbacks) |
| | BPE 0.9995 + byte-fallback | 5,901 | 57,840 | 2,663,003 | 5.1222 | **0 (0.0%)** | **0.55%** |
| | Unigram 0.9995 + byte-fallback | 5,901 | 57,840 | 2,681,822 | 5.0863 | **0 (0.0%)** | **0.55%** |
| | Byte-Level BPE | 364 | 63,628 | 2,669,852 | 5.1091 | 1,305 (2.0%) | 1.65% |
| **128,000** | BPE auto-char | 6,685 | 121,056 | 2,363,147 | 5.7722 | **0 (0.0%)** | **0.39%** (byte fallbacks) |
| | **Unigram auto-char** | 5,790 | 121,951 | **2,353,548** | **5.7957** | **0 (0.0%)** | 0.62% (byte fallbacks) |
| | BPE 0.9995 + byte-fallback | 5,901 | 121,840 | 2,364,317 | 5.7693 | **0 (0.0%)** | 0.62% |
| | Unigram 0.9995 + byte-fallback | 5,901 | 121,840 | 2,386,457 | 5.7158 | **0 (0.0%)** | 0.62% |
| | Byte-Level BPE | 376 | 127,613 | 2,369,487 | 5.7567 | 1,597 (1.2%) | 0.94% |

> **Note**:
> - BPT (Bytes Per Token) is `Total Raw UTF-8 Bytes / Total Tokens`. Higher values indicate fewer tokens required to encode the same text.
> - `*Invalid UTF-8 Subwords in Vocab`: Counts learned subword/character pieces in the vocabulary (excluding the 256 base byte tokens `<0x00>`–`<0xFF>`) that cannot be decoded into valid UTF-8 strings in isolation. `Encoding Invalid Token Rate` measures the fraction of emitted tokens during evaluation that are raw byte fallback tokens (SentencePiece) or malformed UTF-8 byte fragments (Byte-Level BPE).
> - The BPT gap between Auto-char and Byte-Level BPE is within ±1.3% at every vocabulary size (Byte-Level BPE ahead at 8k, Auto-char ahead at 16k and above), i.e. compression is effectively equivalent, while Auto-char produces zero invalid UTF-8 subwords.
> - The **fraction** of invalid UTF-8 subwords in the Byte-Level BPE vocabulary decreases with vocabulary size (9.7% → 1.2%), but the **absolute count increases monotonically** (773 → 1,597).

---

### 4.2 Per-Language Breakdown

#### (1) Relative BPT Improvement Over Byte-Level BPE

Relative BPT improvement over Byte-Level BPE at each vocabulary size, defined as $\frac{\text{BPT}_{\text{Auto}} - \text{BPT}_{\text{BBPE}}}{\text{BPT}_{\text{BBPE}}} \times 100$ (%):

| Language | Script Family (UTF-8 Bytes) | BPE Auto-char vs Byte-Level BPE (8k / 16k / 32k / 64k / 128k) | Unigram Auto-char vs Byte-Level BPE (8k / 16k / 32k / 64k / 128k) | Characteristic Pattern |
|---|---|---:|---:|---|
| **th** (Thai) | Abugida (Thai, 3B) | **+2.83% / +4.38% / +5.14% / +4.82% / +4.06%** | **+1.23% / +2.72% / +4.89% / +5.15% / +5.74%** | Auto-char clearly ahead at all sizes |
| **he** (Hebrew) | Abjad (Hebrew, 2B) | **+1.45% / +1.21% / +0.56% / +0.52% / +0.59%** | **+6.20% / +5.05% / +3.41% / +2.01% / +2.48%** | Both models ahead at all sizes |
| **ar** (Arabic) | Abjad (Arabic, 2B) | -1.11% / -0.34% / -0.34% / ±0.00% / **+0.04%** | **+2.55% / +3.25% / +3.36% / +2.90% / +2.03%** | Unigram auto-char ahead; BPE on par |
| **de** (German) | Latin (compound words) | -0.53% / -0.23% / -0.04% / ±0.00% / -0.01% | **+1.59% / +2.28% / +2.33% / +2.42% / +1.87%** | Unigram auto-char excels on compounds |
| **ru** (Russian) | Cyrillic (2B) | **+1.24% / +0.59% / +0.28% / +0.57% / +0.71%** | **+1.50% / +1.68% / +0.76% / +0.63% / +1.00%** | Both models slightly ahead at all sizes |
| **hi** (Hindi) | Abugida (Devanagari, 3B) | -0.21% / -0.15% / **+0.44% / +0.10% / +0.33%** | **+1.04% / +1.45% / +1.57% / +1.00% / +1.20%** | Unigram auto-char ahead; BPE on par |
| **el** (Greek) | Alphabet (Greek, 2B) | -0.44% / -0.26% / -0.02% / -0.08% / **+0.06%** | **+2.42% / +1.70% / +1.59% / +0.01%** / -0.31% | Unigram auto-char ahead; gap closes by 64k |
| **en** (English) | Basic Latin (ASCII) | -0.72% / -0.19% / -0.07% / -0.08% / -0.06% | -1.22% / **+0.54% / +1.01% / +1.04% / +0.41%** | Unigram auto-char ahead from 16k onward |
| **fr** (French) | Latin + accents | -0.68% / -0.22% / -0.32% / -0.27% / -0.38% | -0.30% / **+0.59% / +0.67% / +0.32%** / -0.60% | Roughly comparable (within 1%) |
| **ja** (Japanese) | Ideographic + Syllabic | -0.59% / -0.27% / **+0.11% / +0.51% / +0.36%** | -4.80% / -1.30% / **+0.53% / +1.87% / +1.99%** | Auto-char takes the lead from 32k onward |
| **vi** (Vietnamese) | Latin + complex diacritics | -0.24% / -0.33% / -0.05% / -0.07% / -0.04% | -2.20% / -1.89% / -1.10% / -1.55% / -2.20% | BPE on par; Unigram slightly behind |
| **ko** (Korean) | Syllabic (Hangul, 3B) | -0.51% / -0.02% / -0.16% / **+0.06% / +0.13%** | -6.31% / -3.74% / -0.35% / -0.51% / -0.64% | BPE on par; Unigram penalized by pruning Hangul syllables at small vocab |
| **zh** (Chinese) | Ideographic (CJK, 3B) | -2.64% / -0.83% / -0.15% / -0.26% / ±0.00% | -9.63% / -5.41% / -1.94% / **+0.23%** / -0.25% | Behind at small vocab due to Hanzi pruning; largely resolved by 64k |

#### (2) Per-Language BPT and Invalid / Byte Fallback Rate at Vocab Size 32,000

| Language | Byte-Level BPE BPT | BPE auto-char BPT | Unigram auto-char BPT | BPE 0.9995 BPT | Unigram 0.9995 BPT | Byte-Level BPE Invalid Rate | BPE auto-char Fallback Rate | Unigram auto-char Fallback Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **th** (Thai) | 7.5367 | **7.9239** | 7.9053 | 7.8258 | 7.7507 | 3.84% | **0.11%** | 0.18% |
| **hi** (Hindi) | 6.9333 | 6.9639 | **7.0420** | 6.8741 | 6.8656 | 1.53% | **0.27%** | 0.41% |
| **ru** (Russian) | 5.3029 | 5.3177 | **5.3434** | 5.2389 | 5.1435 | 0.29% | **0.05%** | 0.07% |
| **el** (Greek) | 5.1664 | 5.1653 | **5.2484** | 5.0973 | 5.1179 | 0.27% | **0.17%** | 0.25% |
| **ar** (Arabic) | 4.9309 | 4.9141 | **5.0967** | 4.8665 | 4.8911 | 0.19% | **0.15%** | 0.16% |
| **he** (Hebrew) | 4.7355 | 4.7622 | **4.8972** | 4.7108 | 4.6834 | 9.47% | **0.05%** | 0.06% |
| **vi** (Vietnamese) | **4.3032** | 4.3010 | 4.2558 | 4.2792 | 4.1744 | 1.32% | **0.66%** | 1.77% |
| **ja** (Japanese) | 4.2584 | 4.2630 | 4.2808 | 4.3071 | **4.3306** | 5.41% | **4.30%** | 6.88% |
| **ko** (Korean) | **4.0989** | 4.0922 | 4.0844 | 4.0907 | 3.9591 | 5.53% | **2.32%** | 5.25% |
| **fr** (French) | 3.6668 | 3.6550 | **3.6913** | 3.6138 | 3.5788 | **0.20%** | **0.20%** | **0.20%** |
| **en** (English) | 3.6029 | 3.6004 | **3.6395** | 3.5663 | 3.4809 | **0.07%** | 0.08% | **0.07%** |
| **de** (German) | 3.5693 | 3.5679 | **3.6525** | 3.5365 | 3.5340 | 0.12% | **0.09%** | 0.12% |
| **zh** (Chinese) | 3.4554 | 3.4502 | 3.3882 | 3.5335 | **3.5603** | 9.32% | **7.20%** | 12.78% |

> **Note (Global Corpus Optimum vs. Single-Script Character Coverage in Multilingual Corpora)**:
> In Table (2) for Chinese (`zh`) and Japanese (`ja`), single-language BPT is slightly higher for fixed `0.9995` models (`zh`: `3.5335` / `3.5603`, `ja`: `4.3071` / `4.3306`) than for Auto-char. This occurs because in an equally weighted 13-language corpus (30 MB per language), protecting thousands of low-frequency Hanzi that appear only in CJK saves fewer global tokens than allocating those vocabulary slots to high-frequency multi-character subwords shared across Latin, Cyrillic, and Arabic scripts. While fixed `0.9995` unconditionally prioritizes CJK character coverage at the expense of non-CJK subword compression, Auto-char maximizes total token reduction across the entire multilingual corpus. To guarantee character coverage for a specific language, increase that language's share of the training data or specify the characters explicitly with `--required_chars`.

---

### 4.3 Effect of Corpus Size (78 MB → 390 MB)

Keeping the language mix and all other experimental conditions identical, the table below shows how the overall BPT relative to Byte-Level BPE changes when the training corpus grows from 78.26 MB (6 MB per language) to 390.88 MB (30 MB per language), **measured on the same 13 MB evaluation set**.

| Vocab Size | BPE auto-char (78 MB → 390 MB) | Unigram auto-char (78 MB → 390 MB) |
|---:|---:|---:|
| 8,000 | -0.63% → -0.36% | -1.57% → -1.21% |
| 16,000 | -0.13% → +0.08% | -0.55% → +0.17% |
| 32,000 | -0.06% → +0.21% | ±0.00% → +1.02% |
| 64,000 | +0.09% → +0.26% | -0.19% → +1.01% |

- The relative advantage of Auto-char widens as the corpus grows. The effect is largest for Unigram auto-char: at 78 MB it trailed Byte-Level BPE slightly at 16k and 64k, whereas at 390 MB it is ahead at every vocabulary size from 16k upward.
- Conversely, the number of invalid UTF-8 subwords in the Byte-Level BPE vocabulary does not shrink with more data; at 64k it grew from 1,149 to 1,305 (and the encoding invalid token rate from 1.41% to 1.65%). This is a structural property that additional data does not fix.

---

## 5. Usage

Because Auto-char mode relies on decomposing unselected low-frequency characters into byte pieces (`<0x00>`–`<0xFF>`), **`--byte_fallback=true` is mandatory** (training will return an error if `byte_fallback` is `false`). For Unigram models, **`--use_sparse_pruning=true` is also mandatory**.

### Auto-char + BPE
```bash
spm_train \
  --input=corpus.txt \
  --model_prefix=spm_bpe_autochar \
  --vocab_size=32000 \
  --model_type=bpe \
  --auto_character_coverage=true \
  --byte_fallback=true
```

### Auto-char + Unigram
```bash
spm_train \
  --input=corpus.txt \
  --model_prefix=spm_unigram_autochar \
  --vocab_size=32000 \
  --model_type=unigram \
  --auto_character_coverage=true \
  --use_sparse_pruning=true \
  --byte_fallback=true
```

---

## Appendix: Experimental Setup

### A.1 Dataset

| Item | Value |
|---|---|
| Corpus | `wikimedia/wikipedia` (Wikipedia dumps) |
| Languages (13) | `en`, `de`, `fr`, `vi`, `ru`, `ar`, `he`, `hi`, `th`, `ja`, `zh`, `ko`, `el` |
| Training data | 30 MB per language (390.88 MB total) |
| Evaluation data | 1 MB hold-out text per language (13 MB total, disjoint from training data) |
| Vocabulary sizes | 8,000 / 16,000 / 32,000 / 64,000 / 128,000 |

The smaller training corpus used for the comparison in §4.3 has the same 13-language composition with 6.0 MB per language (78.26 MB total), and both sets of models (78 MB and 390 MB) are evaluated on the identical 13 MB evaluation set (1 MB per language). In both corpora, the training and evaluation splits are drawn from disjoint row groups of the same shard.

### A.2 SentencePiece Training Flags (Shared by All Four SPM Models)

```bash
spm_train \
  --input=train_multilingual.txt \
  --model_prefix=<prefix> \
  --model_type=<bpe|unigram> \
  --vocab_size=<8000|16000|32000|64000|128000> \
  --normalization_rule_name=identity \
  --add_dummy_prefix=false \
  --remove_extra_whitespaces=false \
  --split_by_whitespace=true \
  --split_by_unicode_script=false \
  --split_by_number=false \
  --split_digits=false \
  --treat_whitespace_as_suffix=false \
  --allow_whitespace_only_pieces=false \
  --pretokenization_delimiter= \
  --max_sentencepiece_length=128 \
  --max_sentence_length=16384 \
  --byte_fallback=true
```

Model-specific flags:

| Model | Additional Flags |
|---|---|
| BPE auto-char | `--auto_character_coverage=true` |
| Unigram auto-char | `--auto_character_coverage=true --use_sparse_pruning=true` |
| BPE 0.9995 + byte-fallback | `--auto_character_coverage=false --character_coverage=0.9995` |
| Unigram 0.9995 + byte-fallback | `--auto_character_coverage=false --character_coverage=0.9995 --use_sparse_pruning=false` |

### A.3 Byte-Level BPE Configuration (HuggingFace Tokenizers)

The `Byte-Level BPE` baseline was trained with the HuggingFace Tokenizers library. To match SentencePiece's pretokenization exactly, we avoid `Whitespace()` (which also splits on punctuation and digits) and the GPT-2 regex (`ByteLevel(use_regex=True)`), and instead **split on whitespace only, merging whitespace into the following token**. This produces segmentation identical to SentencePiece's `--split_by_whitespace=true`.

```python
tokenizer = Tokenizer(BPE(unk_token=None))
tokenizer.pre_tokenizer = Sequence([
    Split(Regex(r"\s+"), behavior="merged_with_next"),   # whitespace-only splitting
    ByteLevel(add_prefix_space=False, use_regex=False),  # byte-to-visible-char mapping only
])
trainer = BpeTrainer(
    vocab_size=<8000|16000|32000|64000|128000>,
    min_frequency=2,
    special_tokens=[],
    initial_alphabet=ByteLevel.alphabet(),   # 256 raw bytes as the initial alphabet
)
```

> [!NOTE]
> Using `ByteLevel(use_regex=False)` alone applies no pretokenization at all, allowing BBPE to learn multi-word tokens spanning whitespace and making the pretokenization conditions non-equivalent to SentencePiece. It must be combined with `Split` as shown above.

### A.4 Evaluation Metrics

| Metric | Definition |
|---|---|
| BPT (Bytes Per Token) | Total raw UTF-8 bytes of the evaluation text divided by total tokens. Higher is better. |
| Single Chars / Subwords | Among vocabulary pieces excluding control tokens and the 256 raw byte tokens, the number of pieces of length 1 / length ≥ 2. |
| Invalid UTF-8 Subwords in Vocab | Among the pieces above, the number that cannot be decoded into a valid UTF-8 string in isolation. |
| Encoding Invalid Token Rate | Fraction of emitted tokens on the evaluation text that are raw byte fallback tokens (SentencePiece) or malformed UTF-8 byte fragments (Byte-Level BPE). |
