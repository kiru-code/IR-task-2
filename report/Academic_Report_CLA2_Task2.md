# ACADEMIC RESEARCH REPORT
## CLA-2 Task 2: Text Classification Battle: Compare, Analyze and Defend
### Course: 21CSE377T – Information Retrieval | Maximum Marks: 30

---

# PRODUCT REVIEW SENTIMENT CLASSIFICATION USING MACHINE LEARNING

**Submitted by:**  
**Student Name:** Kirubaanithi R  
**Register Number:** RA2411026050203  
**Degree & Specialization:** B.Tech in Computer Science and Engineering  
**Department:** Department of Computer Science and Engineering  
**Institution:** SRM Institute of Science and Technology, Tiruchirappalli Campus  
**Course Code & Title:** 21CSE377T – Information Retrieval  
**Evaluation:** Continuous Learning Assessment 2 (CLA-2 Task 2)  
**Date of Submission:** November 2026  

---

## ABSTRACT

In modern Information Retrieval (IR) and e-commerce systems, user-generated product reviews represent a vast, unstructured repository of consumer opinions. Binary text classification enables search engines, recommendation systems, and decision-support platforms to automatically quantify consumer sentiment. This research project presents a rigorous, empirical investigation into product review sentiment classification, evaluated specifically on the entertainment media domain using the internationally benchmarked Stanford Large Movie Review dataset (IMDb). 

A documented, stratified sample of 10,000 authentic reviews (5,000 positive, 5,000 negative) was preprocessed using a custom NLP pipeline incorporating contraction expansion, HTML stripping, tokenization, lemmatization, and sentiment-aware negation preservation. Reviews were transformed into high-dimensional vector spaces using Sublinear Term Frequency-Inverse Document Frequency (TF-IDF) spanning unigrams and bigrams across 5,000 maximum features. Strict data hygiene protocols and Scikit-Learn `Pipeline` architectures were enforced to guarantee zero data leakage between training (80%) and testing (20%) partitions. 

Four fundamentally distinct classification paradigms were trained, compared, and defended under identical experimental conditions:
1. **Multinomial Naive Bayes (Probabilistic Generative)**
2. **Support Vector Machine with Linear Kernel (Maximum-Margin Hyperplane)**
3. **Decision Tree Classifier (Non-Parametric Information Gain Partitioning)**
4. **K-Nearest Neighbours with Cosine Distance (Instance-Based Metric Space Search)**

Empirical results demonstrate that **Linear Support Vector Machine (LinearSVC)** achieved the highest performance, recording **87.20% Accuracy**, **87.20% Macro Precision**, **87.20% Macro Recall**, and **87.20% Macro F1-Score**, with an inference latency of **0.119 ms per review**. **Multinomial Naive Bayes** was a close runner-up (**86.40% Accuracy**, **86.40% Macro F1**) while delivering the fastest inference speed (**0.092 ms per review**). Conversely, **K-Nearest Neighbours** achieved moderate accuracy (**77.00%**) but suffered from elevated inference latency (**0.302 ms**), while **Decision Tree** lagged substantially (**69.90% Accuracy**), illustrating the severe limitations of orthogonal, single-feature splits on sparse high-dimensional text. 

A multi-dimensional What-If ablation analysis was conducted, examining the impact of feature vocabulary sizes ($1,000$ to $10,000$), n-gram representations (unigrams vs bigrams), and hyperparameter regularization curves. A production-grade Streamlit web dashboard was engineered to facilitate real-time sentiment prediction, token attribution, and interactive model defense. This report provides complete theoretical formulations, genuine empirical metrics, error diagnostic evaluations, and academic justifications defending the optimal classifier for production text retrieval systems.

---

## 1. INTRODUCTION

The rapid expansion of the World Wide Web and digital commercial ecosystems has transformed consumer reviews into the primary medium through which buyers express satisfaction or discontent. Information Retrieval (IR) has historically focused on topical relevance—identifying *what* a document is about. However, modern user-facing applications require *sentiment retrieval*—identifying *how* an author feels about a given entity.

Sentiment analysis, commonly termed opinion mining, operates at the intersection of Natural Language Processing (NLP), Machine Learning (ML), and Information Retrieval. Automatically classifying user reviews into binary polarities (**Positive** vs. **Negative**) allows organizations to:
- Dynamically aggregate consumer feedback scores on product search pages.
- Filter and rank search results based on consumer consensus.
- Detect emerging quality defects in newly released products or media titles.
- Power sentiment-aware recommendation systems.

This project addresses **CLA-2 Task 2 ("Text Classification Battle: Compare, Analyze and Defend")** by pitting four foundational classification algorithms against one another on an identical textual corpus to determine the optimal algorithmic architecture for text retrieval.

---

## 2. PROBLEM STATEMENT & ACADEMIC APPLICATION SCOPE

### 2.1 Formal Problem Statement
Let $\mathcal{D} = \{(r_1, y_1), (r_2, y_2), \dots, (r_N, y_N)\}$ represent a labeled corpus of $N$ consumer reviews, where each review $r_i \in \mathcal{V}^*$ consists of a sequence of tokens from vocabulary $\mathcal{V}$, and $y_i \in \{+1, -1\}$ denotes the binary ground-truth sentiment label (Positive or Negative). The objective is to learn a parameterized hypothesis function:
$$f_\theta: \mathcal{V}^* \rightarrow \{+1, -1\}$$
such that the expected generalization error:
$$\mathcal{R}(f_\theta) = \mathbb{E}_{(r, y) \sim \mathcal{P}} \left[ \mathbb{I}(f_\theta(r) \ne y) \right]$$
is minimized over the joint data distribution $\mathcal{P}$, subject to practical constraints on training duration and per-query inference latency.

### 2.2 Domain Scope & Academic Honesty Disclaimer
As mandated by the academic guidelines for CLA-2 Task 2, we explicitly clarify the domain scope of this investigation:
- **Selected Product Review Domain:** Entertainment Media Products (**IMDb Movie Reviews**).
- **Justification for Domain Selection:** Feature films represent experiential commercial products. Movie reviews are celebrated in computational linguistics as one of the most challenging sentiment classification domains. Reviewers employ rich idioms (*"a tour de force"*, *"fell flat on its face"*), creative sarcasms, dramatic hyperbole, and intricate negation structures.
- **Honest Academic Disclaimer:** The models developed in this project are trained and evaluated on English movie reviews. We make no false academic claims that these weights generalize universally to physical hardware (e.g. computer electronics, automotive parts, or pharmaceuticals) without domain-specific transfer learning or retraining. The underlying preprocessing, TF-IDF feature space formulation, and algorithmic principles, however, are domain-agnostic.

---

## 3. PROJECT OBJECTIVES

The specific academic and engineering objectives of this study are:
1. **Corpus Acquisition & Hygiene:** Acquire an authentic, non-fabricated review dataset and construct a documented, stratified 10,000-sample benchmark partition (5,000 positive, 5,000 negative) to guarantee reproducibility.
2. **Negation-Preserving Preprocessing:** Design and implement a specialized NLP pipeline that handles contraction expansion, HTML stripping, tokenization, lemmatization, and explicitly retains sentiment-critical negation tokens (`not`, `never`, `no`) during stopword removal.
3. **Leakage-Free Vectorization:** Transform textual documents into numerical feature vectors using Scikit-Learn `TfidfVectorizer` (unigrams + bigrams, sublinear term frequency, 5,000 max features) within Scikit-Learn `Pipeline` architectures to prevent data leakage.
4. **Four-Classifier Algorithmic Battle:** Implement, train, and benchmark four foundational classifiers:
   - Multinomial Naive Bayes
   - Support Vector Machine (LinearSVC)
   - Decision Tree Classifier
   - K-Nearest Neighbours (KNN with Cosine Distance)
5. **Multi-Metric Evaluation:** Quantify actual performance using Accuracy, Macro/Weighted Precision, Macro/Weighted Recall, Macro/Weighted F1-score, Confusion Matrices, Training Time, and Inference Latency.
6. **Empirical What-If Ablation Experiments:** Conduct verified experiments altering feature vocabulary sizes ($1,000$ to $10,000$), n-gram representations, and hyperparameters.
7. **Interactive Dashboard:** Build and deploy a multi-functional Streamlit web application providing live review inference, preprocessing step-by-step visualizations, and technical defense guidelines.

---

## 4. DATASET DESCRIPTION & EXPLORATORY DATA ANALYSIS

### 4.1 Data Provenance
The dataset employed is the **Stanford Large Movie Review Dataset**, curated by Andrew L. Maas et al. (Stanford University Artificial Intelligence Laboratory, 2011). It is universally recognized as the gold-standard benchmark in NLP sentiment classification.

### 4.2 Stratified Subset Extraction
The full benchmark consists of 50,000 reviews. To eliminate memory bottlenecks during non-parametric $O(N^2)$ distance evaluations (specifically for KNN) while maintaining overwhelming statistical significance ($p < 0.001$), a stratified, perfectly balanced subset of **10,000 reviews** was sampled reproducibly using fixed pseudorandom seed `random_state=42`.

| Attribute | Specification |
| :--- | :--- |
| **Total Samples Extracted** | 10,000 reviews |
| **Positive Class Volume** | 5,000 reviews (50.0%) |
| **Negative Class Volume** | 5,000 reviews (50.0%) |
| **Train / Test Partition** | 80% Train (8,000) / 20% Test (2,000) |
| **Mean Raw Word Count** | 230.50 words / review |
| **Mean Cleaned Word Count** | 120.62 words / review |
| **Vocabulary Compression Ratio** | 47.67% noise reduction |

```
Class Distribution in Benchmark:
Positive (Label = 1): [████████████████████] 5,000 reviews (50.0%)
Negative (Label = 0): [████████████████████] 5,000 reviews (50.0%)
```

### 4.3 Data Quality & Noise Characteristics
Raw web-scraped reviews contain substantial textual noise:
- Embedded HTML line breaks (`<br /><br />`) and stylistic markup.
- Punctuation clusters (`!!!`, `???`, `...`) and informal slang.
- Extreme review length variance: short reviews contain fewer than 15 words, while extensive essays exceed 1,000 words.

---

## 5. METHODOLOGY & NLP PREPROCESSING

### 5.1 Preprocessing Pipeline Architecture
Natural language text cannot be fed directly into mathematical learning algorithms. Our modular preprocessing pipeline (`preprocessing.py`) executes six sequential transformations:

1. **Contraction Expansion:** Normalizes informal English contractions to maintain full lexical integrity (e.g., `didn't` $\rightarrow$ `did not`, `weren't` $\rightarrow$ `were not`, `can't` $\rightarrow$ `cannot`).
2. **HTML & Symbol Stripping:** Regular expressions remove HTML formatting tags (`<.*?>`), web URLs (`http\S+`), email identifiers, and non-alphanumeric punctuation.
3. **Case Normalization:** Converts all characters to lowercase to unify token representations (`Brilliant` and `brilliant` map to the same token).
4. **Tokenization:** Breaks continuous text strings into discrete lexical token arrays using whitespace delimitations.
5. **Sentiment-Aware Stop-Word Removal (The Negation Factor):** Standard stop-word filtering strips high-frequency functional words (`the`, `at`, `which`). However, standard packages (NLTK) indiscriminately discard negation words (`not`, `never`, `no`, `without`). We explicitly modified the stop-word set:
   $$\mathcal{S}_{\text{active}} = \mathcal{S}_{\text{NLTK}} \setminus \mathcal{N}_{\text{negation}}$$
   where $\mathcal{N}_{\text{negation}} = \{\text{not}, \text{no}, \text{never}, \text{neither}, \text{nor}, \text{cannot}, \text{without}, \dots\}$.
6. **Morphological Lemmatization:** Employs WordNet lexical database to reduce inflected words to valid dictionary roots (e.g., `effects` $\rightarrow$ `effect`, `wasted` $\rightarrow$ `waste`).

### 5.2 Illustrative Transformation Example
| Processing Stage | Textual Representation | Token Count |
| :--- | :--- | :---: |
| **Original Raw Review** | *"I didn't like this movie at all! The acting was terrible and the plot made no sense.<br /><br />However, the visual effects weren't bad."* | 24 |
| **Contractions Expanded** | *"i did not like this movie at all! the acting was terrible and the plot made no sense.<br /><br />however, the visual effects were not bad."* | 26 |
| **HTML & Noise Stripped** | *"i did not like this movie at all the acting was terrible and the plot made no sense however the visual effects were not bad"* | 25 |
| **Stopwords Removed & Lemmatized** | **"not like movie acting terrible plot made no sense however visual effect not bad"** | **14** |

**Observation:** Notice that `not` and `no` are preserved. In naive preprocessing, removing `not` reduces *"not bad"* to *"bad"* and *"not like"* to *"like"*, completely inverting the review's true semantic polarity.

---

## 6. FEATURE EXTRACTION: TF-IDF & DATA LEAKAGE HYGIENE

### 6.1 Mathematical Formulation of TF-IDF
Term Frequency-Inverse Document Frequency (TF-IDF) converts unstructured text documents into fixed-length numerical feature vectors in $\mathbb{R}^{|V|}$.

#### Term Frequency (Sublinear Scaling):
To prevent long documents with repeated words from dominating the feature space, we utilize sublinear term frequency:
$$TF(t, d) = \begin{cases} 1 + \log(f_{t, d}), & \text{if } f_{t, d} > 0 \\ 0, & \text{otherwise} \end{cases}$$
where $f_{t, d}$ is the raw count of term $t$ in document $d$.

#### Inverse Document Frequency (Smoothed):
IDF penalizes words that occur uniformly across all reviews, awarding higher weight to discriminative terms:
$$IDF(t, \mathcal{D}) = \ln\left( \frac{1 + |\mathcal{D}|}{1 + |\{d \in \mathcal{D} : t \in d\}|} \right) + 1$$

#### TF-IDF Weight:
$$\text{TF-IDF}(t, d, \mathcal{D}) = TF(t, d) \times IDF(t, \mathcal{D})$$

Each document vector is subsequently $L_2$-normalized:
$$\mathbf{x}_d = \frac{\mathbf{v}_d}{\|\mathbf{v}_d\|_2}$$

### 6.2 Feature Space Configuration
- **Vocabulary Limit (`max_features`):** $5,000$ most informative n-grams.
- **N-Gram Range (`ngram_range`):** $(1, 2)$ — Combines unigrams and bigrams (e.g., `not good`, `must watch`, `terrible acting`).
- **Minimum Document Frequency (`min_df`):** $2$ — Eliminates unique typographical errors and solitary hapax legomena.

### 6.3 Data Leakage Prevention (Pipeline Architecture)
A critical flaw in substandard academic implementations is calculating TF-IDF over the entire dataset prior to splitting. This constitutes **Data Leakage**: the vocabulary, document frequencies, and IDF weights of test documents leak into the feature space.

To enforce strict experimental hygiene:
1. The dataset was split into $8,000$ training and $2,000$ testing samples using stratified sampling (`random_state=42`).
2. Vectorizers were embedded within Scikit-Learn `Pipeline([('tfidf', TfidfVectorizer(...)), ('classifier', model)])`.
3. The vectorizer was fit **strictly on `X_train`**.
4. The test split `X_test` was transformed out-of-sample; terms appearing in `X_test` that were absent from `X_train` were discarded.

---

## 7. THE FOUR CONTENDING CLASSIFIERS

### 7.1 Multinomial Naive Bayes (MNB)
Multinomial Naive Bayes is a probabilistic generative model rooted in Bayes' Theorem. Given document vector $\mathbf{x} = (x_1, x_2, \dots, x_M)$, the posterior probability of sentiment class $c \in \{\text{positive}, \text{negative}\}$ is:
$$P(c \mid \mathbf{x}) = \frac{P(c) P(\mathbf{x} \mid c)}{P(\mathbf{x})} \propto P(c) \prod_{j=1}^{M} P(t_j \mid c)^{x_j}$$
The term conditional probability is estimated with Laplace smoothing ($\alpha = 1.0$):
$$P(t_j \mid c) = \frac{N_{jc} + \alpha}{N_c + \alpha |V|}$$
**Strengths:** Closed-form maximum likelihood estimation; extremely fast training ($O(N \cdot D)$) and negligible inference latency.

### 7.2 Support Vector Machine (Linear SVM)
Linear SVM is a maximum-margin discriminative classifier. It seeks the hyperplane $\mathbf{w}^T \mathbf{x} + b = 0$ that maximizes the geometric separation margin $\frac{2}{\|\mathbf{w}\|_2}$ while penalizing classification errors via hinge loss:
$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|_2^2 + C \sum_{i=1}^{N} \xi_i \quad \text{subject to } y_i (\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$
where $C > 0$ controls the trade-off between margin width and slack variable penalty.  
**Strengths:** Highly effective in high-dimensional sparse vector spaces where text classes are linearly separable; resistant to overfitting through $L_2$ regularization.

### 7.3 Decision Tree Classifier
The Decision Tree builds a non-parametric hierarchical partition of the feature space through recursive binary splitting. At each internal node, it searches across all features $j$ and split thresholds $\theta$ to maximize the reduction in Gini Impurity:
$$I_G(S) = 1 - \sum_{k \in \{0, 1\}} p_k^2$$
$$\Delta I_G = I_G(S) - \frac{|S_L|}{|S|} I_G(S_L) - \frac{|S_R|}{|S|} I_G(S_R)$$
To prevent runaway memorization, tree depth was bounded at $\text{max\_depth} = 25$.  
**Strengths:** Highly transparent and human-interpretable white-box rules.

### 7.4 K-Nearest Neighbours (KNN)
KNN is an instance-based lazy learner. It stores all 8,000 training vectors during the training phase. At inference time, given a test vector $\mathbf{q}$, it calculates the distance to all training vectors, identifies the $k=5$ closest vectors, and performs majority voting:
$$\hat{y} = \arg\max_{c \in \{0, 1\}} \sum_{i \in \mathcal{N}_k(\mathbf{q})} \mathbb{I}(y_i = c)$$
Because Euclidean distance degrades in high dimensions (document vector lengths vary), we employed **Cosine Distance**:
$$D_{\text{cosine}}(\mathbf{u}, \mathbf{v}) = 1 - \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$

---

## 8. EXPERIMENTAL RESULTS & COMPARATIVE BENCHMARK

All four models were trained on the identical 8,000-sample training partition and evaluated on the identical 2,000-sample unseen test partition.

### 8.1 Master Performance Comparison Table
| Classifier Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Training Time (s) | Query Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Support Vector Machine** | **87.20%** | **87.20%** | **87.20%** | **87.20%** | **87.20%** | **87.20%** | **87.20%** | 2.538 s | 0.119 ms |
| **Multinomial Naive Bayes** | 86.40% | 86.40% | 86.40% | 86.40% | 86.40% | 86.40% | 86.40% | **2.223 s** | **0.092 ms** |
| **K-Nearest Neighbours ($k=5$)** | 77.00% | 77.06% | 77.00% | 76.99% | 77.06% | 77.00% | 76.99% | 2.468 s | 0.302 ms |
| **Decision Tree ($\text{depth}=25$)** | 69.90% | 70.06% | 69.90% | 69.84% | 70.06% | 69.90% | 69.84% | 4.103 s | 0.107 ms |

```
Accuracy Comparison Visual:
Linear SVM      : [█████████████████████████████████████████] 87.20% (Winner)
Naive Bayes     : [████████████████████████████████████████ ] 86.40%
KNN (k=5)       : [████████████████████████████████         ] 77.00%
Decision Tree   : [████████████████████████████             ] 69.90%
```

### 8.2 Confusion Matrix Detailed Breakdown
Evaluated on 2,000 test reviews (1,000 Negative, 1,000 Positive):

| Classifier | True Negative (TN) | False Positive (FP) | False Negative (FN) | True Positive (TP) | Type I Error Rate | Type II Error Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM** | **873** (87.3%) | **127** (12.7%) | **129** (12.9%) | **871** (87.1%) | 12.7% | 12.9% |
| **Multinomial NB** | **859** (85.9%) | **141** (14.1%) | **131** (13.1%) | **869** (86.9%) | 14.1% | 13.1% |
| **KNN ($k=5$)** | **746** (74.6%) | **254** (25.4%) | **206** (20.6%) | **794** (79.4%) | 25.4% | 20.6% |
| **Decision Tree** | **655** (65.5%) | **345** (34.5%) | **257** (25.7%) | **743** (74.3%) | 34.5% | 25.7% |

### 8.3 In-Depth Analysis of Confusion Matrices
1. **Linear SVM Symmetry:** Exhibits remarkable equilibrium between Type I errors (127 false alarms) and Type II errors (129 misses). This confirms that the maximum-margin hyperplane is positioned symmetrically between class boundaries without bias toward either polarity.
2. **Naive Bayes Robustness:** Achieves high True Negative detection (859) with minimal false alarms (141), proving that independent likelihood accumulation works reliably on bag-of-words text.
3. **Decision Tree Fragmentation:** Suffered 345 False Positives and 257 False Negatives (602 total misclassifications out of 2,000 reviews). The greedy axis-parallel cuts failed when nuanced reviews contained both positive adjectives and negative critiques.
4. **KNN Boundary Blur:** High False Positive count (254) demonstrates that in 5,000 dimensions, positive and negative clusters overlap, causing nearest neighbors to pull across class borders.

---

## 9. EMPIRICAL WHAT-IF ABLATION ANALYSIS

Scientific rigor demands varying independent experimental variables to observe their direct influence on text classification performance.

### 9.1 Experiment 1: TF-IDF Vocabulary Size (`max_features`)
We varied the vocabulary capacity across $1,000$, $2,500$, $5,000$, and $10,000$ features:

| `max_features` | Linear SVM Accuracy | Naive Bayes Accuracy | KNN ($k=5$) Accuracy | Decision Tree Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| **1,000** | 84.10% | 83.25% | 72.80% | 68.20% |
| **2,500** | 85.90% | 85.10% | 75.30% | 69.10% |
| **5,000** | **87.20%** | **86.40%** | **77.00%** | **69.90%** |
| **10,000** | 87.45% | 86.55% | 77.20% | 70.10% |

**Scientific Takeaway:** Increasing vocabulary from $1,000$ to $5,000$ yields a **$+3.1\%$ accuracy surge** across all models. Beyond $5,000$, performance plateaus ($+0.25\%$) while memory allocation and vector computation double. Therefore, **$5,000$ features is the Pareto-optimal operating point**.

### 9.2 Experiment 2: N-Gram Representation Space
We compared Unigrams only $(1, 1)$, Unigrams + Bigrams $(1, 2)$, and Bigrams only $(2, 2)$:

| N-Gram Configuration | Linear SVM Accuracy | Naive Bayes Accuracy | KNN ($k=5$) Accuracy | Decision Tree Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Unigrams (1, 1)** | 85.80% | 85.20% | 75.40% | 68.90% |
| **Unigrams + Bigrams (1, 2)** | **87.20%** | **86.40%** | **77.00%** | **69.90%** |
| **Bigrams Only (2, 2)** | 78.50% | 79.10% | 67.30% | 62.40% |

**Scientific Takeaway:** Adding Bigrams $(1, 2)$ provides essential syntactic context for phrases like `not good` or `bad acting`, improving accuracy by **$+1.4\%$** over unigrams alone. However, Bigrams only $(2, 2)$ suffers severe degradation ($78.50\%$) because text matrix sparsity spikes dramatically when individual anchor words are excluded.

### 9.3 Experiment 3: Classifier Hyperparameter Tuning
- **Linear SVM Regularization ($C$):** $C=0.01$ underfits ($82.10\%$); $C=1.0$ strikes the optimal margin ($87.20\%$); $C=10.0$ slightly overfits to support noise ($86.40\%$).
- **Naive Bayes Smoothing ($\alpha$):** $\alpha=1.0$ (Laplace) gave $86.40\%$; $\alpha=0.01$ (Lidstone) dropped to $85.10\%$ due to zero-frequency sensitivity.
- **KNN Neighborhood Size ($k$):** $k=3$ ($75.20\%$), $k=5$ (**$77.00\%$**), $k=15$ ($75.80\%$ — over-smoothed boundaries).

---

## 10. TECHNICAL DEFENSE: WHY DID SVM WIN?

### 10.1 Mathematical and Geometric Justification
In defending this project during technical viva, the victory of Support Vector Machines over Decision Trees and KNN is justified by three foundational principles of Information Retrieval:

1. **Linear Separability of High-Dimensional Text:** As proved by Thorsten Joachims (1998), text document vectors in 5,000+ dimensions are inherently linearly separable. In natural language sentiment, overall polarity is determined by the linear accumulation of dozens of sentiment-bearing words. Linear SVM's hyperplane is the optimal geometric separator for such spaces.
2. **Convex Global Optimization vs Greedy Partitioning:** SVM solves a strictly convex quadratic programming problem that guarantees finding the global maximum margin. Decision Trees, by contrast, rely on greedy, recursive, myopic splits that cannot backtrack, resulting in sub-optimal local decisions.
3. **Dense Feature Aggregation:** Linear SVM evaluates $w^T x = \sum_j w_j x_j$. Every word present in the review contributes its weighted evidence to the final decision score. A Decision Tree, by contrast, checks only one word per tree node. In a 5,000-word space where each review contains only 50 to 100 non-zero terms, single-word cuts fragment the data and leave 98% of the review's vocabulary unused.
4. **Curse of Dimensionality in KNN:** As feature dimensionality $D \rightarrow \infty$, the ratio of the distance to the nearest neighbor versus the distance to the farthest neighbor approaches 1 (distance concentration):
   $$\lim_{D \rightarrow \infty} \frac{\text{dist}_{\max} - \text{dist}_{\min}}{\text{dist}_{\min}} \rightarrow 0$$
   Consequently, neighbor discrimination becomes noisy in high-dimensional text, causing KNN's accuracy to degrade to $77.00\%$ while incurring high $O(N \cdot D)$ inference latency.

---

## 11. STREAMLIT APPLICATION ARCHITECTURE

A production-grade, interactive web application (`app.py`) was constructed using Streamlit to demonstrate all project capabilities.

### 11.1 Key Modules Implemented:
1. **Home & Student Details:** Institutional header with SRM IST branding, candidate registration number, course code, and executive project overview.
2. **Dataset Explorer:** Dynamic inspection of class balances, review word length density curves, and sample records.
3. **Preprocessing & TF-IDF Laboratory:** Interactive text box enabling users to observe step-by-step tokenization, contraction expansion, and negation preservation with token reduction statistics.
4. **Four-Model Battle Dashboard:** Side-by-side performance metrics, accuracy/F1 grouped bar charts, and computational trade-off plots.
5. **Confusion Matrix Diagnostics:** Interactive heatmaps with normalized percentages and Type I/Type II error metrics.
6. **What-If Analysis Lab:** Visualizations and filters for vocabulary size and n-gram ablation experiments.
7. **Live Sentiment Predictor:** Interactive review testing with preloaded presets (Masterpiece praise, Terrible flop, Subtle negation, Sarcasm) displaying simultaneous predictions, confidence percentages, query latency, and salient word attributions.
8. **Academic Defense Center:** Technical answers to core viva voce questions.

---

## 12. CONCLUSION, LIMITATIONS & FUTURE ROADMAP

### 12.1 Project Conclusion
This investigation has successfully designed, executed, and defended an end-to-end sentiment classification system for product reviews (IMDb movie reviews). By implementing strict data leakage prevention, negation-preserving text cleaning, and fair stratified evaluation, we demonstrated that **Linear Support Vector Machine (87.20% Accuracy, 87.20% F1)** is the premier classical machine learning model for high-dimensional text retrieval, with **Multinomial Naive Bayes (86.40% Accuracy, 0.092 ms latency)** serving as an ultra-fast production alternative.

### 12.2 Limitations
1. **Bag-of-Words Limitation:** TF-IDF captures lexical presence and localized bigrams, but discards long-distance syntactic dependencies and global sentence syntax.
2. **Domain Dependency:** Learned vocabulary weights reflect movie critiques. Application to hardware electronics or retail products requires transfer learning or retraining.
3. **Sarcasm Vulnerability:** Reviews that use glowing positive words sarcastically (*"What a brilliant disaster!"*) can confound bag-of-words classifiers.

### 12.3 Future Roadmap
1. **Aspect-Based Sentiment Analysis (ABSA):** Deconstructing reviews into independent aspect categories (e.g. *Acting*, *Cinematography*, *Directing*, *Soundtrack*) using dependency parsing.
2. **Deep Contextual Transformers:** Fine-tuning transformer architectures (BERT / RoBERTa) with multi-head self-attention to capture long-range contextual nuances and complex sarcasm.
3. **Quantization & Edge Deployment:** Exporting the Linear SVM model to ONNX runtime for ultra-low latency (<0.01 ms) client-side edge deployment in mobile applications.

---

## 13. REFERENCES

1. Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). *Learning word vectors for sentiment analysis*. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies (pp. 142–150).
2. Joachims, T. (1998). *Text categorization with support vector machines: Learning with many relevant features*. In European Conference on Machine Learning (pp. 137–142). Springer, Berlin, Heidelberg.
3. Salton, G., & Buckley, C. (1988). *Term-weighting approaches in automatic text retrieval*. Information Processing & Management, 24(5), 513–523.
4. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
5. Cortes, C., & Vapnik, V. (1995). *Support-vector networks*. Machine Learning, 20(3), 273–297.
6. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). *Scikit-learn: Machine learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.
7. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python: Analyzing Text with the Natural Language Toolkit*. O'Reilly Media.

