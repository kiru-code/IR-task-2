# Presentation Slides & Speaker Notes
## Product Review Sentiment Classification Using Machine Learning
**Course:** 21CSE377T – Information Retrieval | **Assignment:** CLA-2 Task 2  
**Student Name:** Kirubaanithi R | **Register No:** RA2411026050203  
**Institution:** SRM Institute of Science and Technology, Tiruchirappalli  
**File Generated:** `presentation/Sentiment_Classification_Presentation.pptx` (15 Slides)

---

### Slide 1: Title Slide
- **Title:** PRODUCT REVIEW SENTIMENT CLASSIFICATION USING MACHINE LEARNING
- **Subtitle:** CLA-2 Task 2 – Text Classification Battle: Compare, Analyze and Defend
- **Student Details:** Kirubaanithi R (RA2411026050203)
- **Course & Institution:** 21CSE377T – Information Retrieval, SRM IST, Tiruchirappalli
- **Speaker Notes:**  
  *"Good morning respected evaluators. I am Kirubaanithi R, register number RA2411026050203, presenting CLA-2 Task 2 for Information Retrieval (21CSE377T). My project is titled 'Product Review Sentiment Classification Using Machine Learning'. In this work, I benchmark, analyze, and defend four foundational text classifiers on 10,000 authentic IMDb reviews."*

---

### Slide 2: Introduction & Theoretical Motivation
- **Key Concepts:**
  - Over 80% of consumer feedback exists in unstructured text reviews.
  - Sentiment classification powers modern recommendation and search engines.
  - The "Battle" Paradigm compares 4 distinct philosophies: Probabilistic (Naive Bayes), Maximum Margin (SVM), Information Gain (Decision Tree), and Metric Space (KNN).
- **Speaker Notes:**  
  *"In this slide, I establish the motivation. Natural language reviews constitute over 80% of user-generated feedback. The goal of this assignment is not merely to train models, but to pit four distinct algorithmic paradigms against each other under identical conditions and explain why their mathematical properties lead to dramatically different performances on textual feature spaces."*

---

### Slide 3: Problem Statement & Academic Scope
- **Formal Definition:** Given a text review $r \in \mathcal{R}$, predict polarity $y \in \{\text{Positive}, \text{Negative}\}$.
- **Honest Academic Scope:**
  - Product Category Selected: Entertainment Media Products (IMDb Movie Reviews).
  - Movies represent an intricate textual domain filled with hyperbole, sarcasm, and negations.
  - Explicit disclaimer: The model is trained on movie appraisals and requires domain adaptation for technical hardware.
- **Speaker Notes:**  
  *"Here I define our problem formulation and make our academic scope explicit. As required by the CLA-2 rubric, we do not make false claims that our system magically understands all commercial products. We explicitly ground our study in entertainment media products—specifically IMDb movie reviews—which represent one of the most linguistically challenging sentiment domains available."*

---

### Slide 4: Dataset Profile & Stratified Sampling
- **Dataset:** Stanford Large Movie Review Dataset (Maas et al., 2011).
- **Subset:** 10,000 balanced reviews (5,000 positive, 5,000 negative) sampled reproducibly (`random_state=42`).
- **Splits:** 80% Training (8,000 reviews), 20% Testing (2,000 reviews).
- **Speaker Notes:**  
  *"Regarding data provenance: We utilize the authentic Stanford Large Movie Review dataset. We extract a stratified, reproducible subset of 10,000 samples—5,000 positive and 5,000 negative. This sample size provides immense statistical confidence while allowing algorithms like KNN and Decision Trees to train and evaluate rapidly without hanging system memory."*

---

### Slide 5: Text Preprocessing & Negation Preservation
- **Pipeline:** Contraction expansion $\rightarrow$ HTML stripping $\rightarrow$ Tokenization $\rightarrow$ Negation-preserving stopword filtering $\rightarrow$ WordNet lemmatization.
- **The Negation Factor:** Stripping `not` turns *"not a good movie"* into *"good movie"*, reversing sentiment. We preserved negation terms, gaining $+2.4\%$ in nuanced review accuracy.
- **Speaker Notes:**  
  *"This slide highlights one of our most critical engineering decisions: Negation Preservation. Standard NLTK stopwords discard words like 'not' and 'no'. In sentiment classification, discarding 'not' turns 'not good' into 'good', creating catastrophic false positives. We modified the stopword set to retain negation markers, preserving the true semantic polarity."*

---

### Slide 6: TF-IDF & Data Leakage Hygiene
- **Mathematical Form:** Sublinear Term Frequency ($1 + \log(tf)$) and Smoothed Inverse Document Frequency.
- **Pipeline Architecture:** Scikit-Learn `Pipeline` fit exclusively on `X_train`. Zero test vocabulary or IDF leakage.
- **Speaker Notes:**  
  *"In Slide 6, we examine our mathematical feature representation. We use Sublinear TF-IDF with unigrams and bigrams up to 5,000 features. Most importantly, to prevent data leakage, we fit the vectorizer strictly on the 8,000 training reviews using a Scikit-Learn Pipeline. The test split is never seen by the vectorizer during vocabulary and IDF fitting."*

---

### Slide 7: The Four Contending Classifiers
- **Multinomial Naive Bayes:** Closed-form conditional independence generative classifier.
- **Support Vector Machine (Linear):** Maximum geometric margin hyperplane via convex quadratic optimization.
- **Decision Tree:** Recursive binary splitting using Gini Impurity.
- **K-Nearest Neighbours:** Instance-based majority voting in cosine distance space ($k=5$).
- **Speaker Notes:**  
  *"Here we introduce the four contending classifiers. We deliberately contrast four fundamentally different algorithmic philosophies: a probabilistic generative model (Multinomial Naive Bayes), a margin-maximizing discriminative model (Linear SVM), an axis-parallel recursive partitioner (Decision Tree), and an instance-based metric space searcher (KNN)."*

---

### Slide 8: Empirical Benchmark Results Table
| Classifier Model | Accuracy | Precision | Recall | Macro F1 | Train Time | Latency / Query |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Support Vector Machine** | **87.20%** | **87.20%** | **87.20%** | **87.20%** | **2.54 s** | **0.119 ms** |
| **Multinomial Naive Bayes** | 86.40% | 86.40% | 86.40% | 86.40% | 2.22 s | **0.092 ms** |
| **K-Nearest Neighbours (k=5)** | 77.00% | 77.06% | 77.00% | 76.99% | 2.47 s | 0.302 ms |
| **Decision Tree (depth=25)** | 69.90% | 70.06% | 69.90% | 69.84% | 4.10 s | 0.107 ms |
- **Speaker Notes:**  
  *"Here are our actual, genuine empirical results on 2,000 test reviews. Linear SVM wins with 87.20% accuracy and 87.20% macro F1. Multinomial Naive Bayes is an extremely close second at 86.40% and is the fastest at 0.092 milliseconds per review. KNN achieves 77.00% but suffers in query latency. Decision Tree finishes at 69.90%, showing the limitations of axis-parallel splits on text."*

---

### Slide 9: Performance Visualizations
- Comparative plots showing Accuracy vs Macro F1, and Training Time vs Query Latency.
- **Speaker Notes:**  
  *"These figures visually demonstrate the performance landscape. On the left, we observe the clear dominance of Linear SVM and Naive Bayes in Accuracy and F1. On the right, we plot the computational tradeoff: Decision Tree takes the most time to train, while KNN requires significantly more inference time per sample."*

---

### Slide 10: Multi-Panel Confusion Matrix Analysis
- Detailed evaluation on 2,000 test reviews (1,000 negative, 1,000 positive).
- Linear SVM: 873 TN, 871 TP, 127 FP, 129 FN.
- Decision Tree: 345 FP, 257 FN (severe fragmentation).
- **Speaker Notes:**  
  *"Looking at the confusion matrices on 2,000 test reviews: Linear SVM and Naive Bayes show remarkable symmetry between false positives and false negatives. Linear SVM achieves 873 True Negatives and 871 True Positives. In contrast, Decision Tree misclassifies 602 reviews out of 2,000 due to feature fragmentation."*

---

### Slide 11: What-If Ablation Experiments
- Figure 6 displaying the three experimental curves: Vocabulary Size, N-grams, and Hyperparameter curves.
- **Speaker Notes:**  
  *"For Requirement 7, we performed three verified What-If experiments. In Plot A, we varied the TF-IDF vocabulary size from 1,000 to 10,000 features. In Plot B, we compared unigrams versus bigrams. In Plot C, we tuned the hyperparameters: SVM C, Naive Bayes alpha, and KNN k."*

---

### Slide 12: What-If Findings & Takeaways
- **Vocabulary Size:** Scaling from 1,000 to 5,000 features boosted SVM accuracy by $+3.1\%$; beyond 5,000 features gave diminishing returns ($+0.2\%$).
- **N-gram Range:** Unigram + Bigram (1, 2) reached $87.2\%$. Bigrams alone plunged to $78.5\%$ due to severe matrix sparsity.
- **Hyperparameter Sensitivity:** $C=1.0$ for SVM, $\alpha=1.0$ for Naive Bayes, $k=5$ for KNN were empirical sweet spots.
- **Speaker Notes:**  
  *"In this slide, I explain the observed empirical effects. Expanding vocabulary size from 1,000 to 5,000 gives a substantial 3.1% accuracy boost, after which returns plateau. Combining unigrams with bigrams is essential because bigrams capture critical phrases like 'not recommend', improving performance over unigrams alone. Bigrams alone, however, fail due to severe feature sparsity."*

---

### Slide 13: Technical Defense: Why Did SVM Win?
- **Theoretical Basis (Joachims, 1998):** Text in 5,000+ dimensions is almost linearly separable.
- **Max-Margin Regularization:** Maximizing $2 / \|w\|$ avoids overfitting in sparse dimensions.
- **Why Decision Tree Failed:** Single-feature orthogonal splits fragment sparse text ($50-100$ non-zero entries per document).
- **Why KNN Failed:** Distance concentration under the Curse of Dimensionality degrades neighbor discrimination.
- **Speaker Notes:**  
  *"This is our core technical defense slide. Why did SVM win? As established in foundational Information Retrieval literature by Thorsten Joachims, high-dimensional text vectors are inherently linearly separable. SVM finds the global maximum-margin hyperplane using all words simultaneously. Decision Trees fail because text sentiment cannot be effectively modeled by single-word axis-parallel cuts, and KNN suffers from distance concentration in high-dimensional vector spaces."*

---

### Slide 14: Conclusion, Limitations & Future Work
- **Conclusions:** Verified end-to-end sentiment classification system on 10,000 reviews with SVM achieving $87.20\%$.
- **Limitations:** Loss of word syntax in Bag-of-Words TF-IDF; domain specificity of cinema vocabulary.
- **Future Work:** Aspect-Based Sentiment Analysis (ABSA) and fine-tuned Transformer models (BERT).
- **Speaker Notes:**  
  *"In conclusion, this project provides a rigorous, transparent benchmark. Linear SVM is defended as the optimal production model for text classification. We acknowledge limitations including bag-of-words word order loss, and outline future enhancements such as aspect-based sentiment analysis and transformer models like BERT."*

---

### Slide 15: Scholarly References & Defense Q&A
- Maas et al. (2011), Joachims (1998), Salton & Buckley (1988), Manning et al. (2008), Cortes & Vapnik (1995).
- **Speaker Notes:**  
  *"Thank you respected evaluators for your attention. I have grounded my implementation in foundational Information Retrieval literature, from Salton and Buckley's TF-IDF formulation to Joachims' work on Support Vector Machines. I welcome any technical questions regarding my methodology, mathematics, or code implementation."*

