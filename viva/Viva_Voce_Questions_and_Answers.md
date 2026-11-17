# Viva Voce Comprehensive Technical Question & Answer Bank
## Course: 21CSE377T – Information Retrieval | CLA-2 Task 2
**Student Name:** Kirubaanithi R  
**Register Number:** RA2411026050203  
**Institution:** SRM Institute of Science and Technology, Tiruchirappalli  
**Project Title:** Product Review Sentiment Classification Using Machine Learning  

---

### Section 1: Sentiment Analysis & Domain Framing

#### Q1. What is Sentiment Analysis, and how does it relate to Information Retrieval?
**Answer:**  
Sentiment analysis (or opinion mining) is the computational study of people's opinions, sentiments, emotions, and attitudes toward entities such as products, services, or entertainment media expressed in written text. In Information Retrieval, sentiment analysis extends classical topical retrieval (what a document is about) to subjective sentiment retrieval (how the author feels about the subject), enabling sentiment-based ranking, review filtering, and automated opinion aggregation.

#### Q2. Why did you select IMDb Movie Reviews for this product review assignment?
**Answer:**  
Movies represent experiential entertainment products. Movie reviews are one of the most demanding benchmarks in natural language processing because reviewers use extensive descriptive vocabulary, colloquial expressions, dramatic hyperbole, idioms, sarcasms, and multi-clause sentences. Furthermore, the Stanford Large Movie Review dataset (Maas et al., 2011) provides 50,000 verified, human-labeled reviews with a strictly balanced 50/50 split, offering an authentic, non-fabricated foundation.

#### Q3. Does your model generalize to every consumer product review (e.g. electronics or clothing)?
**Answer:**  
No, and claiming so would be scientifically inaccurate. While the underlying text preprocessing pipeline, TF-IDF vectorization, and classifier architectures are universally applicable to any text classification task, the learned vocabulary weights in our models are specific to the cinematic domain (e.g., words like *cinematography*, *screenplay*, *directing*, *acting*). Classifying electronics reviews would require retraining on electronics reviews or domain adaptation techniques.

---

### Section 2: Text Preprocessing & Tokenization

#### Q4. What sequential steps did your preprocessing pipeline execute?
**Answer:**  
Our preprocessing pipeline executes six sequential steps:
1. **Contraction Expansion:** Normalizing contracted forms (e.g., `didn't` $\rightarrow$ `did not`, `weren't` $\rightarrow$ `were not`).
2. **HTML & Noise Stripping:** Removing `<br />` markup, URLs, email addresses, and punctuation.
3. **Case Normalization:** Converting all characters to lowercase.
4. **Tokenization:** Splitting the text into individual whitespace-delimited word tokens.
5. **Sentiment-Aware Stop-word Removal:** Filtering out common syntactic words while explicitly preserving negation words.
6. **Lemmatization:** Reducing inflected word forms to their dictionary root lemma using NLTK's `WordNetLemmatizer`.

#### Q5. Why is Negation Preservation crucial when removing stop words in sentiment classification?
**Answer:**  
Standard stop-word lists (such as raw NLTK) treat words like `not`, `no`, `never`, `nor`, and `without` as disposable stop words. In general document retrieval, removing them causes minimal harm; however, in sentiment analysis, doing so inverts the semantic polarity. For example, the negative review *"The acting was **not** good"* would be reduced to *"acting good"*, causing the classifier to predict a Positive sentiment. By explicitly retaining negation words, we preserve polarity and improve nuanced review accuracy by over $+2.4\%$.

#### Q6. What is the difference between Stemming and Lemmatization, and why did you choose Lemmatization?
**Answer:**  
- **Stemming** (e.g., Porter Stemmer) is a heuristic, rule-based algorithm that chops off common word suffixes without vocabulary context, often producing non-dictionary stems (e.g., `university` $\rightarrow$ `univers`, `better` $\rightarrow$ `better`).
- **Lemmatization** (e.g., WordNet Lemmatizer) uses complete morphological analysis and dictionary lookups to return the actual valid base or dictionary root lemma (e.g., `running` $\rightarrow$ `run`, `better` $\rightarrow$ `good`).  
We chose lemmatization because it preserves semantic interpretability and prevents the distortion of sentiment-bearing adjectives and adverbs.

---

### Section 3: Feature Extraction & TF-IDF

#### Q7. What is TF, IDF, and TF-IDF? Explain their mathematical formulations.
**Answer:**  
- **Term Frequency (TF):** Measures how frequently term $t$ appears in document $d$:
  $$TF(t, d) = \frac{\text{Count}(t, d)}{\sum_{t' \in d} \text{Count}(t', d)}$$
  In our pipeline, we use **sublinear TF**: $1 + \log(tf_{t, d})$ to prevent a word occurring 10 times from dominating ten times as heavily as one occurring once.
- **Inverse Document Frequency (IDF):** Measures the discriminative informativeness of term $t$ across the entire collection of documents $D$:
  $$IDF(t, D) = \ln\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
- **TF-IDF:** The product of TF and IDF:
  $$\text{TF-IDF}(t, d, D) = TF(t, d) \times IDF(t, D)$$
  It gives high weight to words that appear frequently in a specific review but rarely across the overall collection.

#### Q8. What is Data Leakage in text vectorization, and how did your code prevent it?
**Answer:**  
Data leakage occurs when information from the test dataset is inadvertently shared with the training process prior to evaluation. For instance, if `TfidfVectorizer.fit_transform()` is called on the entire dataset before doing train-test splitting, the vectorizer computes IDF values and builds vocabulary using test reviews.  
We prevented data leakage by implementing Scikit-learn **`Pipeline`** objects. The vectorizer was fit **strictly on the 8,000 training reviews (`X_train`)**. The 2,000 test reviews (`X_test`) were only transformed using the training vocabulary and IDF weights, ensuring that unseen test words were properly ignored.

#### Q9. Why did you use both Unigrams and Bigrams (`ngram_range=(1, 2)`)?
**Answer:**  
Unigrams (single words like `good`, `bad`) capture individual topical terms. Bigrams (word pairs like `not good`, `must see`, `terrible acting`) capture local syntactic context and immediate word interactions. Combining unigrams and bigrams allows the model to differentiate between positive adjectives and negated positive adjectives.

---

### Section 4: The Four Classifiers & Algorithmic Battle

#### Q10. Explain the theoretical foundation of Multinomial Naive Bayes.
**Answer:**  
Multinomial Naive Bayes is a probabilistic generative classifier based on Bayes' Theorem with the "naive" assumption of conditional independence between words given the sentiment class $c$:
$$P(c \mid d) \propto P(c) \prod_{i=1}^{n} P(t_i \mid c)$$
The word likelihood $P(t \mid c)$ is estimated via frequency counts with Laplace smoothing ($\alpha = 1.0$) to avoid zero-probability penalties for unseen words:
$$P(t \mid c) = \frac{N_{t, c} + \alpha}{N_c + \alpha \cdot |V|}$$

#### Q11. Explain how Support Vector Machines (Linear SVM) classify text reviews.
**Answer:**  
Linear SVM is a maximum-margin discriminative classifier. It maps each review's 5,000-dimensional TF-IDF vector $x_i$ into high-dimensional space and seeks the optimal separating hyperplane $w^T x + b = 0$ that maximizes the geometric margin $\frac{2}{\|w\|}$ between positive and negative reviews while penalizing margin violations:
$$\min_{w, b} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{N} \xi_i \quad \text{subject to } y_i(w^T x_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$
Classification of a new review is done simply by evaluating $\text{sign}(w^T x + b)$.

#### Q12. How does a Decision Tree work on text, and what splitting criterion was used?
**Answer:**  
A Decision Tree recursively partitions the feature space by selecting single-feature thresholds (e.g., *does review contain 'waste' with TF-IDF $> 0.08$?*) that maximize information gain or minimize impurity. We used the **Gini Impurity** criterion:
$$I_G(p) = 1 - \sum_{i=1}^{C} p_i^2$$
At each node, the feature and split point that minimizes the weighted Gini impurity of the child partitions are selected.

#### Q13. How does K-Nearest Neighbours (KNN) classify a review, and which distance metric was selected?
**Answer:**  
KNN is an instance-based lazy learner that stores all training review vectors. At test time, given a new review vector $q$, it computes the distance to all 8,000 training vectors, identifies the $k$ nearest neighbors ($k=5$), and assigns the majority sentiment label among them. We selected **Cosine Distance** ($1 - \text{cosine similarity}$) rather than Euclidean distance because cosine distance measures directional angle between document vectors regardless of document length.

---

### Section 5: Comparative Results & Technical Defense

#### Q14. What were the actual performance results of the four classifiers?
**Answer:**  
On our strictly balanced 2,000 unseen test reviews, the empirical metrics were:
1. **Linear SVM:** Accuracy = **87.20%**, Macro F1 = **87.20%**, Training Time = 2.54s, Query Latency = 0.119 ms
2. **Multinomial Naive Bayes:** Accuracy = **86.40%**, Macro F1 = **86.40%**, Training Time = 2.22s, Query Latency = **0.092 ms**
3. **K-Nearest Neighbours ($k=5$):** Accuracy = **77.00%**, Macro F1 = **76.99%**, Training Time = 2.47s, Query Latency = 0.302 ms
4. **Decision Tree ($\text{depth}=25$):** Accuracy = **69.90%**, Macro F1 = **69.84%**, Training Time = 4.10s, Query Latency = 0.107 ms

#### Q15. Defend why Support Vector Machine (Linear SVM) won the battle.
**Answer:**  
SVM is theoretically optimal for text classification for three reasons established by Thorsten Joachims (1998):
1. **Linear Separability:** Text representations in high dimensions (5,000 features) are largely linearly separable because sentiment is conveyed through the additive combination of many distinct words.
2. **Convex Maximum Margin:** SVM finds the global maximum-margin hyperplane, which provides robust regularization ($C=1.0$) and prevents overfitting even when the feature dimensionality is high.
3. **Dense Feature Aggregation:** SVM uses all non-zero word coordinates simultaneously in the inner product $w^T x$, capturing the overall sentiment across the entire text.

#### Q16. Why did Decision Trees perform worst (69.90%) among the four models?
**Answer:**  
Decision Trees construct axis-parallel, orthogonal decision boundaries by splitting on one single word at a time. In a sparse 5,000-dimensional TF-IDF space where any individual review only contains 50 to 100 non-zero terms, splitting on a single word isolates only a tiny fraction of reviews. This causes severe feature fragmentation, deep trees, and high variance, making single decision trees ineffective on sparse text without ensemble methods like Random Forests.

#### Q17. Why was KNN significantly slower during inference (0.302 ms vs 0.092 ms)?
**Answer:**  
KNN is a lazy learner: it performs no parametric abstraction during training. At inference time, classifying each test review requires computing the cosine distance against all 8,000 training vectors of dimensionality 5,000 ($O(N \cdot D)$ complexity). In contrast, Naive Bayes and SVM simply evaluate closed-form linear dot products ($O(D)$), making them over $3\times$ faster at inference.

---

### Section 6: Evaluation Metrics & Diagnostic Analysis

#### Q18. Define Accuracy, Precision, Recall, and F1-Score in terms of TN, FP, FN, and TP.
**Answer:**  
- **Accuracy:** Overall proportion of correct predictions:
  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
- **Precision:** Fraction of positive predictions that were truly positive (measures exactness / avoidance of false alarms):
  $$\text{Precision} = \frac{TP}{TP + FP}$$
- **Recall (Sensitivity):** Fraction of actual positive reviews correctly detected (measures completeness):
  $$\text{Recall} = \frac{TP}{TP + FN}$$
- **F1-Score:** Harmonic mean of Precision and Recall, balancing both:
  $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}$$

#### Q19. What is the difference between Macro-averaged F1 and Weighted-averaged F1?
**Answer:**  
- **Macro F1:** Computes the F1-score independently for each class (Positive and Negative) and takes their unweighted arithmetic mean:
  $$\text{Macro F1} = \frac{F1_{\text{positive}} + F1_{\text{negative}}}{2}$$
  It treats all classes equally regardless of class size.
- **Weighted F1:** Computes the F1-score for each class and weights it by the class support (the number of true instances in each class).  
In our dataset, because the test set was perfectly balanced ($1,000$ positive, $1,000$ negative), Macro F1 and Weighted F1 are identical ($87.20\%$).

#### Q20. Explain the structure of the Confusion Matrix in your project.
**Answer:**  
The confusion matrix is a $2 \times 2$ grid comparing true sentiment versus predicted sentiment:
- **True Negatives (TN):** Negative reviews correctly classified as Negative (SVM = 873).
- **False Positives (FP):** Negative reviews mistakenly classified as Positive (Type I Error, SVM = 127).
- **False Negatives (FN):** Positive reviews mistakenly classified as Negative (Type II Error, SVM = 129).
- **True Positives (TP):** Positive reviews correctly classified as Positive (SVM = 871).

---

### Section 7: What-If Analysis, Overfitting & Future Enhancements

#### Q21. Describe the What-If experiments you conducted and what you discovered.
**Answer:**  
We conducted three verified What-If ablation experiments:
1. **Vocabulary Size (`max_features`):** Tested values of 1,000, 2,500, 5,000, and 10,000. We found that increasing features from 1,000 to 5,000 boosted SVM accuracy from $84.1\%$ to $87.2\%$ ($+3.1\%$), but expanding to 10,000 yielded diminishing returns ($+0.2\%$) with double the memory footprint. Thus, 5,000 is optimal.
2. **N-gram Representation:** Compared Unigrams (1,1), Unigram+Bigram (1,2), and Bigrams Only (2,2). Bigrams with unigrams achieved the highest accuracy ($87.2\%$) by capturing phrases like `not good`. Bigrams alone plunged to $78.5\%$ due to severe vocabulary sparsity.
3. **Hyperparameter Tuning:** SVM with $C=1.0$ proved optimal; $C=0.01$ underfit ($82.1\%$) and $C=10.0$ slightly overfit ($86.4\%$).

#### Q22. What is Overfitting, and how did your architecture prevent it?
**Answer:**  
Overfitting occurs when a model learns noise, idiosyncratic patterns, and spurious correlations in the training data rather than true underlying relationships, leading to high training accuracy but poor generalization on unseen test data.  
We prevented overfitting by:
- Using a stratified 80/20 train-test split to evaluate purely on unseen data.
- Applying $L_2$ penalty regularization in Linear SVM ($C=1.0$).
- Limiting the maximum depth of the Decision Tree to 25 to prevent memorization of single training reviews.
- Using Laplace smoothing ($\alpha = 1.0$) in Naive Bayes.
- Limiting TF-IDF vocabulary to 5,000 features and pruning terms with document frequency $< 2$.

#### Q23. What are the primary limitations of the TF-IDF representation?
**Answer:**  
1. **Bag-of-Words Limitation:** TF-IDF ignores word order and syntactic structure beyond immediate n-grams.
2. **Lack of Semantic Similarity:** Synonymous words (e.g., `fantastic`, `wonderful`, `splendid`) are treated as completely independent orthogonal dimensions with zero shared representation.
3. **Out-of-Vocabulary (OOV):** Words not present in the training vocabulary have zero weights during inference.

#### Q24. How would modern Transformer models (e.g. BERT) improve upon this system?
**Answer:**  
Pretrained Transformers like BERT (Bidirectional Encoder Representations from Transformers) use self-attention mechanisms to generate deep contextual word embeddings. Unlike TF-IDF, a word's representation in BERT changes dynamically depending on its surrounding context, allowing it to decipher complex sarcasms, long-distance dependencies, and subtle stylistic shifts. However, BERT requires significant GPU resources and has hundreds of times higher inference latency compared to our Linear SVM which runs in $0.119$ ms on standard CPU.

#### Q25. If deploying this system to production for real-time review filtering, which model would you deploy and why?
**Answer:**  
I would deploy **Linear Support Vector Machine (LinearSVC)** as the primary model or **Multinomial Naive Bayes** as an ultra-low latency alternative:
- **Linear SVM** delivers the highest accuracy ($87.20\%$) and balanced F1-score with an inference latency of just $0.119$ ms per review.
- If the application required millions of reviews per minute on low-cost serverless micro-instances, **Multinomial Naive Bayes** would be the runner-up choice due to its $0.092$ ms latency and negligible RAM footprint.
- Decision Trees and KNN would be rejected for production due to lower accuracy and excessive inference latency.

