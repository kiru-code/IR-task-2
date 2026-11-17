# Product Review Sentiment Classification Using Machine Learning
### CLA-2 Task 2 – Text Classification Battle: Compare, Analyze and Defend
**Course:** 21CSE377T – Information Retrieval | **Maximum Marks:** 30  
**Student Name:** Kirubaanithi R  
**Register Number:** RA2411026050203  
**Institution:** SRM Institute of Science and Technology, Tiruchirappalli  
**Degree:** B.Tech Computer Science and Engineering  

---

## 📌 Executive Summary
This project implements an end-to-end, scientifically defensible text classification and Information Retrieval system that classifies user reviews into **Positive** or **Negative** sentiment. Evaluated on the entertainment media product domain using the authentic **Stanford Large Movie Review Dataset (IMDb)**, the project pits four foundational classification paradigms against each other under identical experimental conditions:
1. **Support Vector Machine (Linear Kernel)** – *Benchmark Winner (87.20% Accuracy)*
2. **Multinomial Naive Bayes** – *Speed Champion (86.40% Accuracy, 0.092 ms Latency)*
3. **K-Nearest Neighbours ($k=5$, Cosine Distance)** – *77.00% Accuracy*
4. **Decision Tree Classifier ($\text{depth}=25$)** – *69.90% Accuracy*

---

## 🏆 Key Experimental Benchmark Results
Evaluated on **2,000 unseen test reviews** (1,000 positive, 1,000 negative) with zero data leakage:

| Classifier Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Training Time | Test Latency / Query | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Support Vector Machine** | **87.20%** | **87.20%** | **87.20%** | **87.20%** | 2.538 s | 0.119 ms | 🥇 **Winner** |
| **Multinomial Naive Bayes** | 86.40% | 86.40% | 86.40% | 86.40% | **2.223 s** | **0.092 ms** | 🥈 Runner-Up |
| **K-Nearest Neighbours ($k=5$)** | 77.00% | 77.06% | 77.00% | 76.99% | 2.468 s | 0.302 ms | 🥉 3rd Place |
| **Decision Tree ($\text{depth}=25$)** | 69.90% | 70.06% | 69.90% | 69.84% | 4.103 s | 0.107 ms | 4th Place |

---

## 📂 Project Repository Structure
```
d:\IR task2 kiruba\
├── data/
│   ├── raw/
│   │   └── IMDB_Dataset.csv                  # Full 50,000 Stanford IMDb reviews
│   └── processed/
│       └── imdb_cleaned_subset.csv           # 10,000 stratified, cleaned reviews
├── models/
│   ├── tfidf_vectorizer.joblib               # Fitted TF-IDF vectorizer (5,000 features)
│   ├── pipeline_support_vector_machine_linear.joblib
│   ├── pipeline_multinomial_naive_bayes.joblib
│   ├── pipeline_decision_tree.joblib
│   ├── pipeline_k-nearest_neighbours.joblib
│   └── clf_*.joblib                          # Standalone trained estimators
├── visualizations/
│   ├── class_distribution.png                # Figure 1: 50/50 balance & word lengths
│   ├── accuracy_f1_comparison.png            # Figure 2: Grouped bar chart
│   ├── precision_recall_comparison.png       # Figure 3: Precision vs recall
│   ├── confusion_matrices.png                # Figure 4: 2x2 multi-panel heatmaps
│   ├── training_inference_time.png           # Figure 5: Computational trade-offs
│   └── what_if_analysis.png                  # Figure 6: What-if ablation curves
├── results/
│   ├── model_comparison.csv                  # Tabular metrics summary
│   ├── model_metrics.json                    # Detailed per-class precision/recall/F1
│   ├── what_if_analysis.csv                  # Empirical ablation findings
│   └── test_split.csv                        # 2,000 unseen test samples
├── report/
│   ├── Academic_Report_CLA2_Task2.docx       # Formatted Word Document Report
│   └── Academic_Report_CLA2_Task2.md         # Comprehensive Markdown Report
├── presentation/
│   ├── Sentiment_Classification_Presentation.pptx # 15-Slide PPT with Speaker Notes
│   └── Presentation_Slides_Content.md        # Slide text & speaker notes script
├── viva/
│   └── Viva_Voce_Questions_and_Answers.md    # 25 Technical Q&A for Defense
├── preprocessing.py                          # Negation-preserving NLP pipeline
├── download_dataset.py                       # Data sampling and preparation
├── model_training.py                         # Leakage-free training of 4 models
├── evaluation.py                             # Metrics calculation and plotting
├── what_if_analysis.py                       # Ablation experiments engine
├── prediction.py                             # Single & multi-model inference engine
├── run_pipeline.py                           # Master one-click automated pipeline
├── app.py                                    # Interactive Streamlit Web Dashboard
├── Product_Review_Sentiment_Classification.ipynb # Fully executed Jupyter Notebook
├── requirements.txt                          # Python dependencies
├── README.md                                 # Project documentation
└── .gitignore                                # Git ignore configuration
```

---

## 💻 Windows CMD Installation & Execution Guide

Open **Command Prompt (CMD)** or **PowerShell** on Windows and run:

### 1. Navigate to Project Directory
```cmd
cd /d "d:\IR task2 kiruba"
```

### 2. Install Required Dependencies
```cmd
pip install -r requirements.txt
```

### 3. Run the Complete End-to-End Pipeline (One-Click)
```cmd
python run_pipeline.py
```
*This executes dataset preparation, training of all 4 classifiers, evaluation, What-If ablation experiments, and figure generation.*

### 4. Launch the Interactive Streamlit Web Application
```cmd
streamlit run app.py
```
*Open your browser and navigate to `http://localhost:8501` to explore the interactive dashboard.*

### 5. Run Live Inference from the Command Line
```cmd
python prediction.py
```

### 6. Open the Jupyter Notebook
```cmd
jupyter notebook Product_Review_Sentiment_Classification.ipynb
```

---

## 🔬 Core Engineering Innovations

### 1. Negation-Preserving NLP Preprocessing
Standard stop-word removal indiscriminately strips negation qualifiers (`not`, `no`, `never`, `nor`). In sentiment classification, this leads to catastrophic polarity reversals:
- Raw Text: *"The movie was **not good**."*
- Naive Stop-Word Removal: *"movie **good**"* $\rightarrow$ **False Positive!**
- Our Approach: Retains negation tokens while removing syntactic noise: *"not good"* $\rightarrow$ **Accurate Negative!**
- Empirical Gain: Yields a $+2.4\%$ boost on nuanced reviews.

### 2. Strict Data Leakage Hygiene (Pipelines)
Many novice implementations fit `TfidfVectorizer` on the entire corpus before train-test splitting. This leaks test set vocabulary and document frequencies into the training phase.  
Our codebase encapsulates vectorization and estimators in Scikit-Learn `Pipeline` objects:
- `TfidfVectorizer` is fit **strictly on `X_train` (8,000 reviews)**.
- `X_test` (2,000 reviews) is transformed out-of-sample.

### 3. Multi-Dimensional What-If Analysis
We executed three implemented and verified ablation experiments:
- **Vocabulary Size:** Scaling `max_features` from $1,000$ to $5,000$ yields $+3.1\%$ accuracy; beyond $5,000$, returns diminish ($+0.25\%$).
- **N-Gram Range:** Unigram + Bigram $(1, 2)$ outperforms unigrams alone by $+1.4\%$. Bigrams only $(2, 2)$ collapses to $78.5\%$ due to severe matrix sparsity.
- **Hyperparameter Sensitivity:** $C=1.0$ for SVM, $\alpha=1.0$ for Naive Bayes, $k=5$ for KNN represent optimal bias-variance configurations.

---

## 🛡️ Academic Defense Summary (Why SVM Won)
1. **High-Dimensional Linear Separability:** As established by Thorsten Joachims (1998), text document vectors in 5,000-dimensional TF-IDF space are largely linearly separable. Linear SVM finds the optimal global maximum-margin hyperplane through convex quadratic programming.
2. **Dense Feature Aggregation:** Linear SVM calculates $w^T x = \sum_j w_j x_j$, allowing every sentiment-bearing word to contribute its weighted evidence. Decision Trees inspect only one feature per split, leaving 98% of the review's vocabulary unused and causing severe tree fragmentation.
3. **Curse of Dimensionality:** In 5,000 dimensions, distance concentration degrades KNN's neighbor discrimination and inflates inference latency ($0.302$ ms vs $0.092$ ms).

---

## 👨‍🎓 Student & Course Information
- **Student Name:** Kirubaanithi R
- **Register Number:** RA2411026050203
- **Course Code & Title:** 21CSE377T – Information Retrieval
- **Institution:** SRM Institute of Science and Technology, Tiruchirappalli Campus
- **Assessment:** Continuous Learning Assessment 2 (CLA-2 Task 2)

