"""
What-If Analysis & Ablation Experiments Module
Author: Kirubaanithi R (Reg No: RA2411026050203)
Course: 21CSE377T - Information Retrieval
SRM Institute of Science and Technology, Tiruchirappalli
"""

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score


def run_what_if_experiments(
    data_path: str = 'data/processed/imdb_cleaned_subset.csv',
    results_dir: str = 'results',
    vis_dir: str = 'visualizations',
    random_state: int = 42
):
    """
    Conducts empirical What-If experiments:
    1. Feature Dimension Ablation: Varying TF-IDF max_features (1000, 2500, 5000, 10000)
    2. N-Gram Range Ablation: Unigrams (1,1) vs Unigrams+Bigrams (1,2) vs Bigrams only (2,2)
    3. Hyperparameter Sensitivity:
       - Naive Bayes smoothing parameter alpha
       - SVM regularization parameter C
       - KNN neighbor count k
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    print("Loading preprocessed dataset for What-If Analysis...")
    df = pd.read_csv(data_path)
    df['cleaned_review'] = df['cleaned_review'].fillna("")
    X = df['cleaned_review'].tolist()
    y = np.array(df['sentiment'].tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )

    # -------------------------------------------------------------
    # EXPERIMENT 1: Impact of TF-IDF Vocabulary Size (max_features)
    # -------------------------------------------------------------
    print("\n--- EXPERIMENT 1: TF-IDF Vocabulary Size Sensitivity ---")
    vocab_sizes = [1000, 2500, 5000, 10000]
    vocab_results = []

    for max_feat in vocab_sizes:
        print(f"Testing max_features = {max_feat}...")
        vec = TfidfVectorizer(max_features=max_feat, ngram_range=(1, 2), sublinear_tf=True)
        X_tr = vec.fit_transform(X_train)
        X_te = vec.transform(X_test)

        models = {
            'Multinomial Naive Bayes': MultinomialNB(alpha=1.0),
            'Linear SVM': LinearSVC(C=1.0, random_state=random_state, max_iter=3000),
            'Decision Tree': DecisionTreeClassifier(max_depth=25, random_state=random_state),
            'KNN (k=5)': KNeighborsClassifier(n_neighbors=5, metric='cosine', n_jobs=-1)
        }

        for model_name, clf in models.items():
            t0 = time.time()
            clf.fit(X_tr, y_train)
            train_time = time.time() - t0
            y_pred = clf.predict(X_te)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            vocab_results.append({
                'Experiment': 'Vocabulary Size',
                'Parameter': 'max_features',
                'Parameter_Value': str(max_feat),
                'Vocabulary_Size': max_feat,
                'Model': model_name,
                'Accuracy': round(acc, 4),
                'Macro_F1': round(f1, 4),
                'Training_Time_s': round(train_time, 4)
            })

    vocab_df = pd.DataFrame(vocab_results)

    # -------------------------------------------------------------
    # EXPERIMENT 2: Impact of N-Gram Representation
    # -------------------------------------------------------------
    print("\n--- EXPERIMENT 2: N-Gram Range Sensitivity ---")
    ngram_configs = [
        ('Unigrams (1,1)', (1, 1)),
        ('Unigram + Bigram (1,2)', (1, 2)),
        ('Bigrams Only (2,2)', (2, 2))
    ]
    ngram_results = []

    for label, ngram_range in ngram_configs:
        print(f"Testing N-gram configuration: {label}...")
        vec = TfidfVectorizer(max_features=5000, ngram_range=ngram_range, sublinear_tf=True)
        X_tr = vec.fit_transform(X_train)
        X_te = vec.transform(X_test)

        models = {
            'Multinomial Naive Bayes': MultinomialNB(alpha=1.0),
            'Linear SVM': LinearSVC(C=1.0, random_state=random_state, max_iter=3000),
            'Decision Tree': DecisionTreeClassifier(max_depth=25, random_state=random_state),
            'KNN (k=5)': KNeighborsClassifier(n_neighbors=5, metric='cosine', n_jobs=-1)
        }

        for model_name, clf in models.items():
            clf.fit(X_tr, y_train)
            y_pred = clf.predict(X_te)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            ngram_results.append({
                'Experiment': 'N-Gram Representation',
                'Parameter': 'ngram_range',
                'Parameter_Value': label,
                'Vocabulary_Size': 5000,
                'Model': model_name,
                'Accuracy': round(acc, 4),
                'Macro_F1': round(f1, 4),
                'Training_Time_s': 0.0
            })

    ngram_df = pd.DataFrame(ngram_results)

    # -------------------------------------------------------------
    # EXPERIMENT 3: Classifier Hyperparameter Sensitivity
    # -------------------------------------------------------------
    print("\n--- EXPERIMENT 3: Hyperparameter Sensitivity ---")
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    X_tr = vec.fit_transform(X_train)
    X_te = vec.transform(X_test)
    hparam_results = []

    # SVM C sensitivity
    for c_val in [0.01, 0.1, 1.0, 5.0, 10.0]:
        clf = LinearSVC(C=c_val, random_state=random_state, max_iter=3000)
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        hparam_results.append({
            'Experiment': 'Hyperparameter - SVM C',
            'Parameter': 'C',
            'Parameter_Value': str(c_val),
            'Vocabulary_Size': 5000,
            'Model': 'Linear SVM',
            'Accuracy': round(acc, 4),
            'Macro_F1': round(f1, 4),
            'Training_Time_s': 0.0
        })

    # Naive Bayes Alpha sensitivity
    for alpha_val in [0.01, 0.1, 0.5, 1.0, 2.0]:
        clf = MultinomialNB(alpha=alpha_val)
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        hparam_results.append({
            'Experiment': 'Hyperparameter - MNB Alpha',
            'Parameter': 'alpha',
            'Parameter_Value': str(alpha_val),
            'Vocabulary_Size': 5000,
            'Model': 'Multinomial Naive Bayes',
            'Accuracy': round(acc, 4),
            'Macro_F1': round(f1, 4),
            'Training_Time_s': 0.0
        })

    # KNN k sensitivity
    for k_val in [3, 5, 7, 11, 15]:
        clf = KNeighborsClassifier(n_neighbors=k_val, metric='cosine', n_jobs=-1)
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        hparam_results.append({
            'Experiment': 'Hyperparameter - KNN k',
            'Parameter': 'k',
            'Parameter_Value': str(k_val),
            'Vocabulary_Size': 5000,
            'Model': 'KNN',
            'Accuracy': round(acc, 4),
            'Macro_F1': round(f1, 4),
            'Training_Time_s': 0.0
        })

    hparam_df = pd.DataFrame(hparam_results)

    # Combine all experiment results
    all_what_if_df = pd.concat([vocab_df, ngram_df, hparam_df], ignore_index=True)
    what_if_csv = os.path.join(results_dir, 'what_if_analysis.csv')
    all_what_if_df.to_csv(what_if_csv, index=False)
    print(f"\nAll What-If experiments saved to {what_if_csv}")

    # -------------------------------------------------------------
    # GENERATE WHAT-IF VISUALIZATIONS
    # -------------------------------------------------------------
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Plot 1: Vocabulary Size
    sns.lineplot(
        data=vocab_df,
        x='Vocabulary_Size',
        y='Accuracy',
        hue='Model',
        marker='o',
        linewidth=2.5,
        markersize=8,
        ax=axes[0]
    )
    axes[0].set_title('A: Impact of TF-IDF Vocabulary Size (max_features)', fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xlabel('Max Features Vocabulary Size', fontsize=11)
    axes[0].set_ylabel('Test Accuracy', fontsize=11)
    axes[0].set_ylim(0.65, 0.90)

    # Plot 2: N-gram Variations
    sns.barplot(
        data=ngram_df,
        x='Parameter_Value',
        y='Accuracy',
        hue='Model',
        ax=axes[1]
    )
    axes[1].set_title('B: Impact of N-gram Feature Space', fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel('N-gram Configuration', fontsize=11)
    axes[1].set_ylabel('Test Accuracy', fontsize=11)
    axes[1].set_ylim(0.60, 0.90)
    axes[1].tick_params(axis='x', rotation=15)

    # Plot 3: Hyperparameter Sensitivities
    svm_hparam = hparam_df[hparam_df['Experiment'] == 'Hyperparameter - SVM C']
    mnb_hparam = hparam_df[hparam_df['Experiment'] == 'Hyperparameter - MNB Alpha']
    knn_hparam = hparam_df[hparam_df['Experiment'] == 'Hyperparameter - KNN k']

    axes[2].plot(svm_hparam['Parameter_Value'], svm_hparam['Accuracy'], marker='s', label='SVM (C)', color='#1f77b4', linewidth=2)
    axes[2].plot(mnb_hparam['Parameter_Value'], mnb_hparam['Accuracy'], marker='^', label='Naive Bayes (α)', color='#2ca02c', linewidth=2)
    axes[2].plot(knn_hparam['Parameter_Value'], knn_hparam['Accuracy'], marker='d', label='KNN (k)', color='#ff7f0e', linewidth=2)
    axes[2].set_title('C: Hyperparameter Tuning Curves', fontsize=12, fontweight='bold', pad=10)
    axes[2].set_xlabel('Hyperparameter Setting Value', fontsize=11)
    axes[2].set_ylabel('Test Accuracy', fontsize=11)
    axes[2].legend(loc='lower right')
    axes[2].set_ylim(0.70, 0.90)

    plt.tight_layout()
    plot_path = os.path.join(vis_dir, 'what_if_analysis.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"What-If visualization saved to {plot_path}")

    return all_what_if_df


if __name__ == '__main__':
    run_what_if_experiments()

