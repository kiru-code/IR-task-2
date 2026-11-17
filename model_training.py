"""
Model Training Module for Product Review Sentiment Classification
Author: Kirubaanithi R (Reg No: RA2411026050203)
Course: 21CSE377T - Information Retrieval
SRM Institute of Science and Technology, Tiruchirappalli
"""

import os
import time
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix


def train_and_evaluate_models(
    data_path: str = 'data/processed/imdb_cleaned_subset.csv',
    models_dir: str = 'models',
    results_dir: str = 'results',
    random_state: int = 42
):
    """
    Trains and compares four fundamental classification algorithms:
    1. Multinomial Naive Bayes
    2. Support Vector Machine (LinearSVC)
    3. Decision Tree
    4. K-Nearest Neighbours (KNN)

    Prevents data leakage by fitting TF-IDF strictly on the training set using Pipelines.
    Saves trained models, vectorizer, and quantitative evaluation metrics.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print(f"Loading preprocessed dataset from {data_path}...")
    df = pd.read_csv(data_path)
    # Fill any null values if present
    df['cleaned_review'] = df['cleaned_review'].fillna("")

    X = df['cleaned_review'].tolist()
    y = df['sentiment'].tolist()

    # Stratified 80-20 Train-Test Split (reproducible seed)
    print("Partitioning data into Stratified Train (80%) and Test (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print(f"Train samples: {len(X_train)} (Balanced: {(y_train == 'positive').sum()} pos, {(y_train == 'negative').sum()} neg)")
    print(f"Test samples:  {len(X_test)} (Balanced: {(y_test == 'positive').sum()} pos, {(y_test == 'negative').sum()} neg)")

    # Save test set for independent testing and evaluation scripts
    test_split_df = pd.DataFrame({'cleaned_review': X_test, 'sentiment': y_test})
    test_split_df.to_csv(os.path.join(results_dir, 'test_split.csv'), index=False)

    # Standard TF-IDF configuration
    # sublinear_tf=True scales term frequency logarithmically: 1 + log(tf)
    tfidf_config = {
        'max_features': 5000,
        'ngram_range': (1, 2),
        'sublinear_tf': True,
        'min_df': 2
    }

    # Define the Four Required Classifiers
    classifiers = {
        'Multinomial Naive Bayes': MultinomialNB(alpha=1.0),
        'Support Vector Machine (Linear)': LinearSVC(C=1.0, random_state=random_state, max_iter=3000),
        'Decision Tree': DecisionTreeClassifier(criterion='gini', max_depth=25, random_state=random_state),
        'K-Nearest Neighbours': KNeighborsClassifier(n_neighbors=5, metric='cosine', n_jobs=-1)
    }

    # Fit standalone TF-IDF Vectorizer on training data for direct feature analysis
    print("\nFitting standalone TF-IDF Vectorizer on training data...")
    vectorizer = TfidfVectorizer(**tfidf_config)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    joblib.dump(vectorizer, os.path.join(models_dir, 'tfidf_vectorizer.joblib'))
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)} features")

    results = []
    detailed_metrics = {}
    saved_pipelines = {}

    print("\n" + "="*80)
    print(f"{'CLASSIFIER BATTLE: TRAINING & FAIR EVALUATION':^80}")
    print("="*80)

    for name, clf in classifiers.items():
        print(f"\nTraining [{name}]...")
        
        # Build end-to-end Pipeline ensuring strict data hygiene
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(**tfidf_config)),
            ('classifier', clf)
        ])

        # Measure Training Time
        start_train = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start_train

        # Measure Inference Time on full test set
        start_test = time.time()
        y_pred = pipeline.predict(X_test)
        inference_time = time.time() - start_test
        avg_latency_ms = (inference_time / len(X_test)) * 1000

        # Calculate Evaluation Metrics
        acc = accuracy_score(y_test, y_pred)
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test, y_pred, average='macro', zero_division=0
        )
        prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        prec_pos, rec_pos, f1_pos, _ = precision_recall_fscore_support(
            y_test, y_pred, average=None, labels=['negative', 'positive'], zero_division=0
        )

        cm = confusion_matrix(y_test, y_pred, labels=['negative', 'positive'])
        # cm structure: [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = cm.ravel()

        model_slug = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        pipeline_path = os.path.join(models_dir, f"pipeline_{model_slug}.joblib")
        joblib.dump(pipeline, pipeline_path)
        saved_pipelines[name] = pipeline_path

        # Also save standalone trained classifier
        raw_clf_path = os.path.join(models_dir, f"clf_{model_slug}.joblib")
        # Fit on transformed vectors directly for fast model inspection
        clf.fit(X_train_vec, y_train)
        joblib.dump(clf, raw_clf_path)

        # Store summary row
        results.append({
            'Model': name,
            'Accuracy': round(acc, 4),
            'Precision (Macro)': round(prec_macro, 4),
            'Recall (Macro)': round(rec_macro, 4),
            'F1-Score (Macro)': round(f1_macro, 4),
            'Precision (Weighted)': round(prec_weighted, 4),
            'Recall (Weighted)': round(rec_weighted, 4),
            'F1-Score (Weighted)': round(f1_weighted, 4),
            'Training Time (s)': round(train_time, 4),
            'Test Latency (ms/sample)': round(avg_latency_ms, 3),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'TP': int(tp)
        })

        detailed_metrics[name] = {
            'accuracy': float(acc),
            'precision_macro': float(prec_macro),
            'recall_macro': float(rec_macro),
            'f1_macro': float(f1_macro),
            'precision_weighted': float(prec_weighted),
            'recall_weighted': float(rec_weighted),
            'f1_weighted': float(f1_weighted),
            'class_metrics': {
                'negative': {
                    'precision': float(prec_pos[0]),
                    'recall': float(rec_pos[0]),
                    'f1': float(f1_pos[0])
                },
                'positive': {
                    'precision': float(prec_pos[1]),
                    'recall': float(rec_pos[1]),
                    'f1': float(f1_pos[1])
                }
            },
            'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]],
            'training_time_seconds': float(train_time),
            'inference_time_seconds': float(inference_time),
            'latency_ms_per_sample': float(avg_latency_ms)
        }

        print(f"  Accuracy:  {acc*100:.2f}% | Macro F1: {f1_macro*100:.2f}%")
        print(f"  Training Time: {train_time:.3f}s | Latency: {avg_latency_ms:.3f} ms/query")
        print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # Create Comparison DataFrame and save
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values(by='Accuracy', ascending=False)
    comparison_csv = os.path.join(results_dir, 'model_comparison.csv')
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"\nModel comparison table saved to {comparison_csv}")

    # Save detailed JSON metrics
    metrics_json = os.path.join(results_dir, 'model_metrics.json')
    with open(metrics_json, 'w') as f:
        json.dump(detailed_metrics, f, indent=4)
    print(f"Detailed metrics saved to {metrics_json}")

    print("\n" + "="*80)
    print("FINAL MODEL COMPARISON SUMMARY:")
    print("="*80)
    print(comparison_df.to_string(index=False))

    return comparison_df, detailed_metrics


if __name__ == '__main__':
    train_and_evaluate_models()
