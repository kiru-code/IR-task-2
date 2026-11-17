"""
Inference and Sentiment Prediction Engine
Author: Kirubaanithi R (Reg No: RA2411026050203)
Course: 21CSE377T - Information Retrieval
SRM Institute of Science and Technology, Tiruchirappalli
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from preprocessing import preprocess_review, get_preprocessing_steps

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

# Map friendly model names to filenames
MODEL_FILES = {
    'Support Vector Machine': ('pipeline_support_vector_machine_linear.joblib', 'clf_support_vector_machine_linear.joblib'),
    'Multinomial Naive Bayes': ('pipeline_multinomial_naive_bayes.joblib', 'clf_multinomial_naive_bayes.joblib'),
    'K-Nearest Neighbours': ('pipeline_k-nearest_neighbours.joblib', 'clf_k-nearest_neighbours.joblib'),
    'Decision Tree': ('pipeline_decision_tree.joblib', 'clf_decision_tree.joblib')
}

_LOADED_PIPELINES = {}
_LOADED_VECTORIZER = None
_LOADED_MODELS = {}


def load_pipeline(model_name: str = 'Support Vector Machine'):
    """Lazy loads the pipeline for the given classifier."""
    global _LOADED_PIPELINES
    if model_name not in _LOADED_PIPELINES:
        if model_name not in MODEL_FILES:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_FILES.keys())}")
        pipe_path = os.path.join(MODELS_DIR, MODEL_FILES[model_name][0])
        if not os.path.exists(pipe_path):
            raise FileNotFoundError(f"Model file not found at {pipe_path}. Run model_training.py first.")
        _LOADED_PIPELINES[model_name] = joblib.load(pipe_path)
    return _LOADED_PIPELINES[model_name]


def load_vectorizer_and_model(model_name: str = 'Support Vector Machine'):
    """Loads standalone vectorizer and model for feature extraction."""
    global _LOADED_VECTORIZER, _LOADED_MODELS
    if _LOADED_VECTORIZER is None:
        vec_path = os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib')
        if not os.path.exists(vec_path):
            raise FileNotFoundError(f"Vectorizer not found at {vec_path}. Run model_training.py first.")
        _LOADED_VECTORIZER = joblib.load(vec_path)

    if model_name not in _LOADED_MODELS:
        clf_path = os.path.join(MODELS_DIR, MODEL_FILES[model_name][1])
        if not os.path.exists(clf_path):
            raise FileNotFoundError(f"Model not found at {clf_path}. Run model_training.py first.")
        _LOADED_MODELS[model_name] = joblib.load(clf_path)

    return _LOADED_VECTORIZER, _LOADED_MODELS[model_name]


def get_feature_contributions(cleaned_text: str, model_name: str = 'Support Vector Machine', top_n: int = 6):
    """
    Extracts the most influential words in the review based on TF-IDF weight
    and model weights (for SVM/Naive Bayes).
    """
    try:
        vec, clf = load_vectorizer_and_model(model_name)
        tokens = cleaned_text.split()
        if not tokens:
            return []

        feature_names = np.array(vec.get_feature_names_out())
        X_vec = vec.transform([cleaned_text])
        non_zero_indices = X_vec.nonzero()[1]

        if len(non_zero_indices) == 0:
            return []

        contributions = []
        for idx in non_zero_indices:
            term = feature_names[idx]
            tfidf_weight = float(X_vec[0, idx])

            if model_name == 'Support Vector Machine' and hasattr(clf, 'coef_'):
                score = float(clf.coef_[0, idx]) * tfidf_weight
                direction = "Positive" if score > 0 else "Negative"
            elif model_name == 'Multinomial Naive Bayes' and hasattr(clf, 'feature_log_prob_'):
                # Difference between positive log prob and negative log prob
                log_ratio = float(clf.feature_log_prob_[1, idx] - clf.feature_log_prob_[0, idx])
                score = log_ratio * tfidf_weight
                direction = "Positive" if score > 0 else "Negative"
            else:
                score = tfidf_weight
                direction = "Salient"

            contributions.append({
                'term': term,
                'weight': round(score, 4),
                'tfidf': round(tfidf_weight, 4),
                'polarity_influence': direction
            })

        contributions.sort(key=lambda x: abs(x['weight']), reverse=True)
        return contributions[:top_n]
    except Exception as e:
        return []


def predict_sentiment(raw_text: str, model_name: str = 'Support Vector Machine') -> dict:
    """
    Predicts sentiment (Positive or Negative) for any given movie/product review.
    Returns predicted label, probability/confidence, latency, and key features.
    """
    if not raw_text or not raw_text.strip():
        return {
            'error': 'Input text is empty.',
            'predicted_sentiment': None,
            'confidence': 0.0
        }

    t0 = time.time()
    steps = get_preprocessing_steps(raw_text)
    cleaned = steps['final_preprocessed_text']

    pipeline = load_pipeline(model_name)
    prediction = pipeline.predict([cleaned])[0]

    # Calculate confidence or decision score
    clf_step = pipeline.named_steps['classifier']
    tfidf_step = pipeline.named_steps['tfidf']
    X_sample = tfidf_step.transform([cleaned])

    confidence = None
    if hasattr(clf_step, 'predict_proba'):
        # For Naive Bayes, KNN, Decision Tree
        probs = clf_step.predict_proba(X_sample)[0]
        class_idx = list(clf_step.classes_).index(prediction)
        confidence = float(probs[class_idx])
    elif hasattr(clf_step, 'decision_function'):
        # For LinearSVC: sigmoid of decision distance
        dist = float(clf_step.decision_function(X_sample)[0])
        prob = 1.0 / (1.0 + np.exp(-dist))
        confidence = prob if prediction == 'positive' else (1.0 - prob)
    else:
        confidence = 0.85

    latency_ms = (time.time() - t0) * 1000
    top_features = get_feature_contributions(cleaned, model_name)

    return {
        'model_name': model_name,
        'raw_text': raw_text,
        'cleaned_text': cleaned,
        'predicted_sentiment': prediction.capitalize(),
        'confidence': round(confidence * 100, 2),
        'latency_ms': round(latency_ms, 2),
        'top_contributing_terms': top_features,
        'preprocessing_steps': steps
    }


def predict_all_models(raw_text: str) -> dict:
    """Runs prediction across all four classifiers simultaneously for direct battle comparison."""
    results = {}
    for model_name in MODEL_FILES.keys():
        results[model_name] = predict_sentiment(raw_text, model_name)
    return results


if __name__ == '__main__':
    sample_positive = "An absolute cinematic masterpiece! The cinematography was stunning, the acting was phenomenal, and the emotional climax brought tears to my eyes."
    sample_negative = "Do not waste your time on this movie. The acting was wooden, the storyline made no sense, and it was dreadfully boring from start to finish."
    sample_negation = "This was not a good movie at all, and I would never recommend it to anyone looking for quality entertainment."

    print("--- DEMONSTRATION OF LIVE PREDICTIONS ---")
    for sample in [sample_positive, sample_negative, sample_negation]:
        print("\n" + "="*70)
        print("INPUT:", sample[:80] + "...")
        preds = predict_all_models(sample)
        for m, res in preds.items():
            print(f"[{m}]: {res['predicted_sentiment']} (Confidence: {res['confidence']:.1f}%, Latency: {res['latency_ms']}ms)")

