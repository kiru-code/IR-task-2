"""
Comprehensive Evaluation and Visualization Module
Author: Kirubaanithi R (Reg No: RA2411026050203)
Course: 21CSE377T - Information Retrieval
SRM Institute of Science and Technology, Tiruchirappalli
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    classification_report, confusion_matrix
)


def generate_evaluation_visualizations(
    results_dir: str = 'results',
    vis_dir: str = 'visualizations',
    data_path: str = 'data/processed/imdb_cleaned_subset.csv',
    models_dir: str = 'models'
):
    """
    Produces publication-grade academic visualizations:
    1. Class Distribution & Review Lengths
    2. Accuracy & Macro F1 Comparative Bar Charts
    3. Precision & Recall Comparative Bar Charts
    4. Confusion Matrices (2x2 Multi-Panel Heatmap)
    5. Computational Efficiency (Training Time vs Inference Latency)
    """
    os.makedirs(vis_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({'font.sans-serif': 'Arial', 'font.family': 'sans-serif'})

    # -------------------------------------------------------------
    # 1. Dataset Class Distribution and Length Analysis
    # -------------------------------------------------------------
    print("Generating Figure 1: Class Distribution & Review Lengths...")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Class counts
        counts = df['sentiment'].value_counts()
        colors = ['#2ca02c', '#d62728']
        bars = axes[0].bar(counts.index.str.capitalize(), counts.values, color=colors, width=0.45, edgecolor='black', alpha=0.85)
        axes[0].set_title('IMDb Dataset Class Balance (50/50 Stratified)', fontsize=12, fontweight='bold', pad=12)
        axes[0].set_ylabel('Number of Samples', fontsize=11)
        axes[0].set_ylim(0, max(counts.values) * 1.15)
        for bar in bars:
            h = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., h + 100, f'{int(h):,} ({h/len(df)*100:.1f}%)', 
                         ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Word length distribution before and after cleaning
        sns.kdeplot(df['raw_word_count'], ax=axes[1], label='Raw Reviews', color='#1f77b4', fill=True, alpha=0.3)
        sns.kdeplot(df['cleaned_word_count'], ax=axes[1], label='Cleaned Reviews (Negation Preserved)', color='#ff7f0e', fill=True, alpha=0.3)
        axes[1].set_title('Review Word Count Distribution Before & After Preprocessing', fontsize=12, fontweight='bold', pad=12)
        axes[1].set_xlabel('Word Count per Review', fontsize=11)
        axes[1].set_ylabel('Density', fontsize=11)
        axes[1].set_xlim(0, 600)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'class_distribution.png'), dpi=300)
        plt.close()

    # -------------------------------------------------------------
    # Load Model Comparison Data
    # -------------------------------------------------------------
    comp_csv = os.path.join(results_dir, 'model_comparison.csv')
    if not os.path.exists(comp_csv):
        raise FileNotFoundError(f"Missing {comp_csv}. Run model_training.py first.")

    comp_df = pd.read_csv(comp_csv)
    print(f"Loaded model comparison metrics for {len(comp_df)} classifiers.")

    # -------------------------------------------------------------
    # 2. Accuracy & Macro F1 Comparison
    # -------------------------------------------------------------
    print("Generating Figure 2: Accuracy & F1-Score Comparison...")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(comp_df))
    width = 0.35

    rects1 = ax.bar(x - width/2, comp_df['Accuracy'] * 100, width, label='Accuracy (%)', color='#1f77b4', edgecolor='black', alpha=0.85)
    rects2 = ax.bar(x + width/2, comp_df['F1-Score (Macro)'] * 100, width, label='Macro F1 (%)', color='#2ca02c', edgecolor='black', alpha=0.85)

    ax.set_ylabel('Performance Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Text Classification Battle: Accuracy & Macro F1 Comparison\n(IMDb Movie Reviews Dataset - 10,000 Samples)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(comp_df['Model'], fontsize=10, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.set_ylim(50, 100)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'accuracy_f1_comparison.png'), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 3. Precision & Recall Comparison
    # -------------------------------------------------------------
    print("Generating Figure 3: Precision & Recall Comparison...")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    rects1 = ax.bar(x - width/2, comp_df['Precision (Macro)'] * 100, width, label='Macro Precision (%)', color='#9467bd', edgecolor='black', alpha=0.85)
    rects2 = ax.bar(x + width/2, comp_df['Recall (Macro)'] * 100, width, label='Macro Recall (%)', color='#ff7f0e', edgecolor='black', alpha=0.85)

    ax.set_ylabel('Performance Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Precision vs Recall Trade-off Across Classifiers', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(comp_df['Model'], fontsize=10, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.set_ylim(50, 100)

    autolabel(rects1)
    autolabel(rects2)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'precision_recall_comparison.png'), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 4. Confusion Matrices (2x2 Multi-Panel Grid)
    # -------------------------------------------------------------
    print("Generating Figure 4: Multi-Panel Confusion Matrices...")
    metrics_json = os.path.join(results_dir, 'model_metrics.json')
    with open(metrics_json, 'r') as f:
        detailed_metrics = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    model_order = [
        'Support Vector Machine (Linear)',
        'Multinomial Naive Bayes',
        'K-Nearest Neighbours',
        'Decision Tree'
    ]

    for idx, model_name in enumerate(model_order):
        if model_name in detailed_metrics:
            cm = np.array(detailed_metrics[model_name]['confusion_matrix'])
            # Normalize for percentage display
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            
            # Format annotation text: Count + (Percentage)
            annot_matrix = [
                [f"{cm[0, 0]}\n({cm_norm[0, 0]:.1f}%)", f"{cm[0, 1]}\n({cm_norm[0, 1]:.1f}%)"],
                [f"{cm[1, 0]}\n({cm_norm[1, 0]:.1f}%)", f"{cm[1, 1]}\n({cm_norm[1, 1]:.1f}%)"]
            ]

            sns.heatmap(
                cm, annot=annot_matrix, fmt='', cmap='Blues', cbar=False,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                ax=axes[idx], annot_kws={'fontsize': 11, 'fontweight': 'bold'}
            )
            acc = detailed_metrics[model_name]['accuracy'] * 100
            f1 = detailed_metrics[model_name]['f1_macro'] * 100
            axes[idx].set_title(f"{model_name}\nAcc: {acc:.2f}% | F1: {f1:.2f}%", fontsize=11, fontweight='bold', pad=10)
            axes[idx].set_xlabel('Predicted Sentiment', fontsize=10, fontweight='bold')
            axes[idx].set_ylabel('True Sentiment', fontsize=10, fontweight='bold')

    plt.suptitle('Confusion Matrices Comparison (Test Split = 2,000 Reviews)', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'confusion_matrices.png'), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 5. Computational Efficiency (Training Time vs Inference Latency)
    # -------------------------------------------------------------
    print("Generating Figure 5: Training Time vs Inference Latency...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training Time
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    bars_train = axes[0].barh(comp_df['Model'], comp_df['Training Time (s)'], color=colors, edgecolor='black', alpha=0.85)
    axes[0].set_xlabel('Training Time (seconds)', fontsize=11, fontweight='bold')
    axes[0].set_title('Training Time on 8,000 Reviews (TF-IDF + Fit)', fontsize=12, fontweight='bold', pad=12)
    axes[0].set_xlim(0, max(comp_df['Training Time (s)']) * 1.25)
    for bar in bars_train:
        w = bar.get_width()
        axes[0].text(w + 0.08, bar.get_y() + bar.get_height()/2, f'{w:.2f}s', va='center', fontsize=10, fontweight='bold')

    # Inference Latency
    bars_lat = axes[1].barh(comp_df['Model'], comp_df['Test Latency (ms/sample)'], color=colors, edgecolor='black', alpha=0.85)
    axes[1].set_xlabel('Latency per Review (milliseconds)', fontsize=11, fontweight='bold')
    axes[1].set_title('Inference Latency per Query (2,000 Test Reviews)', fontsize=12, fontweight='bold', pad=12)
    axes[1].set_xlim(0, max(comp_df['Test Latency (ms/sample)']) * 1.3)
    for bar in bars_lat:
        w = bar.get_width()
        axes[1].text(w + 0.01, bar.get_y() + bar.get_height()/2, f'{w:.3f} ms', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'training_inference_time.png'), dpi=300)
    plt.close()

    print("All evaluation visualizations successfully generated in 'visualizations/' directory.")


if __name__ == '__main__':
    generate_evaluation_visualizations()

