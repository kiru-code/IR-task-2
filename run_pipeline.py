"""
Master End-to-End Pipeline Execution Script
Author: Kirubaanithi R (Reg No: RA2411026050203)
Course: 21CSE377T - Information Retrieval
SRM Institute of Science and Technology, Tiruchirappalli
"""

import os
import sys
import time


def run_full_pipeline():
    print("=" * 80)
    print("      PRODUCT REVIEW SENTIMENT CLASSIFICATION USING MACHINE LEARNING")
    print("          SRM INSTITUTE OF SCIENCE AND TECHNOLOGY, TIRUCHIRAPPALLI")
    print("                 STUDENT: KIRUBAANITHI R (RA2411026050203)")
    print("=" * 80)

    start_total = time.time()

    # Step 1: Dataset Verification and Processing
    print("\n[STEP 1/4] Preparing Dataset & Processing Negation-Aware Reviews...")
    from download_dataset import prepare_dataset
    processed_csv = 'data/processed/imdb_cleaned_subset.csv'
    if not os.path.exists(processed_csv):
        prepare_dataset()
    else:
        print(f"Verified processed dataset at {processed_csv}")

    # Step 2: Model Training & Comparison
    print("\n[STEP 2/4] Training & Fairly Comparing 4 Classification Models...")
    from model_training import train_and_evaluate_models
    comp_df, metrics = train_and_evaluate_models()

    # Step 3: What-If Analysis & Ablation Experiments
    print("\n[STEP 3/4] Running Empirical What-If Ablation Experiments...")
    from what_if_analysis import run_what_if_experiments
    run_what_if_experiments()

    # Step 4: Visualizations Generation
    print("\n[STEP 4/4] Generating Academic Visualizations...")
    from evaluation import generate_evaluation_visualizations
    generate_evaluation_visualizations()

    total_time = time.time() - start_total
    print("\n" + "=" * 80)
    print(f"PIPELINE EXECUTION COMPLETED SUCCESSFULLY IN {total_time:.2f} SECONDS!")
    print("All models, metrics, CSV reports, and PNG visualizations are generated.")
    print("=" * 80)


if __name__ == '__main__':
    run_full_pipeline()

