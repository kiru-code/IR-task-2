"""
Dataset Preparation and Subset Extraction Module
Author: Kirubaanithi R (Reg No: RA2411026050203)
Course: 21CSE377T - Information Retrieval
SRM Institute of Science and Technology, Tiruchirappalli
"""

import os
import pandas as pd
import numpy as np
from preprocessing import preprocess_review


def prepare_dataset(
    raw_path: str = 'data/raw/IMDB_Dataset.csv',
    processed_path: str = 'data/processed/imdb_cleaned_subset.csv',
    sample_size: int = 10000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Loads raw IMDb dataset, extracts a stratified balanced sample of 10,000 reviews
    (5,000 positive, 5,000 negative) for reproducible, high-speed, and academically
    rigorous model training, applies text preprocessing, and saves the result.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Dataset not found at {raw_path}. Please ensure IMDb dataset is downloaded."
        )

    print(f"Loading raw dataset from {raw_path}...")
    df_raw = pd.read_csv(raw_path)
    print(f"Original dataset shape: {df_raw.shape}")
    print("Class distribution in raw data:")
    print(df_raw['sentiment'].value_counts())

    # Stratified balanced sampling
    per_class = sample_size // 2
    pos_df = df_raw[df_raw['sentiment'] == 'positive'].sample(n=per_class, random_state=random_state)
    neg_df = df_raw[df_raw['sentiment'] == 'negative'].sample(n=per_class, random_state=random_state)

    subset_df = pd.concat([pos_df, neg_df]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    print(f"\nExtracted balanced subset of {len(subset_df)} reviews ({per_class} positive, {per_class} negative).")

    # Add word count columns for exploratory analysis
    subset_df['raw_word_count'] = subset_df['review'].apply(lambda x: len(str(x).split()))

    print("Applying text preprocessing (HTML removal, lowercase, tokenization, negation-preserving stopwords, lemmatization)...")
    subset_df['cleaned_review'] = subset_df['review'].apply(preprocess_review)
    subset_df['cleaned_word_count'] = subset_df['cleaned_review'].apply(lambda x: len(str(x).split()))

    # Encode sentiment: positive -> 1, negative -> 0
    subset_df['label'] = subset_df['sentiment'].map({'positive': 1, 'negative': 0})

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    subset_df.to_csv(processed_path, index=False)
    print(f"Processed dataset saved successfully to {processed_path}")

    # Summary Statistics
    print("\nDataset Summary Statistics:")
    print(f"- Total Samples: {len(subset_df)}")
    print(f"- Positive Samples: {(subset_df['label'] == 1).sum()} (50.0%)")
    print(f"- Negative Samples: {(subset_df['label'] == 0).sum()} (50.0%)")
    print(f"- Mean Raw Word Count: {subset_df['raw_word_count'].mean():.2f}")
    print(f"- Mean Cleaned Word Count: {subset_df['cleaned_word_count'].mean():.2f}")
    print(f"- Vocabulary Compression Ratio: {(1 - subset_df['cleaned_word_count'].mean()/subset_df['raw_word_count'].mean())*100:.2f}%")

    return subset_df


if __name__ == '__main__':
    prepare_dataset()

