"""
Streamlit Web Application: Product Review Sentiment Classification Dashboard
Student: Kirubaanithi R (Reg No: RA2411026050203)
Course: 21CSE377T - Information Retrieval | CLA-2 Task 2
Institution: SRM Institute of Science and Technology, Tiruchirappalli
"""

import os
import json
import time
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Import project modules
from preprocessing import get_preprocessing_steps, preprocess_review
from prediction import predict_all_models, predict_sentiment, MODEL_FILES

# Set page configuration
st.set_page_config(
    page_title="Review Sentiment Classification | SRM IST",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS)
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0d3b66, #001f3f);
        color: white;
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .student-badge {
        background: #f4d35e;
        color: #0d3b66;
        padding: 5px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85rem;
        display: inline-block;
        margin-right: 8px;
    }
    .metric-card {
        background: #f8f9fa;
        border-left: 5px solid #0d3b66;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }
    .winner-card {
        background: #eef9f1;
        border-left: 5px solid #28a745;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 20px;
    }
    .sentiment-pos {
        background: #d4edda;
        color: #155724;
        font-weight: bold;
        padding: 6px 14px;
        border-radius: 20px;
        display: inline-block;
    }
    .sentiment-neg {
        background: #f8d7da;
        color: #721c24;
        font-weight: bold;
        padding: 6px 14px;
        border-radius: 20px;
        display: inline-block;
    }
    .step-box {
        background: #ffffff;
        border: 1px solid #e1e4e8;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <h1 style="margin: 0; font-size: 2.1rem; color: #ffffff;">🎬 Product Review Sentiment Classification</h1>
            <p style="margin: 6px 0 0 0; font-size: 1.05rem; opacity: 0.9;">
                Information Retrieval & Text Classification Battle: Compare, Analyze and Defend (CLA-2 Task 2)
            </p>
        </div>
        <div style="text-align: right; margin-top: 10px;">
            <span class="student-badge">Student: Kirubaanithi R</span>
            <span class="student-badge">Reg No: RA2411026050203</span>
            <div style="font-size: 0.85rem; color: #e0e0e0; margin-top: 5px;">
                SRM Institute of Science and Technology, Tiruchirappalli | Course: 21CSE377T
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Helper function to load data and metrics
@st.cache_data
def load_data_and_metrics():
    comp_path = 'results/model_comparison.csv'
    metrics_path = 'results/model_metrics.json'
    what_if_path = 'results/what_if_analysis.csv'
    data_path = 'data/processed/imdb_cleaned_subset.csv'

    comp_df = pd.read_csv(comp_path) if os.path.exists(comp_path) else None
    with open(metrics_path, 'r') as f:
        metrics = json.load(f) if os.path.exists(metrics_path) else None
    what_if_df = pd.read_csv(what_if_path) if os.path.exists(what_if_path) else None
    df = pd.read_csv(data_path) if os.path.exists(data_path) else None

    return comp_df, metrics, what_if_df, df

comp_df, metrics, what_if_df, df_sample = load_data_and_metrics()

# Sidebar Navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Select Module:",
    [
        "🏠 Home & Project Scope",
        "📊 Dataset Exploration",
        "🔬 Preprocessing & TF-IDF Lab",
        "⚔️ Four-Model Battle Dashboard",
        "🎯 Confusion Matrix & Diagnostics",
        "🧪 What-If Analysis Lab",
        "⚡ Live Sentiment Predictor",
        "🛡️ Academic Defense & Viva Cheat Sheet"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Course:** 21CSE377T - Information Retrieval  
**Evaluation:** CLA-2 Task 2 (30 Marks)  
**Student:** Kirubaanithi R  
**Register No:** RA2411026050203  
**Dataset:** IMDb 10,000 Subset (50/50 Balanced)  
**Best Model:** Linear SVM (87.20% Accuracy)
""")


# ==============================================================================
# PAGE 1: HOME & PROJECT SCOPE
# ==============================================================================
if page == "🏠 Home & Project Scope":
    st.subheader("📋 Executive Project Overview")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown(r"""
        ### Objective
        The primary objective of this project is to build, evaluate, and critically defend a **binary text classification system** that predicts whether a consumer review conveys **Positive** or **Negative** sentiment.

        #### Academic Application Framing:
        - **Product Category Selected:** Entertainment Media Products (**IMDb Movie Reviews**).
        - **Why IMDb?** Movies are experiential products where sentiment is expressed through intricate human language, subtle nuances, sarcasms, and emphatic vocabulary.
        - **Honest Academic Disclaimer:** This classifier is trained specifically on English IMDb movie reviews. While its foundational algorithms generalise to broader consumer goods, its vocabulary weights reflect cinematic appraisals rather than physical electronics or retail products.

        ### Key Technical Components:
        1. **Text Preprocessing:** Contraction expansion, HTML stripping, tokenization, **negation-preserving stopword filtering**, and lemmatization.
        2. **Feature Representation:** Scikit-learn `TfidfVectorizer` (sublinear term frequency, unigram + bigram feature space, $5,000$ max features).
        3. **Data Leakage Prevention:** Built with Scikit-learn `Pipeline` objects to ensure TF-IDF is fit strictly on training partitions ($80\%$) and applied out-of-sample on test partitions ($20\%$).
        4. **Four Diverse Classifiers Evaluated:**
           - **Multinomial Naive Bayes (Probabilistic Generative)**
           - **Support Vector Machine (Maximum Margin Hyperplane)**
           - **Decision Tree (Recursive Information Gain Splitting)**
           - **K-Nearest Neighbours (Instance-based Metric Space Search)**
        """)

    with col2:
        st.markdown("""
        <div class="winner-card">
            <h3 style="margin-top: 0; color: #155724;">🏆 Benchmark Winner: Linear SVM</h3>
            <p style="font-size: 1.05rem;">
                <b>Accuracy:</b> 87.20%<br>
                <b>Macro F1-Score:</b> 87.20%<br>
                <b>Training Time:</b> 2.54 seconds<br>
                <b>Inference Latency:</b> 0.119 ms / review
            </p>
            <p style="margin-bottom: 0; font-size: 0.9rem; color: #2e663a;">
                <b>Why it won:</b> High-dimensional TF-IDF text vectors (5,000 features) are largely linearly separable. SVM finds the optimal global margin with convex optimization without suffering from axis-parallel limitations.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if comp_df is not None:
            st.markdown("#### High-Level Performance Summary")
            summary_table = comp_df[['Model', 'Accuracy', 'F1-Score (Macro)', 'Training Time (s)']].copy()
            summary_table['Accuracy'] = (summary_table['Accuracy'] * 100).map('{:.2f}%'.format)
            summary_table['F1-Score (Macro)'] = (summary_table['F1-Score (Macro)'] * 100).map('{:.2f}%'.format)
            st.dataframe(summary_table, use_container_width=True, hide_index=True)


# ==============================================================================
# PAGE 2: DATASET EXPLORATION
# ==============================================================================
elif page == "📊 Dataset Exploration":
    st.subheader("📊 Dataset Exploration & Class Distribution")
    st.markdown("""
    The dataset utilized is the **Stanford Large Movie Review Dataset (Maas et al., 2011)**, the international gold-standard benchmark for binary sentiment classification.
    To ensure reproducible, rapid experimentation and prevent memory bottlenecks during KNN instance-based evaluation, a **stratified, balanced sample of 10,000 reviews** ($5,000$ positive, $5,000$ negative) was sampled with `random_state=42`.
    """)

    if df_sample is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Samples", f"{len(df_sample):,}")
        col2.metric("Positive Reviews", f"{(df_sample['sentiment'] == 'positive').sum():,} (50%)")
        col3.metric("Negative Reviews", f"{(df_sample['sentiment'] == 'negative').sum():,} (50%)")
        col4.metric("Avg Raw Length", f"{df_sample['raw_word_count'].mean():.1f} words")

        st.markdown("---")
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.markdown("#### 1. Balanced Class Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            counts = df_sample['sentiment'].value_counts()
            colors = ['#2ca02c', '#d62728']
            bars = ax.bar(['Positive', 'Negative'], [counts['positive'], counts['negative']], color=colors, width=0.4, edgecolor='black')
            ax.set_ylabel("Count")
            ax.set_ylim(0, 6000)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 150, f"{int(bar.get_height()):,}\n(50.0%)", ha='center', fontweight='bold')
            st.pyplot(fig)

        with col_chart2:
            st.markdown("#### 2. Word Length Density (Raw vs Cleaned)")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.kdeplot(df_sample['raw_word_count'], ax=ax, label='Raw Reviews', color='#1f77b4', fill=True, alpha=0.3)
            sns.kdeplot(df_sample['cleaned_word_count'], ax=ax, label='Cleaned Reviews', color='#ff7f0e', fill=True, alpha=0.3)
            ax.set_xlabel("Words per Review")
            ax.set_ylabel("Density")
            ax.set_xlim(0, 600)
            ax.legend()
            st.pyplot(fig)

        st.markdown("#### 🔍 Sample Records from Verified Dataset")
        display_df = df_sample[['sentiment', 'raw_word_count', 'cleaned_word_count', 'review']].head(6)
        st.dataframe(display_df, use_container_width=True)


# ==============================================================================
# PAGE 3: PREPROCESSING & TF-IDF LAB
# ==============================================================================
elif page == "🔬 Preprocessing & TF-IDF Lab":
    st.subheader("🔬 Interactive Preprocessing & TF-IDF Laboratory")
    st.markdown("""
    Explore how raw user text undergoes transformation into cleaned tokens and mathematical vectors.
    Notice the preservation of sentiment-critical negation tokens (`not`, `never`, `no`), which prevents polarity reversal!
    """)

    sample_default = "I did NOT like this movie at all! The acting was terrible and the plot made no sense. However, the visual effects weren't bad."
    user_input = st.text_area("Enter any custom review to deconstruct its preprocessing pipeline:", value=sample_default, height=100)

    if user_input:
        steps = get_preprocessing_steps(user_input)

        col_left, col_right = st.columns([3, 2])
        with col_left:
            st.markdown("#### Pipeline Step-by-Step Deconstruction:")

            st.markdown("**1. Raw Input Review:**")
            st.markdown(f'<div class="step-box">{steps["raw_text"]}</div>', unsafe_allow_html=True)

            st.markdown("**2. Contraction Expansion (`didn't` -> `did not`, `weren't` -> `were not`):**")
            st.markdown(f'<div class="step-box">{steps["contractions_expanded"]}</div>', unsafe_allow_html=True)

            st.markdown("**3. HTML Stripping & Noise Removal:**")
            st.markdown(f'<div class="step-box">{steps["html_and_noise_removed"]}</div>', unsafe_allow_html=True)

            st.markdown("**4. Tokenization:**")
            st.write(steps["tokens"])

            st.markdown("**5. Stop-word Removal (WITH Sentiment Negation Preservation):**")
            st.info(f"Preserved Negation Words: `not`, `no` are preserved! Standard stopwords like `this`, `at`, `all`, `the`, `was`, `and` are stripped.")
            st.write(steps["stopwords_removed"])

            st.markdown("**6. Lemmatization (WordNet Root Forms):**")
            st.write(steps["lemmatized_tokens"])

            st.markdown("**7. Final Preprocessed Text String:**")
            st.success(steps["final_preprocessed_text"])

        with col_right:
            st.markdown("#### Preprocessing Statistics:")
            st.metric("Raw Word Count", steps["raw_word_count"])
            st.metric("Cleaned Word Count", steps["cleaned_word_count"])
            st.metric("Noise Compression", f"{steps['reduction_percentage']}%")

            st.markdown("---")
            st.markdown("""
            #### 📐 Mathematical Foundations: TF-IDF
            $$TF(t, d) = \\frac{\\text{Count}(t, d)}{\\sum_{t' \\in d} \\text{Count}(t', d)}$$
            $$IDF(t, D) = \\ln\\left(\\frac{1 + |D|}{1 + |\\{d \\in D : t \\in d\\}|}\\right) + 1$$
            $$\\text{TF-IDF}(t, d, D) = TF(t, d) \\times IDF(t, D)$$
            
            **Why TF-IDF?**
            - **Term Frequency (TF):** Rewards words that appear frequently within a specific review.
            - **Inverse Document Frequency (IDF):** Penalizes words that appear ubiquitously across all reviews.
            - **Sublinear TF:** We apply $1 + \\log(tf)$ so that a word occurring 10 times is not weighted 10 times as strongly as a word occurring once.
            """)


# ==============================================================================
# PAGE 4: FOUR-MODEL BATTLE DASHBOARD
# ==============================================================================
elif page == "⚔️ Four-Model Battle Dashboard":
    st.subheader("⚔️ The Four-Model Battle: Comprehensive Benchmark")
    st.markdown("""
    All four classifiers were evaluated under **strictly identical experimental conditions**:
    - **Dataset:** 10,000 stratified samples (8,000 training, 2,000 testing).
    - **Representation:** Identical TF-IDF Vectorizer ($5,000$ features, Unigram + Bigram, Sublinear TF).
    - **Data Leakage Hygiene:** Pipelines fit strictly on training splits.
    """)

    if comp_df is not None:
        st.markdown("### 🏆 Performance Comparison Matrix")
        styled_df = comp_df[['Model', 'Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)', 'Training Time (s)', 'Test Latency (ms/sample)']].copy()
        styled_df['Accuracy'] = (styled_df['Accuracy'] * 100).map('{:.2f}%'.format)
        styled_df['Precision (Macro)'] = (styled_df['Precision (Macro)'] * 100).map('{:.2f}%'.format)
        styled_df['Recall (Macro)'] = (styled_df['Recall (Macro)'] * 100).map('{:.2f}%'.format)
        styled_df['F1-Score (Macro)'] = (styled_df['F1-Score (Macro)'] * 100).map('{:.2f}%'.format)
        styled_df['Training Time (s)'] = styled_df['Training Time (s)'].map('{:.3f} s'.format)
        styled_df['Test Latency (ms/sample)'] = styled_df['Test Latency (ms/sample)'].map('{:.3f} ms'.format)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["📊 Accuracy & F1 Bar Chart", "⚖️ Precision vs Recall", "⏱️ Computational Latency"])

        with tab1:
            st.image('visualizations/accuracy_f1_comparison.png', caption='Figure 2: Accuracy and Macro F1 Comparison across all 4 Classifiers', use_container_width=True)

        with tab2:
            st.image('visualizations/precision_recall_comparison.png', caption='Figure 3: Precision vs Recall across all 4 Classifiers', use_container_width=True)

        with tab3:
            st.image('visualizations/training_inference_time.png', caption='Figure 5: Training Time (s) and Inference Latency (ms/query)', use_container_width=True)


# ==============================================================================
# PAGE 5: CONFUSION MATRIX & DIAGNOSTICS
# ==============================================================================
elif page == "🎯 Confusion Matrix & Diagnostics":
    st.subheader("🎯 Confusion Matrices & Diagnostic Error Analysis")
    st.markdown("""
    Detailed breakdown of True Negatives (TN), False Positives (FP), False Negatives (FN), and True Positives (TP) on the **2,000 unseen test reviews** (1,000 negative, 1,000 positive).
    """)

    st.image('visualizations/confusion_matrices.png', caption='Figure 4: Multi-Panel Normalized Confusion Matrices', use_container_width=True)

    if metrics is not None:
        st.markdown("### Per-Class Detailed Breakdown")
        cols = st.columns(4)
        for idx, (m_name, m_data) in enumerate(metrics.items()):
            with cols[idx]:
                st.markdown(f"**{m_name}**")
                cm = m_data['confusion_matrix']
                st.markdown(f"- **TN (Negative Correct):** {cm[0][0]}")
                st.markdown(f"- **FP (False Alarm):** {cm[0][1]}")
                st.markdown(f"- **FN (Missed Positive):** {cm[1][0]}")
                st.markdown(f"- **TP (Positive Correct):** {cm[1][1]}")
                st.markdown(f"- **Accuracy:** {m_data['accuracy']*100:.2f}%")
                st.markdown(f"- **Neg F1:** {m_data['class_metrics']['negative']['f1']*100:.2f}%")
                st.markdown(f"- **Pos F1:** {m_data['class_metrics']['positive']['f1']*100:.2f}%")


# ==============================================================================
# PAGE 6: WHAT-IF ANALYSIS LAB
# ==============================================================================
elif page == "🧪 What-If Analysis Lab":
    st.subheader("🧪 What-If Analysis & Ablation Experiments")
    st.markdown(r"""
    Rigorous scientific experimentation requires altering independent variables to observe their direct impact on classification performance.
    We conducted three implemented and verified What-If experiments:
    1. **Ablation 1:** Vocabulary Size (`max_features` $\in \{1000, 2500, 5000, 10000\}$).
    2. **Ablation 2:** N-Gram Feature Representations (Unigrams vs Unigrams+Bigrams vs Bigrams only).
    3. **Ablation 3:** Hyperparameter Sensitivity Curves (SVM $C$, Naive Bayes $\alpha$, KNN $k$).
    """)

    st.image('visualizations/what_if_analysis.png', caption='Figure 6: Empirical What-If Ablation Visualizations', use_container_width=True)

    if what_if_df is not None:
        st.markdown("### 📋 What-If Empirical Findings Table")
        exp_choice = st.selectbox("Select Experiment to Filter:", what_if_df['Experiment'].unique())
        filtered_exp = what_if_df[what_if_df['Experiment'] == exp_choice]
        st.dataframe(filtered_exp, use_container_width=True, hide_index=True)

        st.markdown(r"""
        #### 💡 Key Observed Effects & Scientific Takeaways:
        - **Vocabulary Size Effect:** Expanding `max_features` from $1,000$ to $5,000$ yields a **$+3.2\%$ boost** in SVM accuracy. Beyond $5,000$, returns diminish ($+0.3\%$) while increasing memory usage.
        - **N-Gram Range Effect:** Adding Bigrams (`ngram_range=(1,2)`) provides critical context for negation phrases (e.g. `not good`, `never recommend`), yielding an additional **$+1.4\%$ accuracy improvement** over unigrams alone. However, Bigrams only (`(2,2)`) suffers from extreme sparsity and lower performance ($78.5\%$).
        - **Hyperparameter Tuning:** SVM with $C=1.0$ strikes the optimal bias-variance tradeoff; $C=0.01$ underfits ($82.1\%$), while $C=10.0$ slightly overfits ($86.4\%$).
        """)


# ==============================================================================
# PAGE 7: LIVE SENTIMENT PREDICTOR
# ==============================================================================
elif page == "⚡ Live Sentiment Predictor":
    st.subheader("⚡ Live Review Sentiment Predictor")
    st.markdown("Enter any custom movie/product review or choose from preloaded test cases to see real-time predictions across all four models simultaneously!")

    preset = st.selectbox(
        "Choose a Preset Test Review (or select Custom):",
        [
            "Custom Input",
            "Preset 1: Masterpiece Praise (Positive)",
            "Preset 2: Terrible Flop (Negative)",
            "Preset 3: Subtle Negation (Negative with 'not bad' / 'never')",
            "Preset 4: Sarcastic Review (Challenging Edge Case)"
        ]
    )

    preset_texts = {
        "Preset 1: Masterpiece Praise (Positive)": "An absolute cinematic masterpiece! The direction was brilliant, the screenplay was engaging, and the lead actor gave an Oscar-worthy performance.",
        "Preset 2: Terrible Flop (Negative)": "A colossal waste of time and money. The acting was painful to watch, the plot was filled with gigantic holes, and I walked out halfway through.",
        "Preset 3: Subtle Negation (Negative with 'not bad' / 'never')": "This was not a good movie at all, and I would never recommend it to anyone who values their time.",
        "Preset 4: Sarcastic Review (Challenging Edge Case)": "Oh brilliant! Another generic two-hour superhero cliché where CGI explosions replace actual acting and meaningful dialogue."
    }

    if preset == "Custom Input":
        input_text = st.text_area("Write or paste a review below:", value="The cinematography was breathtaking, but the confusing plot and terrible pacing ruined what could have been a decent film.", height=120)
    else:
        input_text = st.text_area("Review Text:", value=preset_texts[preset], height=120)

    if st.button("🚀 Analyze Sentiment across All 4 Models", type="primary"):
        if not input_text.strip():
            st.warning("Please enter some review text.")
        else:
            with st.spinner("Processing text and running models..."):
                results = predict_all_models(input_text)

            st.markdown("### 🏆 Prediction Results Battle")
            cols = st.columns(4)

            for idx, (m_name, res) in enumerate(results.items()):
                with cols[idx]:
                    sentiment = res['predicted_sentiment']
                    badge_class = "sentiment-pos" if sentiment == "Positive" else "sentiment-neg"
                    st.markdown(f"**{m_name}**")
                    st.markdown(f'<div class="{badge_class}">{sentiment}</div>', unsafe_allow_html=True)
                    st.metric("Confidence", f"{res['confidence']:.1f}%")
                    st.caption(f"⏱️ Latency: {res['latency_ms']:.2f} ms")

            st.markdown("---")
            st.markdown("#### 🔍 Term Salience & Attributions (Why did the models predict this?)")
            svm_res = results.get("Support Vector Machine", {})
            terms = svm_res.get("top_contributing_terms", [])
            if terms:
                terms_df = pd.DataFrame(terms)
                st.dataframe(terms_df, use_container_width=True, hide_index=True)
            else:
                st.info("No dominant vocabulary terms matched the top features.")

            st.markdown("#### 🔄 Preprocessing Pipeline Transformation for this Review:")
            prep = svm_res.get('preprocessing_steps', {})
            st.code(f"Original:  {prep.get('raw_text', '')}\nCleaned:   {prep.get('final_preprocessed_text', '')}")


# ==============================================================================
# PAGE 8: ACADEMIC DEFENSE & VIVA GUIDE
# ==============================================================================
elif page == "🛡️ Academic Defense & Viva Cheat Sheet":
    st.subheader("🛡️ Academic Defense & Viva Voce Quick Reference")
    st.markdown("""
    Use these structured technical arguments during your evaluation and viva defense.
    """)

    with st.expander("Q1: Why did Support Vector Machine (Linear) outperform Decision Trees and KNN?", expanded=True):
        st.markdown("""
        **Technical Defence:**
        Text representations constructed with TF-IDF generate high-dimensional ($5,000+$ features) and sparse feature matrices.
        - **SVM Strengths:** In high-dimensional vector spaces, text classes are known to be almost linearly separable. Linear SVM finds the maximum-margin hyperplane through convex quadratic programming, effectively utilizing all non-zero term coordinates without falling victim to overfitting.
        - **Decision Tree Weakness:** Decision Trees perform orthogonal, axis-parallel recursive splits on single features (words). In a $5,000$-dimensional space where each review contains only $50-100$ non-zero terms, single-feature splits fragment the feature space and suffer from extreme data sparsity and variance.
        - **KNN Weakness:** KNN suffers from the **Curse of Dimensionality**. In high dimensions, Euclidean or even cosine distances between queries and training points tend to concentrate, making neighbor discrimination noisy and computationally expensive ($O(N \\cdot D)$ per query).
        """)

    with st.expander("Q2: How did you prevent Data Leakage between Training and Test sets?"):
        st.markdown("""
        **Technical Defence:**
        Data leakage occurs when information outside the training dataset is used to create the model (e.g. fitting the TF-IDF vectorizer across the entire dataset before splitting).
        - In our implementation, we used **Scikit-learn `Pipeline` objects**.
        - The `TfidfVectorizer` was fit **strictly on `X_train`** ($8,000$ samples).
        - Test data (`X_test`, $2,000$ samples) was only transformed using the vocabulary and IDF weights learned from the training split.
        - Out-of-vocabulary terms in the test set were safely ignored, ensuring strict generalization integrity.
        """)

    with st.expander("Q3: Why was Negation Preservation critical during stop-word removal?"):
        st.markdown("""
        **Technical Defence:**
        Standard NLTK stop-word lists contain words like `not`, `no`, `never`, `nor`, and `without`.
        - If standard stop-word removal is applied blindly, a phrase like *"not a good movie"* is reduced to *"good movie"*, completely inverting the ground-truth sentiment from Negative to Positive!
        - We explicitly filtered the stop-word dictionary: `STOP_WORDS = raw_stopwords - NEGATION_WORDS`.
        - This retained negative polarity qualifiers while eliminating genuinely non-informative functional words (`the`, `is`, `at`, `which`).
        """)

    with st.expander("Q4: Explain the difference between Macro F1 and Weighted F1."):
        st.markdown("""
        **Technical Defence:**
        - **Macro F1:** Calculates F1 unweighted for each class independently and averages them: $\\frac{F1_{\\text{pos}} + F1_{\\text{neg}}}{2}$. It treats each class with equal importance regardless of class size.
        - **Weighted F1:** Weights each class's F1 score by its support (number of true instances).
        - In our experiment, because our test split was strictly balanced ($1,000$ positive and $1,000$ negative reviews), Macro F1 and Weighted F1 yield identical values ($87.20\\%$ for SVM), confirming robust balance.
        """)
