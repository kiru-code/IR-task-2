"""
PowerPoint Presentation Generator using python-pptx
Author: Kirubaanithi R (Reg No: RA2411026050203)
Course: 21CSE377T - Information Retrieval
SRM Institute of Science and Technology, Tiruchirappalli
"""

import os
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_presentation():
    prs = Presentation()
    # Set slide dimensions to standard 16:9 widescreen
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Theme Colors
    COLOR_PRIMARY = RGBColor(13, 59, 102)     # Deep Navy (#0D3B66)
    COLOR_SECONDARY = RGBColor(244, 211, 94)  # Accent Gold (#F4D35E)
    COLOR_TEXT_DARK = RGBColor(33, 37, 41)    # Charcoal Dark (#212529)
    COLOR_TEXT_LIGHT = RGBColor(240, 240, 240)# Soft White (#F0F0F0)
    COLOR_ACCENT_GREEN = RGBColor(40, 167, 69)# Green (#28A745)
    COLOR_CARD_BG = RGBColor(245, 247, 250)   # Light Gray-Blue

    def add_header(slide, title_text, category_text="21CSE377T: INFORMATION RETRIEVAL | CLA-2 TASK 2"):
        # Header banner shape
        banner = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.1))
        banner.fill.solid()
        banner.fill.fore_color.rgb = COLOR_PRIMARY
        banner.line.color.rgb = COLOR_PRIMARY

        # Category/Course text
        txBox1 = slide.shapes.add_textbox(Inches(0.8), Inches(0.12), Inches(11.7), Inches(0.3))
        tf1 = txBox1.text_frame
        p1 = tf1.paragraphs[0]
        p1.text = category_text.upper()
        p1.font.size = Pt(10)
        p1.font.bold = True
        p1.font.color.rgb = COLOR_SECONDARY

        # Slide Title text
        txBox2 = slide.shapes.add_textbox(Inches(0.8), Inches(0.38), Inches(11.7), Inches(0.6))
        tf2 = txBox2.text_frame
        p2 = tf2.paragraphs[0]
        p2.text = title_text
        p2.font.size = Pt(22)
        p2.font.bold = True
        p2.font.color.rgb = COLOR_TEXT_LIGHT

        # Footer banner
        footer = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(7.05), Inches(13.333), Inches(0.45))
        footer.fill.solid()
        footer.fill.fore_color.rgb = COLOR_PRIMARY
        footer.line.color.rgb = COLOR_PRIMARY

        txBox_f = slide.shapes.add_textbox(Inches(0.8), Inches(7.1), Inches(11.7), Inches(0.35))
        tf_f = txBox_f.text_frame
        p_f = tf_f.paragraphs[0]
        p_f.text = "Kirubaanithi R | RA2411026050203 | SRM Institute of Science and Technology, Tiruchirappalli"
        p_f.font.size = Pt(10)
        p_f.font.color.rgb = COLOR_TEXT_LIGHT

    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    slide1 = prs.slides.add_slide(blank_layout)
    bg1 = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = COLOR_PRIMARY
    bg1.line.color.rgb = COLOR_PRIMARY

    # Accent decorative strip
    strip = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.5), Inches(0.18), Inches(4.5))
    strip.fill.solid()
    strip.fill.fore_color.rgb = COLOR_SECONDARY
    strip.line.color.rgb = COLOR_SECONDARY

    t_box = slide1.shapes.add_textbox(Inches(1.2), Inches(1.4), Inches(11.2), Inches(4.7))
    tf = t_box.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "PRODUCT REVIEW SENTIMENT CLASSIFICATION"
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = COLOR_TEXT_LIGHT

    p2 = tf.add_paragraph()
    p2.text = "USING MACHINE LEARNING"
    p2.font.size = Pt(32)
    p2.font.bold = True
    p2.font.color.rgb = COLOR_SECONDARY
    p2.space_after = Pt(24)

    p3 = tf.add_paragraph()
    p3.text = "CLA-2 Task 2 – Text Classification Battle: Compare, Analyze and Defend"
    p3.font.size = Pt(18)
    p3.font.color.rgb = COLOR_TEXT_LIGHT
    p3.space_after = Pt(28)

    p4 = tf.add_paragraph()
    p4.text = "Student Name: Kirubaanithi R  |  Register No: RA2411026050203"
    p4.font.size = Pt(15)
    p4.font.bold = True
    p4.font.color.rgb = COLOR_TEXT_LIGHT

    p5 = tf.add_paragraph()
    p5.text = "Course: 21CSE377T – Information Retrieval  |  Max Marks: 30"
    p5.font.size = Pt(14)
    p5.font.color.rgb = COLOR_TEXT_LIGHT

    p6 = tf.add_paragraph()
    p6.text = "Department of Computer Science & Engineering, SRM IST, Tiruchirappalli"
    p6.font.size = Pt(13)
    p6.font.color.rgb = COLOR_SECONDARY

    slide1.notes_slide.notes_text_frame.text = (
        "Good morning respected evaluators. I am Kirubaanithi R, register number RA2411026050203, "
        "presenting CLA-2 Task 2 for Information Retrieval (21CSE377T). My project is titled "
        "'Product Review Sentiment Classification Using Machine Learning'. In this work, I benchmark, "
        "analyze, and defend four foundational text classifiers on 10,000 authentic IMDb reviews."
    )

    # =========================================================================
    # SLIDE 2: INTRODUCTION & MOTIVATION
    # =========================================================================
    slide2 = prs.slides.add_slide(blank_layout)
    add_header(slide2, "1. Introduction & Theoretical Motivation")

    card1 = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.2))
    card1.fill.solid()
    card1.fill.fore_color.rgb = COLOR_CARD_BG
    card1.line.color.rgb = COLOR_PRIMARY
    tf_c1 = card1.text_frame
    tf_c1.word_wrap = True
    p = tf_c1.paragraphs[0]
    p.text = "Why Sentiment Classification Matters"
    p.font.bold = True
    p.font.size = Pt(18)
    p.font.color.rgb = COLOR_PRIMARY
    p.space_after = Pt(14)

    bullets1 = [
        "Unstructured Natural Language: Over 80% of enterprise customer data exists as free-form text reviews.",
        "Consumer Decision-Making: Online buyers rely heavily on public review polarity before purchasing entertainment media or electronics.",
        "Automated Information Retrieval: Binary polarity classification enables search engines to aggregate consumer satisfaction scores automatically.",
        "Core Academic Challenge: Human sentiment features high lexical variety, sarcastic nuances, domain jargon, and negations."
    ]
    for b in bullets1:
        p_b = tf_c1.add_paragraph()
        p_b.text = "• " + b
        p_b.font.size = Pt(13)
        p_b.font.color.rgb = COLOR_TEXT_DARK
        p_b.space_after = Pt(10)

    card2 = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.2))
    card2.fill.solid()
    card2.fill.fore_color.rgb = COLOR_CARD_BG
    card2.line.color.rgb = COLOR_PRIMARY
    tf_c2 = card2.text_frame
    tf_c2.word_wrap = True
    p = tf_c2.paragraphs[0]
    p.text = "The 'Battle' Paradigm (CLA-2 Task 2)"
    p.font.bold = True
    p.font.size = Pt(18)
    p.font.color.rgb = COLOR_PRIMARY
    p.space_after = Pt(14)

    bullets2 = [
        "Four Contending Algorithmic Paradigms:",
        "  1. Probabilistic Generative (Multinomial Naive Bayes)",
        "  2. Maximum-Margin Hyperplane (Support Vector Machine)",
        "  3. Non-parametric Information Gain (Decision Tree)",
        "  4. Instance-based Lazy Distance (K-Nearest Neighbours)",
        "Strict Fair Play: Identical data, identical TF-IDF vectorization, and identical stratified test sets.",
        "Defensibility: We examine why certain architectures inherently excel or falter on high-dimensional sparse text vectors."
    ]
    for b in bullets2:
        p_b = tf_c2.add_paragraph()
        p_b.text = "• " + b if not b.startswith("  ") else b
        p_b.font.size = Pt(13)
        p_b.font.color.rgb = COLOR_TEXT_DARK
        p_b.space_after = Pt(8)

    slide2.notes_slide.notes_text_frame.text = (
        "In this slide, I establish the motivation. Natural language reviews constitute over 80% of user-generated "
        "feedback. The goal of this assignment is not merely to train models, but to pit four distinct algorithmic "
        "paradigms against each other under identical conditions and explain why their mathematical properties lead "
        "to dramatically different performances on textual feature spaces."
    )

    # =========================================================================
    # SLIDE 3: PROBLEM STATEMENT & APPLICATION SCOPE
    # =========================================================================
    slide3 = prs.slides.add_slide(blank_layout)
    add_header(slide3, "2. Problem Statement & Honest Academic Scope")

    t_box3 = slide3.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.2))
    tf3 = t_box3.text_frame
    tf3.word_wrap = True

    p = tf3.paragraphs[0]
    p.text = "Formal Problem Definition:"
    p.font.bold = True
    p.font.size = Pt(18)
    p.font.color.rgb = COLOR_PRIMARY
    p.space_after = Pt(8)

    p_body = tf3.add_paragraph()
    p_body.text = (
        "Given an arbitrary textual customer review r in R, construct a parameterized classification function "
        "f: r -> {Positive, Negative} that accurately maps the semantic sentiment polarity while maximizing "
        "generalization accuracy and minimizing inference latency."
    )
    p_body.font.size = Pt(14)
    p_body.font.color.rgb = COLOR_TEXT_DARK
    p_body.space_after = Pt(18)

    p_scope = tf3.add_paragraph()
    p_scope.text = "Domain Scope & Honest Academic Framing:"
    p_scope.font.bold = True
    p_scope.font.size = Pt(18)
    p_scope.font.color.rgb = COLOR_PRIMARY
    p_scope.space_after = Pt(8)

    scope_points = [
        "Product Domain Selected: Entertainment Media Products (IMDb Movie Reviews).",
        "Why Movie Reviews? Film reviews are an intense testbed for sentiment classifiers because reviewers express sentiment through complex idioms ('a feast for the eyes'), sarcasm, dramatic hyperbole, and multi-clause sentences.",
        "Academic Honesty Disclaimer: This model is specifically trained on English movie reviews. We do not claim universal generalization across unrelated commercial domains like hardware electronics without domain adaptation.",
        "Data Hygiene: Zero data leakage is permitted between training and evaluation phases."
    ]
    for pt in scope_points:
        p_pt = tf3.add_paragraph()
        p_pt.text = "• " + pt
        p_pt.font.size = Pt(13)
        p_pt.font.color.rgb = COLOR_TEXT_DARK
        p_pt.space_after = Pt(10)

    slide3.notes_slide.notes_text_frame.text = (
        "Here I define our problem formulation and make our academic scope explicit. As required by the CLA-2 rubric, "
        "we do not make false claims that our system magically understands all commercial products. We explicitly "
        "ground our study in entertainment media products—specifically IMDb movie reviews—which represent one of the "
        "most linguistically challenging sentiment domains available."
    )

    # =========================================================================
    # SLIDE 4: DATASET & STRATIFIED SAMPLING
    # =========================================================================
    slide4 = prs.slides.add_slide(blank_layout)
    add_header(slide4, "3. Dataset Profile & Stratified Sampling")

    card_d1 = slide4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.2))
    card_d1.fill.solid()
    card_d1.fill.fore_color.rgb = COLOR_CARD_BG
    card_d1.line.color.rgb = COLOR_PRIMARY
    tf_d1 = card_d1.text_frame
    tf_d1.word_wrap = True

    p = tf_d1.paragraphs[0]
    p.text = "Stanford Large Movie Review Dataset"
    p.font.bold = True
    p.font.size = Pt(17)
    p.font.color.rgb = COLOR_PRIMARY
    p.space_after = Pt(12)

    d_points = [
        "Source: Andrew Maas et al. (Stanford AI Laboratory, 2011).",
        "Total Benchmark Volume: 50,000 full reviews (25,000 positive, 25,000 negative).",
        "Documented Subset Extracted: 10,000 reviews sampled reproducibly using random_state=42.",
        "Stratification: Perfectly balanced 50.0% Positive (5,000) and 50.0% Negative (5,000).",
        "Train-Test Partitioning: 80% Training (8,000 reviews), 20% Unseen Testing (2,000 reviews).",
        "Noise Characteristics: Embedded HTML tags (<br />), varied lengths (10 to 1,000+ words), colloquial grammar."
    ]
    for dp in d_points:
        p_d = tf_d1.add_paragraph()
        p_d.text = "• " + dp
        p_d.font.size = Pt(12.5)
        p_d.font.color.rgb = COLOR_TEXT_DARK
        p_d.space_after = Pt(8)

    # Insert class distribution chart image
    if os.path.exists('visualizations/class_distribution.png'):
        slide4.shapes.add_picture('visualizations/class_distribution.png', Inches(6.7), Inches(1.5), width=Inches(5.8))

    slide4.notes_slide.notes_text_frame.text = (
        "Regarding data provenance: We utilize the authentic Stanford Large Movie Review dataset. We extract a "
        "stratified, reproducible subset of 10,000 samples—5,000 positive and 5,000 negative. This sample size provides "
        "immense statistical confidence while allowing algorithms like KNN and Decision Trees to train and evaluate "
        "rapidly without hanging system memory."
    )

    # =========================================================================
    # SLIDE 5: PREPROCESSING & NEGATION PRESERVATION
    # =========================================================================
    slide5 = prs.slides.add_slide(blank_layout)
    add_header(slide5, "4. Text Preprocessing & Negation Preservation")

    card_p1 = slide5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.2))
    card_p1.fill.solid()
    card_p1.fill.fore_color.rgb = COLOR_CARD_BG
    card_p1.line.color.rgb = COLOR_PRIMARY
    tf_p1 = card_p1.text_frame
    tf_p1.word_wrap = True

    p = tf_p1.paragraphs[0]
    p.text = "End-to-End Cleaning Pipeline"
    p.font.bold = True
    p.font.size = Pt(17)
    p.font.color.rgb = COLOR_PRIMARY
    p.space_after = Pt(10)

    p_steps = [
        "1. Contraction Expansion: Normalizes short forms (e.g., 'didn\\'t' -> 'did not', 'weren\\'t' -> 'were not').",
        "2. HTML Tag Stripping: Eliminates '<br />', formatting markup, URLs, and non-alphanumeric noise.",
        "3. Case Normalization: Lowercases all character strings.",
        "4. Tokenization: Splits whitespace into individual word tokens.",
        "5. Sentiment-Aware Stop-word Removal: Filters non-informative functional words while strictly PRESERVING negation terms.",
        "6. Lemmatization: Applies WordNetLemmatizer to reduce inflected forms (e.g. 'acting' -> 'act', 'effects' -> 'effect')."
    ]
    for ps in p_steps:
        p_s = tf_p1.add_paragraph()
        p_s.text = ps
        p_s.font.size = Pt(12)
        p_s.font.color.rgb = COLOR_TEXT_DARK
        p_s.space_after = Pt(6)

    card_p2 = slide5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.2))
    card_p2.fill.solid()
    card_p2.fill.fore_color.rgb = COLOR_CARD_BG
    card_p2.line.color.rgb = COLOR_PRIMARY
    tf_p2 = card_p2.text_frame
    tf_p2.word_wrap = True

    p = tf_p2.paragraphs[0]
    p.text = "Why Preserving Negation is Critical"
    p.font.bold = True
    p.font.size = Pt(17)
    p.font.color.rgb = COLOR_PRIMARY
    p.space_after = Pt(10)

    p_neg = tf_p2.add_paragraph()
    p_neg.text = (
        "Standard NLP stopword packages (like raw NLTK) contain words like 'not', 'no', 'never', 'nor'. "
        "If stripped indiscriminately, disastrous polarity reversals occur:\n\n"
        "Raw: 'I did not like this movie; the script made no sense.'\n"
        "Naive Stopword Removal: 'like movie script made sense'  [INVERTED TO POSITIVE!]\n\n"
        "Our Negation-Preserving Approach:\n"
        "'not like movie script made no sense'  [ACCURATE NEGATIVE!]\n\n"
        "Empirical Impact: Preserving negation tokens boosts classification accuracy by over +2.4% on nuanced reviews."
    )
    p_neg.font.size = Pt(13)
    p_neg.font.color.rgb = COLOR_TEXT_DARK

    slide5.notes_slide.notes_text_frame.text = (
        "This slide highlights one of our most critical engineering decisions: Negation Preservation. Standard NLTK "
        "stopwords discard words like 'not' and 'no'. In sentiment classification, discarding 'not' turns 'not good' "
        "into 'good', creating catastrophic false positives. We modified the stopword set to retain negation markers, "
        "preserving the true semantic polarity."
    )

    # =========================================================================
    # SLIDE 6: TF-IDF & DATA LEAKAGE HYGIENE
    # =========================================================================
    slide6 = prs.slides.add_slide(blank_layout)
    add_header(slide6, "5. Feature Representation: TF-IDF & Leakage Hygiene")

    t_box6 = slide6.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.2))
    tf6 = t_box6.text_frame
    tf6.word_wrap = True

    p = tf6.paragraphs[0]
    p.text = "Term Frequency - Inverse Document Frequency (TF-IDF):"
    p.font.bold = True
    p.font.size = Pt(17)
    p.font.color.rgb = COLOR_PRIMARY
    p.space_after = Pt(8)

    tfidf_exp = [
        "Term Frequency (TF): Measures localized prominence of term t in document d: TF(t,d) = 1 + log(count(t,d)) [Sublinear TF prevents frequent words from dominating].",
        "Inverse Document Frequency (IDF): Measures global discriminative specificity: IDF(t, D) = ln((1 + |D|) / (1 + |d in D : t in d|)) + 1.",
        "TF-IDF Weight: High weight is given to words that appear often in a specific review but rarely across the general corpus.",
        "Feature Configuration: max_features = 5,000, ngram_range = (1, 2) [Unigrams + Bigrams capture phrases like 'not recommend', 'great film']."
    ]
    for te in tfidf_exp:
        p_te = tf6.add_paragraph()
        p_te.text = "• " + te
        p_te.font.size = Pt(13)
        p_te.font.color.rgb = COLOR_TEXT_DARK
        p_te.space_after = Pt(8)

    p_leak = tf6.add_paragraph()
    p_leak.text = "Strict Data Leakage Prevention (Pipeline Architecture):"
    p_leak.font.bold = True
    p_leak.font.size = Pt(17)
    p_leak.font.color.rgb = COLOR_PRIMARY
    p_leak.space_after = Pt(8)

    leak_exp = [
        "Common Flaw: Fitting the TF-IDF vectorizer on the full dataset before splitting leaks test vocabulary and IDF weights into the training phase.",
        "Our Solution: We encapsulated vectorizer and models in Scikit-Learn Pipeline objects.",
        "The vectorizer is fit EXCLUSIVELY on X_train (8,000 samples). The 2,000 test reviews are transformed out-of-sample.",
        "Guarantees that test evaluations strictly reflect real-world unseen inference capability."
    ]
    for le in leak_exp:
        p_le = tf6.add_paragraph()
        p_le.text = "• " + le
        p_le.font.size = Pt(13)
        p_le.font.color.rgb = COLOR_TEXT_DARK
        p_le.space_after = Pt(8)

    slide6.notes_slide.notes_text_frame.text = (
        "In Slide 6, we examine our mathematical feature representation. We use Sublinear TF-IDF with unigrams and "
        "bigrams up to 5,000 features. Most importantly, to prevent data leakage, we fit the vectorizer strictly on "
        "the 8,000 training reviews using a Scikit-Learn Pipeline. The test split is never seen by the vectorizer during "
        "vocabulary and IDF fitting."
    )

    # =========================================================================
    # SLIDE 7: THE FOUR REQUIRED CLASSIFIERS
    # =========================================================================
    slide7 = prs.slides.add_slide(blank_layout)
    add_header(slide7, "6. The Four Contending Classifiers")

    boxes = [
        ("Multinomial Naive Bayes", "Probabilistic Generative", "Assumes feature conditional independence given class label: P(c|d) proportional to P(c) * product(P(t|c)). Highly effective on text bag-of-words. Closed-form training.", "#2ca02c"),
        ("Linear Support Vector Machine", "Maximum Margin Hyperplane", "Solves convex quadratic optimization to locate the hyperplane maximizing geometric margin: min ||w||^2 subject to y_i(w^T x_i + b) >= 1. Dominates high-dimensional text.", "#1f77b4"),
        ("Decision Tree Classifier", "Recursive Information Gain", "Constructs a greedy binary tree using Gini impurity splits on individual word features. Non-parametric and highly interpretable, but vulnerable to variance in sparse text.", "#d62728"),
        ("K-Nearest Neighbours", "Instance-Based Lazy Learner", "Classifies reviews based on majority sentiment of k closest vectors in cosine metric space. Does not learn an explicit parametric boundary; computation deferred to query time.", "#ff7f0e")
    ]

    for idx, (m_name, paradigm, desc, col_hex) in enumerate(boxes):
        col_idx = idx % 2
        row_idx = idx // 2
        left = Inches(0.8 + col_idx * 5.9)
        top = Inches(1.5 + row_idx * 2.7)

        c = slide7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, Inches(5.6), Inches(2.4))
        c.fill.solid()
        c.fill.fore_color.rgb = COLOR_CARD_BG
        c.line.color.rgb = RGBColor.from_string(col_hex[1:])
        c.line.width = Pt(2)
        tf_c = c.text_frame
        tf_c.word_wrap = True

        p = tf_c.paragraphs[0]
        p.text = m_name
        p.font.bold = True
        p.font.size = Pt(16)
        p.font.color.rgb = RGBColor.from_string(col_hex[1:])

        p_sub = tf_c.add_paragraph()
        p_sub.text = f"Paradigm: {paradigm}"
        p_sub.font.bold = True
        p_sub.font.size = Pt(11)
        p_sub.font.color.rgb = COLOR_TEXT_DARK
        p_sub.space_after = Pt(4)

        p_d = tf_c.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(11)
        p_d.font.color.rgb = COLOR_TEXT_DARK

    slide7.notes_slide.notes_text_frame.text = (
        "Here we introduce the four contending classifiers. We deliberately contrast four fundamentally different "
        "algorithmic philosophies: a probabilistic generative model (Multinomial Naive Bayes), a margin-maximizing "
        "discriminative model (Linear SVM), an axis-parallel recursive partitioner (Decision Tree), and an instance-based "
        "metric space searcher (KNN)."
    )

    # =========================================================================
    # SLIDE 8: EMPIRICAL BENCHMARK RESULTS TABLE
    # =========================================================================
    slide8 = prs.slides.add_slide(blank_layout)
    add_header(slide8, "7. Empirical Benchmark: Actual Comparative Results")

    # Table of Genuine Results
    rows = 5
    cols = 7
    left = Inches(0.8)
    top = Inches(1.6)
    width = Inches(11.7)
    height = Inches(3.2)

    table_shape = slide8.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table

    headers = ["Classifier Model", "Accuracy", "Precision", "Recall", "Macro F1", "Train Time", "Latency / Query"]
    col_widths = [Inches(3.2), Inches(1.3), Inches(1.3), Inches(1.3), Inches(1.3), Inches(1.5), Inches(1.8)]
    for i, w in enumerate(col_widths):
        table.columns[i].width = w

    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_PRIMARY
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.bold = True
        p.font.size = Pt(13)
        p.font.color.rgb = COLOR_TEXT_LIGHT
        p.alignment = PP_ALIGN.CENTER

    data = [
        ["Linear Support Vector Machine", "87.20%", "87.20%", "87.20%", "87.20%", "2.54 s", "0.119 ms"],
        ["Multinomial Naive Bayes", "86.40%", "86.40%", "86.40%", "86.40%", "2.22 s", "0.092 ms"],
        ["K-Nearest Neighbours (k=5)", "77.00%", "77.06%", "77.00%", "76.99%", "2.47 s", "0.302 ms"],
        ["Decision Tree (depth=25)", "69.90%", "70.06%", "69.90%", "69.84%", "4.10 s", "0.107 ms"]
    ]

    for row_idx, row_data in enumerate(data):
        for col_idx, cell_value in enumerate(row_data):
            cell = table.cell(row_idx + 1, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(235, 247, 238) if row_idx == 0 else RGBColor(255, 255, 255)
            p = cell.text_frame.paragraphs[0]
            p.text = cell_value
            p.font.size = Pt(12)
            p.font.bold = (row_idx == 0 or col_idx == 0)
            p.font.color.rgb = COLOR_PRIMARY if row_idx == 0 else COLOR_TEXT_DARK
            p.alignment = PP_ALIGN.LEFT if col_idx == 0 else PP_ALIGN.CENTER

    # Summary Insights Box below table
    s_box = slide8.shapes.add_textbox(Inches(0.8), Inches(5.0), Inches(11.7), Inches(1.8))
    tf_s = s_box.text_frame
    tf_s.word_wrap = True
    p = tf_s.paragraphs[0]
    p.text = "Key Quantitative Findings:"
    p.font.bold = True
    p.font.size = Pt(14)
    p.font.color.rgb = COLOR_PRIMARY

    bullets_res = [
        "Winner: Linear SVM achieved the highest performance (87.20% Accuracy, 87.20% F1), confirming theoretical superiority in high-dimensional linear separation.",
        "Speed Champion: Multinomial Naive Bayes achieved 0.092 ms inference latency per query with near-peak accuracy (86.40%).",
        "Efficiency Penalty: KNN's test latency was 3.3x slower (0.302 ms) than Naive Bayes due to pairwise metric searches across 8,000 vectors.",
        "Decision Tree Deficit: Decision Tree lagged at 69.90%, requiring the longest training time (4.10 s) to build axis-parallel splits."
    ]
    for br in bullets_res:
        p_br = tf_s.add_paragraph()
        p_br.text = "• " + br
        p_br.font.size = Pt(12)
        p_br.font.color.rgb = COLOR_TEXT_DARK

    slide8.notes_slide.notes_text_frame.text = (
        "Here are our actual, genuine empirical results on 2,000 test reviews. Linear SVM wins with 87.20% accuracy "
        "and 87.20% macro F1. Multinomial Naive Bayes is an extremely close second at 86.40% and is the fastest at "
        "0.092 milliseconds per review. KNN achieves 77.00% but suffers in query latency. Decision Tree finishes at "
        "69.90%, showing the limitations of axis-parallel splits on text."
    )

    # =========================================================================
    # SLIDE 9: PERFORMANCE VISUALIZATIONS
    # =========================================================================
    slide9 = prs.slides.add_slide(blank_layout)
    add_header(slide9, "8. Performance Graphs & Visual Comparisons")

    if os.path.exists('visualizations/accuracy_f1_comparison.png'):
        slide9.shapes.add_picture('visualizations/accuracy_f1_comparison.png', Inches(0.8), Inches(1.5), width=Inches(5.7))

    if os.path.exists('visualizations/training_inference_time.png'):
        slide9.shapes.add_picture('visualizations/training_inference_time.png', Inches(6.8), Inches(1.5), width=Inches(5.7))

    slide9.notes_slide.notes_text_frame.text = (
        "These figures visually demonstrate the performance landscape. On the left, we observe the clear dominance "
        "of Linear SVM and Naive Bayes in Accuracy and F1. On the right, we plot the computational tradeoff: Decision Tree "
        "takes the most time to train, while KNN requires significantly more inference time per sample."
    )

    # =========================================================================
    # SLIDE 10: CONFUSION MATRIX & DIAGNOSTICS
    # =========================================================================
    slide10 = prs.slides.add_slide(blank_layout)
    add_header(slide10, "9. Multi-Panel Confusion Matrix Analysis")

    if os.path.exists('visualizations/confusion_matrices.png'):
        slide10.shapes.add_picture('visualizations/confusion_matrices.png', Inches(0.8), Inches(1.4), width=Inches(6.2))

    t_diag = slide10.shapes.add_textbox(Inches(7.2), Inches(1.5), Inches(5.3), Inches(5.2))
    tf_d = t_diag.text_frame
    tf_d.word_wrap = True

    p = tf_d.paragraphs[0]
    p.text = "Error Diagnostic Insights:"
    p.font.bold = True
    p.font.size = Pt(17)
    p.font.color.rgb = COLOR_PRIMARY
    p.space_after = Pt(10)

    diag_points = [
        "Test Split Integrity: Evaluated on exactly 2,000 unseen reviews (1,000 negative, 1,000 positive).",
        "Linear SVM Equilibrium: Correctly identified 873 True Negatives and 871 True Positives with minimal false alarms (127 FP, 129 FN).",
        "Naive Bayes Consistency: Achieved 859 TN and 869 TP. Shows balanced error distribution without skew toward either class.",
        "KNN Degradation: Incurred 254 False Positives and 206 False Negatives, demonstrating boundary blurring in high dimensions.",
        "Decision Tree Fragmentation: Suffered 345 False Positives and 257 False Negatives; the tree struggled to distinguish negative nuances from positive praise."
    ]
    for dp in diag_points:
        p_dp = tf_d.add_paragraph()
        p_dp.text = "• " + dp
        p_dp.font.size = Pt(12)
        p_dp.font.color.rgb = COLOR_TEXT_DARK
        p_dp.space_after = Pt(8)

    slide10.notes_slide.notes_text_frame.text = (
        "Looking at the confusion matrices on 2,000 test reviews: Linear SVM and Naive Bayes show remarkable symmetry "
        "between false positives and false negatives. Linear SVM achieves 873 True Negatives and 871 True Positives. "
        "In contrast, Decision Tree misclassifies 602 reviews out of 2,000 due to feature fragmentation."
    )

    # =========================================================================
    # SLIDE 11: WHAT-IF ABLATION EXPERIMENTS
    # =========================================================================
    slide11 = prs.slides.add_slide(blank_layout)
    add_header(slide11, "10. Empirical What-If Ablation Experiments")

    if os.path.exists('visualizations/what_if_analysis.png'):
        slide11.shapes.add_picture('visualizations/what_if_analysis.png', Inches(0.8), Inches(1.5), width=Inches(11.7))

    slide11.notes_slide.notes_text_frame.text = (
        "For Requirement 7, we performed three verified What-If experiments. In Plot A, we varied the TF-IDF vocabulary "
        "size from 1,000 to 10,000 features. In Plot B, we compared unigrams versus bigrams. In Plot C, we tuned the "
        "hyperparameters: SVM C, Naive Bayes alpha, and KNN k."
    )

    # =========================================================================
    # SLIDE 12: WHAT-IF DETAILED FINDINGS
    # =========================================================================
    slide12 = prs.slides.add_slide(blank_layout)
    add_header(slide12, "11. What-If Analysis: Observed Effects & Takeaways")

    t_box12 = slide12.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.2))
    tf12 = t_box12.text_frame
    tf12.word_wrap = True

    experiments = [
        ("Experiment 1: TF-IDF Vocabulary Size (max_features = 1k, 2.5k, 5k, 10k)",
         "• Expanding vocabulary from 1,000 to 5,000 features improved Linear SVM accuracy from 84.1% to 87.2% (+3.1% gain).\n"
         "• Expanding from 5,000 to 10,000 produced diminishing returns (+0.2% gain) while doubling vector memory footprints.\n"
         "• Conclusion: 5,000 features represents the Pareto-optimal frontier between accuracy and computational footprint."),
        
        ("Experiment 2: N-gram Feature Space (Unigrams vs Unigrams+Bigrams vs Bigrams Only)",
         "• Unigrams Only (1, 1): Reached 85.8% on SVM. Missed multi-word contextual phrases.\n"
         "• Unigram + Bigrams (1, 2): Yielded 87.2% accuracy. Successfully captured compound phrases ('not good', 'must see').\n"
         "• Bigrams Only (2, 2): Plunged to 78.5% due to severe matrix sparsity and absence of single-word anchor tokens."),
        
        ("Experiment 3: Hyperparameter Regularization Curves",
         "• Linear SVM C Sensitivity: Optimal at C=1.0 (87.2%). C=0.01 underfits (82.1%); C=10.0 overfits margins (86.4%).\n"
         "• Naive Bayes Alpha Smoothing: Laplace smoothing at alpha=1.0 prevents zero-probability traps; alpha=0.01 yielded 85.1%.\n"
         "• KNN Neighborhood Size: k=5 was optimal (77.0%). Increasing to k=15 caused boundary over-smoothing (75.8%).")
    ]

    for title, desc in experiments:
        p_t = tf12.add_paragraph() if tf12.paragraphs[0].text else tf12.paragraphs[0]
        p_t.text = title
        p_t.font.bold = True
        p_t.font.size = Pt(15)
        p_t.font.color.rgb = COLOR_PRIMARY
        p_t.space_after = Pt(4)

        p_d = tf12.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(12)
        p_d.font.color.rgb = COLOR_TEXT_DARK
        p_d.space_after = Pt(12)

    slide12.notes_slide.notes_text_frame.text = (
        "In this slide, I explain the observed empirical effects. Expanding vocabulary size from 1,000 to 5,000 gives "
        "a substantial 3.1% accuracy boost, after which returns plateau. Combining unigrams with bigrams is essential "
        "because bigrams capture critical phrases like 'not recommend', improving performance over unigrams alone. "
        "Bigrams alone, however, fail due to severe feature sparsity."
    )

    # =========================================================================
    # SLIDE 13: TECHNICAL DEFENSE: WHY DID SVM WIN?
    # =========================================================================
    slide13 = prs.slides.add_slide(blank_layout)
    add_header(slide13, "12. Technical Defense: Algorithmic Justification")

    c_def1 = slide13.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.2))
    c_def1.fill.solid()
    c_def1.fill.fore_color.rgb = COLOR_CARD_BG
    c_def1.line.color.rgb = COLOR_PRIMARY
    tf_def1 = c_def1.text_frame
    tf_def1.word_wrap = True

    p = tf_def1.paragraphs[0]
    p.text = "Why Support Vector Machine Won (87.20%)"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = COLOR_PRIMARY
    p.space_after = Pt(8)

    svm_wins = [
        "Linear Separability in High Dimensions: Joachims (1998) proved that textual document vectors in 5,000+ dimensions are nearly linearly separable.",
        "Maximum Margin Regularization: SVM maximizes the margin (2 / ||w||) between support vectors, preventing overfitting even when features exceed sample counts.",
        "Dense Feature Aggregation: All non-zero words contribute to the inner product w^T x, enabling holistic review understanding."
    ]
    for sw in svm_wins:
        p_sw = tf_def1.add_paragraph()
        p_sw.text = "• " + sw
        p_sw.font.size = Pt(12)
        p_sw.font.color.rgb = COLOR_TEXT_DARK
        p_sw.space_after = Pt(8)

    c_def2 = slide13.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.2))
    c_def2.fill.solid()
    c_def2.fill.fore_color.rgb = COLOR_CARD_BG
    c_def2.line.color.rgb = COLOR_PRIMARY
    tf_def2 = c_def2.text_frame
    tf_def2.word_wrap = True

    p = tf_def2.paragraphs[0]
    p.text = "Why Decision Trees (69.90%) and KNN (77.00%) Lagged"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = COLOR_PRIMARY
    p.space_after = Pt(8)

    lag_reasons = [
        "Decision Tree Fragmentation: Decision trees perform single-variable orthogonal splits (e.g., if 'bad' <= 0.05). In sparse text where each review has only ~50 active words, individual splits fragment the data, leading to severe overfitting or under-fitting.",
        "Curse of Dimensionality in KNN: In 5,000-dimensional space, the ratio of distance to the nearest neighbor versus farthest neighbor approaches 1 (distance concentration), degrading neighbor discrimination and increasing test latency."
    ]
    for lr in lag_reasons:
        p_lr = tf_def2.add_paragraph()
        p_lr.text = "• " + lr
        p_lr.font.size = Pt(12)
        p_lr.font.color.rgb = COLOR_TEXT_DARK
        p_lr.space_after = Pt(8)

    slide13.notes_slide.notes_text_frame.text = (
        "This is our core technical defense slide. Why did SVM win? As established in foundational Information Retrieval "
        "literature by Thorsten Joachims, high-dimensional text vectors are inherently linearly separable. SVM finds "
        "the global maximum-margin hyperplane using all words simultaneously. Decision Trees fail because text sentiment "
        "cannot be effectively modeled by single-word axis-parallel cuts, and KNN suffers from distance concentration "
        "in high-dimensional vector spaces."
    )

    # =========================================================================
    # SLIDE 14: CONCLUSION, LIMITATIONS & FUTURE WORK
    # =========================================================================
    slide14 = prs.slides.add_slide(blank_layout)
    add_header(slide14, "13. Conclusion, Limitations & Future Work")

    t_box14 = slide14.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.2))
    tf14 = t_box14.text_frame
    tf14.word_wrap = True

    concl_points = [
        ("Project Conclusions:", [
            "Successfully built and defended an end-to-end sentiment classification system on 10,000 authentic IMDb reviews.",
            "Linear SVM (87.20%) and Multinomial Naive Bayes (86.40%) proved to be the most viable production classifiers.",
            "Demonstrated the crucial importance of negation-preserving preprocessing and strict data leakage hygiene."
        ]),
        ("Technical Limitations:", [
            "Lack of Word Order Semantics: Bag-of-words TF-IDF representations ignore long-range syntax and syntactic dependencies.",
            "Domain Specificity: Cinematic vocabulary does not immediately transfer to technical or e-commerce products without retraining."
        ]),
        ("Future Roadmap:", [
            "Aspect-Based Sentiment Analysis (ABSA): Classifying specific aspects (acting, cinematography, sound, pacing) independently.",
            "Pretrained Transformers: Fine-tuning BERT / RoBERTa architectures with contextual self-attention for complex sarcastic constructs."
        ])
    ]

    for title, bullets in concl_points:
        p_t = tf14.add_paragraph() if tf14.paragraphs[0].text else tf14.paragraphs[0]
        p_t.text = title
        p_t.font.bold = True
        p_t.font.size = Pt(15)
        p_t.font.color.rgb = COLOR_PRIMARY
        p_t.space_after = Pt(4)

        for b in bullets:
            p_b = tf14.add_paragraph()
            p_b.text = "• " + b
            p_b.font.size = Pt(12)
            p_b.font.color.rgb = COLOR_TEXT_DARK
            p_b.space_after = Pt(4)
        p_b.space_after = Pt(10)

    slide14.notes_slide.notes_text_frame.text = (
        "In conclusion, this project provides a rigorous, transparent benchmark. Linear SVM is defended as the optimal "
        "production model for text classification. We acknowledge limitations including bag-of-words word order loss, "
        "and outline future enhancements such as aspect-based sentiment analysis and transformer models like BERT."
    )

    # =========================================================================
    # SLIDE 15: REFERENCES & DEFENSE Q&A
    # =========================================================================
    slide15 = prs.slides.add_slide(blank_layout)
    add_header(slide15, "14. Scholarly References & Technical Q&A")

    t_box15 = slide15.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.2))
    tf15 = t_box15.text_frame
    tf15.word_wrap = True

    p = tf15.paragraphs[0]
    p.text = "Scholarly References:"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = COLOR_PRIMARY
    p.space_after = Pt(8)

    refs = [
        "1. Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning word vectors for sentiment analysis. In Proceedings of ACL (pp. 142-150).",
        "2. Joachims, T. (1998). Text categorization with support vector machines: Learning with many relevant features. In ECML (pp. 137-142). Springer.",
        "3. Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. Information Processing & Management, 24(5), 513-523.",
        "4. Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval. Cambridge University Press.",
        "5. Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273-297."
    ]
    for r in refs:
        p_r = tf15.add_paragraph()
        p_r.text = r
        p_r.font.size = Pt(11)
        p_r.font.color.rgb = COLOR_TEXT_DARK
        p_r.space_after = Pt(6)

    p_qa = tf15.add_paragraph()
    p_qa.text = "\nThank You! The Technical Defense is Open for Questions."
    p_qa.font.bold = True
    p_qa.font.size = Pt(16)
    p_qa.font.color.rgb = COLOR_PRIMARY
    p_qa.alignment = PP_ALIGN.CENTER

    slide15.notes_slide.notes_text_frame.text = (
        "Thank you respected evaluators for your attention. I have grounded my implementation in foundational Information "
        "Retrieval literature, from Salton and Buckley's TF-IDF formulation to Joachims' work on Support Vector Machines. "
        "I welcome any technical questions regarding my methodology, mathematics, or code implementation."
    )

    pptx_path = 'presentation/Sentiment_Classification_Presentation.pptx'
    os.makedirs(os.path.dirname(pptx_path), exist_ok=True)
    prs.save(pptx_path)
    print(f"Presentation successfully generated and saved to {pptx_path} (15 Slides with Speaker Notes)")

if __name__ == '__main__':
    create_presentation()

