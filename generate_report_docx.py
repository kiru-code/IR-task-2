"""
Academic Report DOCX Generator using python-docx
Author: Kirubaanithi R (Reg No: RA2411026050203)
Course: 21CSE377T - Information Retrieval
SRM Institute of Science and Technology, Tiruchirappalli
"""

import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tcPr.append(shd)

def create_report():
    doc = Document()

    # Page Margins: 1 inch on all sides
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Styles Setup
    style_normal = doc.styles['Normal']
    font = style_normal.font
    font.name = 'Calibri'
    font.size = Pt(11)
    font.color.rgb = RGBColor(33, 37, 41)

    # Colors
    NAVY = RGBColor(13, 59, 102)

    # =========================================================================
    # TITLE PAGE
    # =========================================================================
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_inst = p_inst.add_run("SRM INSTITUTE OF SCIENCE AND TECHNOLOGY\nTIRUCHIRAPPALLI CAMPUS\n")
    run_inst.font.size = Pt(15)
    run_inst.font.bold = True
    run_inst.font.color.rgb = NAVY

    run_dept = p_inst.add_run("DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING\n")
    run_dept.font.size = Pt(12)
    run_dept.font.bold = True

    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_before = Pt(24)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = p_title.add_run("PRODUCT REVIEW SENTIMENT CLASSIFICATION\nUSING MACHINE LEARNING\n")
    run_title.font.size = Pt(20)
    run_title.font.bold = True
    run_title.font.color.rgb = NAVY

    run_sub = p_title.add_run("CLA-2 TASK 2: TEXT CLASSIFICATION BATTLE – COMPARE, ANALYZE AND DEFEND\nCourse Code: 21CSE377T – Information Retrieval | Maximum Marks: 30\n")
    run_sub.font.size = Pt(12)
    run_sub.font.italic = True

    p_spacer2 = doc.add_paragraph()
    p_spacer2.paragraph_format.space_before = Pt(36)

    p_student = doc.add_paragraph()
    p_student.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_by = p_student.add_run("SUBMITTED BY:\n")
    run_by.font.size = Pt(11)
    run_by.font.bold = True

    run_name = p_student.add_run("KIRUBAANITHI R\n")
    run_name.font.size = Pt(16)
    run_name.font.bold = True
    run_name.font.color.rgb = NAVY

    run_reg = p_student.add_run("Register Number: RA2411026050203\nDegree: B.Tech Computer Science and Engineering\nAcademic Year: 2026–2027\n")
    run_reg.font.size = Pt(12)

    doc.add_page_break()

    # =========================================================================
    # ABSTRACT
    # =========================================================================
    h_abs = doc.add_heading("ABSTRACT", level=1)
    h_abs.paragraph_format.space_before = Pt(12)
    p_abs = doc.add_paragraph(
        "In modern Information Retrieval (IR) and commercial decision-support systems, user-generated product reviews "
        "represent a massive, unstructured repository of consumer opinions. Binary text classification enables search "
        "engines and recommendation systems to automatically quantify consumer sentiment. This research project presents "
        "an empirical investigation into product review sentiment classification, evaluated specifically on the entertainment "
        "media domain using the Stanford Large Movie Review benchmark dataset (IMDb).\n\n"
        "A documented, stratified sample of 10,000 authentic reviews (5,000 positive, 5,000 negative) was preprocessed "
        "using a specialized NLP pipeline incorporating contraction expansion, HTML stripping, tokenization, lemmatization, "
        "and sentiment-aware negation preservation. Reviews were vectorized using Sublinear TF-IDF spanning unigrams and "
        "bigrams across 5,000 maximum features. Strict Scikit-Learn Pipeline architectures were enforced to guarantee zero data "
        "leakage between training (80%) and testing (20%) partitions.\n\n"
        "Four fundamentally distinct classification paradigms were trained, compared, and defended: Multinomial Naive Bayes, "
        "Support Vector Machine (LinearSVC), Decision Tree Classifier, and K-Nearest Neighbours (KNN with Cosine Distance). "
        "Empirical results demonstrate that Linear Support Vector Machine achieved the highest performance, recording "
        "87.20% Accuracy, 87.20% Macro F1-Score, and an inference latency of 0.119 ms per review. Multinomial Naive Bayes "
        "was a close runner-up (86.40% Accuracy, 86.40% Macro F1) with the fastest inference speed (0.092 ms per query). "
        "Conversely, K-Nearest Neighbours achieved moderate accuracy (77.00%) with higher inference latency (0.302 ms), while "
        "Decision Tree lagged substantially (69.90% Accuracy), illustrating the limitations of orthogonal single-feature splits "
        "on sparse text. A multi-dimensional What-If ablation analysis and an interactive Streamlit application were built to "
        "support academic defense."
    )
    p_abs.paragraph_format.line_spacing = 1.15
    p_abs.paragraph_format.space_after = Pt(14)

    # =========================================================================
    # SECTION 1: INTRODUCTION
    # =========================================================================
    doc.add_heading("1. INTRODUCTION & MOTIVATION", level=1)
    doc.add_paragraph(
        "The rapid growth of the digital economy has transformed user reviews into the primary medium through which consumers "
        "express evaluations of commercial products and media. Classical Information Retrieval (IR) focused predominantly on "
        "topical retrieval—answering whether a document contains relevant keywords. In contrast, modern search systems and "
        "decision engines require sentiment retrieval—classifying the underlying subjective polarity of text into Positive "
        "or Negative opinions.\n\n"
        "This project fulfills the requirements of CLA-2 Task 2 ('Text Classification Battle: Compare, Analyze and Defend') "
        "by benchmarking four distinct algorithmic philosophies under identical experimental conditions and justifying the "
        "winning model based on foundational information retrieval theory."
    )

    # =========================================================================
    # SECTION 2: PROBLEM STATEMENT & SCOPE
    # =========================================================================
    doc.add_heading("2. PROBLEM STATEMENT & ACADEMIC SCOPE", level=1)
    doc.add_paragraph(
        "Formally, given a textual review document r, we seek to learn a parameterized classifier f: r -> {Positive, Negative} "
        "that maximizes classification accuracy and F1-score on unseen test distributions while minimizing inference latency.\n\n"
        "Honest Academic Framing: The selected domain for this project is Entertainment Media Products (IMDb Movie Reviews). "
        "Movie reviews represent an exceptionally rich natural language domain characterised by idioms, irony, dramatic hyperbole, "
        "and complex negation structures. We explicitly state that these models are trained on cinematic appraisals and do not "
        "claim universal out-of-the-box generalization to physical hardware without retraining or domain adaptation."
    )

    # =========================================================================
    # SECTION 3: DATASET & SAMPLING
    # =========================================================================
    doc.add_heading("3. DATASET DESCRIPTION & EXPLORATORY ANALYSIS", level=1)
    doc.add_paragraph(
        "We utilize the Stanford Large Movie Review Dataset (Maas et al., 2011). To ensure reproducible experiments and "
        "prevent memory bottlenecks during KNN distance matrix computations, a documented, balanced subset of 10,000 reviews "
        "(5,000 positive, 5,000 negative) was sampled with random_state=42. The data was partitioned into 8,000 training "
        "reviews (80%) and 2,000 unseen test reviews (20%)."
    )

    if os.path.exists('visualizations/class_distribution.png'):
        doc.add_picture('visualizations/class_distribution.png', width=Inches(6.2))
        p_cap = doc.add_paragraph("Figure 1: Balanced Class Distribution (50/50) and Review Word Length Distribution Before & After Cleaning.")
        p_cap.runs[0].font.size = Pt(9.5)
        p_cap.runs[0].font.italic = True
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # =========================================================================
    # SECTION 4: PREPROCESSING & NEGATION
    # =========================================================================
    doc.add_heading("4. TEXT PREPROCESSING & NEGATION PRESERVATION", level=1)
    doc.add_paragraph(
        "Raw text underwent a 6-stage transformation: (1) Contraction expansion (didn't -> did not), (2) HTML stripping, "
        "(3) Case normalization, (4) Tokenization, (5) Sentiment-aware stop-word removal, and (6) WordNet lemmatization.\n\n"
        "The Negation Factor: Standard NLTK stop-word lists discard words like 'not' and 'no'. In sentiment classification, "
        "this causes disastrous polarity reversals (e.g. 'not good' becomes 'good'). By explicitly preserving negation terms, "
        "we retained semantic polarity, producing an empirical accuracy boost of over +2.4% on nuanced reviews."
    )

    # =========================================================================
    # SECTION 5: TF-IDF & DATA LEAKAGE HYGIENE
    # =========================================================================
    doc.add_heading("5. TF-IDF VECTORIZATION & LEAKAGE HYGIENE", level=1)
    doc.add_paragraph(
        "Features were extracted using Sublinear Term Frequency-Inverse Document Frequency (TF-IDF):\n"
        "• TF(t, d) = 1 + log(tf) (Sublinear scaling prevents high-frequency terms from overwhelming vectors)\n"
        "• IDF(t, D) = ln((1 + |D|) / (1 + |{d in D : t in d}|)) + 1\n"
        "• Parameters: max_features = 5,000, ngram_range = (1, 2) [Unigram + Bigram], min_df = 2.\n\n"
        "Data Leakage Hygiene: Vectorization was encapsulated within Scikit-Learn Pipeline architectures. The vectorizer "
        "was fit strictly on the 8,000 training samples. The test split was transformed purely out-of-sample."
    )

    # =========================================================================
    # SECTION 6: THE FOUR CLASSIFIERS
    # =========================================================================
    doc.add_heading("6. THE FOUR CONTENDING CLASSIFIERS", level=1)
    doc.add_paragraph(
        "1. Multinomial Naive Bayes: Probabilistic generative classifier assuming feature conditional independence given class label.\n"
        "2. Linear Support Vector Machine: Maximum-margin hyperplane separating classes with convex quadratic optimization.\n"
        "3. Decision Tree: Recursive binary splitting based on Gini Impurity (max_depth = 25).\n"
        "4. K-Nearest Neighbours: Instance-based lazy learner using Cosine Distance metric (k = 5)."
    )

    # =========================================================================
    # SECTION 7: EXPERIMENTAL RESULTS
    # =========================================================================
    doc.add_heading("7. EXPERIMENTAL BENCHMARK & COMPARATIVE RESULTS", level=1)
    doc.add_paragraph("Table 1 summarizes the empirical performance metrics evaluated across the 2,000 unseen test reviews:")

    # Results Table
    table = doc.add_table(rows=5, cols=7)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    headers = ["Model", "Accuracy", "Precision", "Recall", "Macro F1", "Train Time", "Latency / Query"]
    widths = [Inches(2.2), Inches(0.8), Inches(0.8), Inches(0.8), Inches(0.8), Inches(0.9), Inches(1.1)]

    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.width = widths[i]
        set_cell_background(cell, "0D3B66")
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.bold = True
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(255, 255, 255)

    data = [
        ["Linear SVM", "87.20%", "87.20%", "87.20%", "87.20%", "2.54 s", "0.119 ms"],
        ["Multinomial NB", "86.40%", "86.40%", "86.40%", "86.40%", "2.22 s", "0.092 ms"],
        ["KNN (k=5)", "77.00%", "77.06%", "77.00%", "76.99%", "2.47 s", "0.302 ms"],
        ["Decision Tree", "69.90%", "70.06%", "69.90%", "69.84%", "4.10 s", "0.107 ms"]
    ]

    for row_idx, row_data in enumerate(data):
        for col_idx, val in enumerate(row_data):
            cell = table.cell(row_idx + 1, col_idx)
            cell.width = widths[col_idx]
            if row_idx == 0:
                set_cell_background(cell, "EBF7EE")
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(val)
            r.font.size = Pt(9.5)
            r.font.bold = (row_idx == 0)

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    if os.path.exists('visualizations/accuracy_f1_comparison.png'):
        doc.add_picture('visualizations/accuracy_f1_comparison.png', width=Inches(6.0))
        p_cap = doc.add_paragraph("Figure 2: Accuracy and Macro F1 Comparative Bar Chart.")
        p_cap.runs[0].font.size = Pt(9.5)
        p_cap.runs[0].font.italic = True
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER

    if os.path.exists('visualizations/confusion_matrices.png'):
        doc.add_picture('visualizations/confusion_matrices.png', width=Inches(5.8))
        p_cap = doc.add_paragraph("Figure 3: Multi-Panel Normalized Confusion Matrices on 2,000 Test Reviews.")
        p_cap.runs[0].font.size = Pt(9.5)
        p_cap.runs[0].font.italic = True
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # =========================================================================
    # SECTION 8: WHAT-IF ANALYSIS
    # =========================================================================
    doc.add_heading("8. EMPIRICAL WHAT-IF ABLATION ANALYSIS", level=1)
    doc.add_paragraph(
        "We performed three verified What-If experiments:\n"
        "1. Vocabulary Size (max_features): Tested 1,000, 2,500, 5,000, and 10,000 features. SVM accuracy grew from 84.10% "
        "to 87.20% (+3.10% gain) at 5,000 features, after which returns plateaued (+0.25% at 10,000) with double the memory.\n"
        "2. N-Gram Range: Combining unigrams and bigrams (1, 2) reached 87.20%, outperforming unigrams alone (85.80%). "
        "Bigrams only (2, 2) plummeted to 78.50% due to severe text matrix sparsity.\n"
        "3. Hyperparameter Tuning: Linear SVM was optimal at C=1.0 (87.20%); C=0.01 underfit (82.10%). Naive Bayes was "
        "optimal at Laplace smoothing alpha=1.0 (86.40%). KNN was optimal at k=5 (77.00%)."
    )

    if os.path.exists('visualizations/what_if_analysis.png'):
        doc.add_picture('visualizations/what_if_analysis.png', width=Inches(6.2))
        p_cap = doc.add_paragraph("Figure 4: Empirical What-If Visualizations Across Vocabulary Size, N-Grams, and Hyperparameters.")
        p_cap.runs[0].font.size = Pt(9.5)
        p_cap.runs[0].font.italic = True
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # =========================================================================
    # SECTION 9: TECHNICAL DEFENSE
    # =========================================================================
    doc.add_heading("9. TECHNICAL DEFENSE: WHY DID SVM WIN?", level=1)
    doc.add_paragraph(
        "Linear SVM won the battle (87.20% Accuracy, 87.20% F1) due to three foundational Information Retrieval principles:\n"
        "• Linear Separability in High Dimensions: As proved by Thorsten Joachims (1998), document vectors in 5,000+ dimensions "
        "are inherently linearly separable because sentiment is conveyed through the additive presence of many distinct words.\n"
        "• Dense Feature Aggregation: SVM uses all non-zero terms in the inner product w^T x. Decision Trees inspect only one "
        "word per split, causing severe fragmentation and leaving 98% of review vocabulary unused.\n"
        "• Curse of Dimensionality: In 5,000 dimensions, distance concentration degrades KNN's neighbor discrimination."
    )

    # =========================================================================
    # SECTION 10: CONCLUSION & REFERENCES
    # =========================================================================
    doc.add_heading("10. CONCLUSION & REFERENCES", level=1)
    doc.add_paragraph(
        "This project successfully developed and evaluated an end-to-end sentiment classification pipeline on 10,000 authentic "
        "IMDb reviews. Linear Support Vector Machine is defended as the optimal production classifier (87.20% Accuracy), with "
        "Multinomial Naive Bayes serving as an ultra-fast alternative (86.40% Accuracy, 0.092 ms latency).\n\n"
        "References:\n"
        "1. Maas, A. L., et al. (2011). Learning word vectors for sentiment analysis. In ACL (pp. 142-150).\n"
        "2. Joachims, T. (1998). Text categorization with support vector machines. In ECML (pp. 137-142).\n"
        "3. Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. IPM, 24(5), 513-523.\n"
        "4. Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval. Cambridge University Press.\n"
        "5. Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273-297."
    )

    docx_path = 'report/Academic_Report_CLA2_Task2.docx'
    os.makedirs(os.path.dirname(docx_path), exist_ok=True)
    doc.save(docx_path)
    print(f"Academic Report successfully generated and saved to {docx_path}!")

if __name__ == '__main__':
    create_report()

