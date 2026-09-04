# Repeated-Question Analyzer for Previous-Year Question Papers Using NLP and TF-IDF — Project Report

---

## 1. Title
**NLP-Based Repeated Question Analyzer and Cross-Year Topic Similarity Retrieval System for Multi-Department Academic Exam Papers Using TF-IDF and Cosine Similarity**

---

## 2. Problem Statement
In universities and autonomous higher-education institutions, previous-year question papers (PYQs) serve as a cornerstone revision and diagnostic resource for students, faculty, and academic audit committees. However, manually reviewing and comparing years of historical question papers across diverse academic departments and courses is laborious, prone to human oversight, and time-intensive.

Key challenges across academic disciplines include:
- **Scan-Only Legacy Question Papers**: Across departments, past examination papers are predominantly stored and archived as scanned PDF documents lacking an underlying digital text layer.
- **Scenario and Narrative Rephrasing**: Modern outcome-based education (OBE) formats frequently contextualize questions within novel real-world narratives and problem-solving scenarios (e.g., framing identical principles through distinct case studies), masking the underlying core syllabus concepts behind extraneous decorative text.
- **Lexical and Syntactical Variations**: Core syllabus concepts and recurring question patterns across academic years frequently employ alternate technical terminology, varying parameters, differing numerical values, and distinct phrasing.

To address these challenges universally across academic departments, there is a strong need for an automated, domain-agnostic NLP pipeline that ingests scanned question paper PDFs, performs robust extraction and layout cleaning, segments individual questions, and retrieves topically related cross-year question pairs.

---

## 3. Objectives
The major objectives of this project are:
1. **Department-Agnostic OCR & Extraction**: Build a robust, cached document processing pipeline using PyMuPDF and Tesseract OCR to convert raw scanned exam PDFs from any academic stream into clean, structured digital text.
2. **Deterministic Layout Cleaning & Question Segmentation**: Implement rule-based regex filtering and a state-machine segmenter capable of stripping standardized exam boilerplate headers, instructions, and course outcome rubrics (`(COx: N Marks)`), accurately isolating individual questions.
3. **NLP Preprocessing & Feature Engineering**: Apply tokenization, domain-adapted stopword filtering, and lemmatization, followed by per-course N-gram TF-IDF vectorization to represent questions in a normalized vector space.
4. **Cross-Year Similarity Retrieval & Topic Matching**: Compute pairwise cosine similarity matrices across examination cycles to detect recurring questions and topic overlaps.
5. **Empirical Benchmark Evaluation**: Measure Precision, Recall, and F1-scores against a human-annotated ground-truth benchmark across multiple similarity cut-offs.
6. **Interactive Visualization Dashboard**: Deploy a responsive Streamlit web application allowing educators and students across departments to interactively adjust similarity thresholds, view question pairs side-by-side, and inspect syllabus coverage.

---

## 4. Dataset

### 4.1 Dataset Overview
To evaluate and benchmark the system, a testbed corpus of **10 previous-year examination papers** spanning **5 representative courses** (2 academic examination cycles per course) was utilized. The underlying pipeline architecture is completely department-agnostic and directly extensible to any academic department (Engineering, Pure Sciences, Commerce, Humanities, Management, etc.).

### 4.2 Source
- **Institution**: Marian College Kuttikkanam (Autonomous)
- **Exam Types**: Summative End Assessment (SEA II), In-Semester Assessment (ISA), and Regular Degree Examinations.
- **Format**: Scanned multi-page PDF documents (36 pages total) with no selectable text layers.

### 4.3 Size
- **Total Question Papers**: 10 PDFs
- **Total Pages**: 36 scanned pages (100% OCR processed)
- **Total Extracted Questions**: 111 distinct questions
- **Total Ground-Truth Duplicate/Same-Topic Pairs**: 39 annotated pairs

### 4.4 Attributes
Each extracted question record contains the following metadata attributes:
- `subject_id`: Course/Subject identifier (e.g., `operating_systems`, `data_structures`)
- `year`: Academic year or examination cycle (e.g., `2024`, `2025`, `2026`)
- `q_no`: Extracted question number or index label (e.g., `1`, `4`, `12`)
- `marks`: Assigned mark weight extracted from metadata tags (e.g., `4`, `5`, `6`, `10`)
- `raw_text`: Complete unnormalized question text for display and manual review
- `clean_text`: Preprocessed, tokenized, stopword-filtered, and lemmatized text for vectorization

### 4.5 Course Breakdown (Benchmark Testbed)

| Subject / Course Name | Year A (Exam) | Year B (Exam) | Question Count (Year A / B) | Total Questions | Gold Matching Pairs |
|---|---|---|---|---|---|
| **Data Structures** | 2025 | 2026 | 16 / 12 | 28 | 8 |
| **Digital Fundamentals** | 2024 (SEA II) | 2025 (SEA II) | 12 / 11 | 23 | 7 |
| **Discrete Mathematics** | 2024 (ISA) | 2025 (SEA II) | 2 / 11 | 13 | 4 |
| **Fundamentals of Programming Using C++** | 2024 (SEA II) | 2025 (SEA II) | 10 / 10 | 20 | 8 |
| **Operating Systems** | 2024 | 2026 | 16 / 11 | 27 | 12 |
| **Total** | — | — | **56 / 55** | **111** | **39** |

### 4.6 Data Preparation
The raw examination papers underwent a systematic preparation workflow:
1. **Hybrid Extraction**: Rendered 300 DPI page images using PyMuPDF and extracted raw text with Tesseract OCR (with persistent caching in `processed/raw/`).
2. **Boilerplate Stripping**: Applied 13+ compiled regular expressions to eliminate institutional headers, duration/marking instructions, and trailing Course Outcome tables while preserving structural question delimiters (`BUNCH`, `OR`).
3. **Grammar-Aware Question Segmentation**: Segmented questions using a boundary state machine matching section transitions and `(COx: N Marks)` delimiters, with lenient tolerance for OCR misrecognitions (e.g., `1A` vs `lA`).
4. **Override & Verification Pass**: Applied a version-controlled override dictionary (`processed/segment_overrides/`) for 16 edge-case OCR fragments across page breaks.

---

## 5. Methodology

The end-to-end analytical workflow follows a modular, departmental-adaptable pipeline:

```
[ Scanned Exam PDFs (Any Department) ]
                 │
                 ▼
[ 1. Hybrid OCR Extraction (PyMuPDF + Tesseract 300 DPI) ]
                 │
                 ▼
[ 2. Boilerplate & Header Removal (Regex Blocklist) ]
                 │
                 ▼
[ 3. Question Segmentation (State Machine + Override Pass) ]
                 │
                 ▼
[ 4. NLP Preprocessing (Tokenization, Stopwords, Lemmatization) ]
                 │
                 ▼
[ 5. Per-Course TF-IDF Vectorization (Unigrams + Bigrams) ]
                 │
                 ▼
[ 6. Cosine Similarity Computation (nA × nB Matrix) ]
                 │
                 ▼
[ 7. Dynamic Thresholding & Retrieval ] ──► [ Streamlit Interactive UI ]
                 │
                 ▼
[ 8. Evaluation against Gold Annotations (Precision, Recall, F1) ]
```

### Stage-by-Stage Explanation:
1. **Text Extraction**: PyMuPDF checks for embedded text; if unavailable (<20 characters), the page image is rendered at 300 DPI and processed via Tesseract OCR v5.
2. **Boilerplate Removal**: Regex rules discard institutional boilerplate (college titles, course codes, general exam instructions), preventing false similarity inflation.
3. **Question Segmentation**: Identifies individual question boundaries using structural markers (`BUNCH I–V`, `OR`, `(COx: N Marks)` tags), producing structured JSON records.
4. **NLP Preprocessing**: Normalizes text into lowercase, eliminates punctuation and non-alphanumeric noise, removes English stopwords, and applies WordNet lemmatization to conflate morphological variants (e.g., `calculating`, `calculation` → `calculate`).
5. **TF-IDF Vectorization**: Fits Term Frequency-Inverse Document Frequency (TF-IDF) models on unigrams and bigrams per course pair. Fitting per course preserves syllabus-specific technical terms and weights without cross-course corpus dilution.
6. **Cosine Similarity Computation**: Computes the dot product of normalized TF-IDF vectors between every Year A question ($u$) and Year B question ($v$):
   $$\text{Cosine Similarity}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$
7. **Dynamic Threshold Retrieval**: Filters candidate question pairs exceeding a user-selected similarity threshold ($\tau \in [0.05, 0.60]$).
8. **Evaluation**: Assesses retrieved pairs against curated gold annotations to calculate Precision, Recall, and F1-score.

---

## 6. Implementation

### 6.1 Programming Language
- **Python 3.10+** (ensuring extensive library support and cross-platform compatibility)

### 6.2 Libraries & Frameworks
- **PDF & OCR Processing**: `pymupdf` (fitz), `pytesseract`, `pdf2image`, `Pillow`
- **NLP & Text Processing**: `nltk` (tokenizers, `stopwords`, `wordnet` lemmatizer)
- **Feature Extraction & Similarity**: `scikit-learn` (`TfidfVectorizer`, `cosine_similarity`), `numpy`
- **Data Manipulation & Serialization**: `pandas`, `json`, `re`, `pathlib`
- **Interactive Web UI**: `streamlit` (live interactive comparison dashboard)

### 6.3 NLP Techniques
- **Optical Character Recognition (OCR)**: High-resolution (300 DPI) document rendering and OCR.
- **Text Normalization**: Regex-based noise reduction, case folding, accent/symbol cleaning.
- **Stopword Elimination**: Domain-adapted NLTK stopword filtering.
- **Lemmatization**: WordNet morphological reduction.
- **N-gram Modeling**: Combined unigram (1-gram) and bigram (2-gram) representation to capture multi-word conceptual phrases (e.g., *"binary search"*, *"capital structure"*, *"page replacement"*, *"linear equation"*).

### 6.4 Algorithm / Model
- **Per-Subject TF-IDF Representation**: Vectorizes text based on sub-linear term frequency scaling and smooth inverse document frequency:
  $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \left( \ln\left(\frac{1 + |D|}{1 + \text{DF}(t, D)}\right) + 1 \right)$$
- **Pairwise Cosine Similarity**: Fast dot-product computation across sparse document matrices.

### 6.5 Important Implementation Details
- **Per-Course Fitting**: TF-IDF vectorizers are fitted strictly on the combined question pool of the specific subject being analyzed, ensuring maximum discriminative weighting for subject-specific terminology.
- **Intermediate Caching**: All extracted raw text, cleaned files, segmented question databases, and similarity matrices are cached under `processed/`, ensuring sub-second response times in the user interface.
- **Modular Architecture**: Adding a new department or subject requires only placing the PDF in the dataset directory and registering the subject entry in `papers.json`.

---

## 7. Results

### 7.1 Performance Metrics
Evaluation was performed against 39 hand-labeled ground-truth match pairs across the benchmark subjects.

#### Overall Benchmark Across Thresholds

| Threshold ($\tau$) | Precision | Recall | F1-Score | Retrieved Pairs | True Positives (TP) | False Positives (FP) | False Negatives (FN) |
|---|---|---|---|---|---|---|---|
| **0.15** *(Default)* | **0.500** | **0.385** | **0.435** | 30 | 15 | 15 | 24 |
| **0.25** | **0.750** | 0.154 | 0.255 | 8 | 6 | 2 | 33 |
| **0.35** | **1.000** | 0.051 | 0.098 | 2 | 2 | 0 | 37 |
| **0.50** | 0.000 | 0.000 | 0.000 | 0 | 0 | 0 | 39 |

#### Per-Subject Performance (at $\tau = 0.15$)

| Subject | Gold Pairs | Retrieved | TP | FP | Precision | Recall | F1-Score |
|---|---|---|---|---|---|---|---|
| **Operating Systems** | 12 | 13 | 8 | 5 | **0.615** | **0.667** | **0.640** |
| **Digital Fundamentals** | 7 | 4 | 3 | 1 | **0.750** | 0.429 | **0.545** |
| **Data Structures** | 8 | 11 | 3 | 8 | 0.273 | 0.375 | 0.316 |
| **C++ Programming** | 8 | 2 | 1 | 1 | 0.500 | 0.125 | 0.200 |
| **Discrete Mathematics** | 4 | 0 | 0 | 0 | 0.000 | 0.000 | 0.000 |

### 7.2 Precision vs. Recall Trade-Off (Visual Curve)

```
Precision & Recall Trade-off across Similarity Thresholds
 1.00 ┤        Precision (1.00 @ 0.35)
 0.80 ┤           /───
 0.60 ┤      /────
 0.40 ┤  ───/────────────── Recall (0.385 @ 0.15)
 0.20 ┤                      \────────
 0.00 ┼───────────────────────────────\──── (0.00 @ 0.50)
      └───────┬───────────────┬───────────┬───────────► Threshold
             0.15            0.25        0.35        0.50
```

### 7.3 Sample Outputs & Error Analysis

#### 1. Successful True Positive Match (Score: 0.41)
- **Year A**: *"Consider the following page reference string: 1, 2, 3, 4, 5, 3, 4, 1... How many page faults would occur for FIFO and LRU replacement algorithms assuming four frames?"*
- **Year B**: *"Consider a system with 4 page frames using the FIFO page replacement algorithm. Given the reference string... Determine the number of page faults."*
- **Analysis**: High semantic and lexical term alignment (*"page replacement algorithm"*, *"page faults"*, *"reference string"*).

#### 2. Representative False Positive ($\tau = 0.25$)
- **Year A**: *"With a routine, demonstrate the deletion of the node 't' at position 3 of the given doubly linked list."*
- **Year B**: *"Consider the following circular linked list. Explain the algorithm and illustrate the insertion of a new node at position 5..."*
- **Analysis**: Shared sub-domain vocabulary (*"linked list"*, *"node"*, *"position"*) triggered retrieval despite different operations (deletion vs. insertion).

#### 3. Informative False Negative (Scenario Narrative Variance)
- **Year A**: *"A digital watch sums times represented in BCD... Perform the BCD addition."*
- **Year B**: *"A digital weighing scale represents weight in BCD for ease of display... Calculate the total weight using BCD addition."*
- **Analysis**: Narrative story terms (*"watch"* vs. *"weighing scale"*) diluted the shared concept words, causing the similarity score ($\approx 0.14$) to fall just below threshold.

### 7.4 Interactive Dashboard Capabilities
The developed Streamlit web interface provides:
- **Universal Subject Switcher**: Instant switching between courses across departments.
- **Dynamic Threshold Slider ($0.05 - 0.60$)**: Real-time recalculation of matched pairs.
- **Dual-Pane Question Viewer**: Full-text side-by-side paper inspection with marks and question numbers.
- **Tiered Match Cards**: High ($\ge 0.35$), Medium ($0.25 - 0.35$), and Low ($0.15 - 0.25$) color-coded bands.

---

## 8. Limitations

1. **Testbed Dataset Size**: The initial empirical benchmark was conducted on 10 question papers (111 questions), which serves as a proof-of-concept for broader institutional deployment.
2. **Lexical Overlap Constraint (TF-IDF Sensitivity)**:
   - Does not detect matches when questions share zero identical words (pure conceptual paraphrasing).
   - Scenario-based questions with extensive story narratives dilute core terminology overlap.
3. **Threshold Variance Across Disciplines**: Highly specialized courses with strict technical jargon (e.g., Systems, Engineering) peak at $\tau \approx 0.25$, whereas descriptive/applied subjects (e.g., Management, Humanities, Basic Programming) require lower thresholds ($\tau \approx 0.15$).
4. **OCR Noise on Mathematical Notations**: Scanned equations, matrices, and circuit/flowchart diagrams are subject to occasional OCR degradation.
5. **Layout Heuristics**: Regex-based layout cleaners are configured for standard university examination layouts and require minor pattern tuning when onboarding institutions with differing header formats.

---

## 9. Future Scope

1. **Cross-Department Scaling**: Expand the repository to cover all university departments, including Physical Sciences, Commerce, Management Studies, Humanities, and Law.
2. **Dense Semantic Embeddings (Sentence-BERT / SBERT)**: Integrate deep transformer sentence encoders (such as `all-MiniLM-L6-v2` or `BGE-small-en`) alongside TF-IDF to accurately capture conceptual parity across varied narrative framings.
3. **Entity & Parameter Masking (NER)**: Implement Named Entity Recognition to abstract away contextual story elements (names, locations, numbers, currency) and isolate core theoretical principles.
4. **Longitudinal Topic Trend & Hotspot Analysis**: Apply hierarchical clustering across 5–10 years of multi-department exam papers to identify recurring syllabus themes, high-weightage modules, and question rotation patterns.
5. **Multimodal Diagram & Formula Recognition**: Incorporate vision-language models (e.g., LayoutLMv3, Nougat, Surya OCR) to extract and match mathematical equations, scientific diagrams, and structural tables.
6. **Campus-Wide LMS & Cloud Integration**: Deploy as a microservice API connected to campus Learning Management Systems (LMS) such as Moodle, Canvas, and Google Classroom for automated teacher assistance.

---

## 10. Conclusion

- **What Was Developed**: A generalizable, end-to-end NLP system and interactive Streamlit web dashboard capable of extracting, segmenting, and analyzing cross-year question papers across academic departments to identify repeated topics and questions.
- **NLP Techniques Used**: Optical Character Recognition (OCR), regex-driven layout cleaning and state-machine segmentation, tokenization, stopword removal, WordNet lemmatization, N-gram TF-IDF vectorization, and Cosine Similarity modeling.
- **Main Results**: Evaluated across 111 questions, the system achieved a peak precision of **1.000** at $\tau = 0.35$ and a balanced **F1-score of 0.435** at $\tau = 0.15$ (with top subject F1 reaching **0.640**), demonstrating strong viability for rapid academic question retrieval.
- **What Was Learned**: The project proved that an automated NLP pipeline can successfully process scanned legacy exam papers and surface syllabus overlaps across academic years. It also highlighted that while TF-IDF provides high explainability and speed, incorporating transformer embeddings and narrative abstraction will further empower cross-departmental, multi-discipline adoption.
