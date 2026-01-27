# OCR-Notice-Classifier
Ensemble-Based Classification of OCR-Extracted University Notices: A Weighted Voting Approach with Domain-Aware Feature Engineering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GitHub stars](https://img.shields.io/github/stars/chakrabt/OCR-Notice-Classifier?style=social)](https://github.com/chakrabt/OCR-Notice-Classifier)

> **Paper:** "Ensemble-Based Classification of OCR-Extracted University Notices: A Weighted Voting Approach with Domain-Aware Feature Engineering"  
> **Authors:** Tamal Chakraborty  
> **Institution:** Department of Computer Science, Mrinalini Datta Mahavidyapith, Kolkata, India  

---
## Documentation

- **[Feature Engineering Details](FEATURES.md)** - Complete specifications for all 35 handcrafted features
- **[Deployment Guide](DEPLOYMENT.md)** - Production deployment instructions, integration examples, and monitoring
- **[Quick Start Guide](README.md)** - Installation and basic usage


## Troubleshooting

### Common Issues

**Issue 1: Model files not found**
```
FileNotFoundError: Model not found! Train model first...
```

**Solution:**
- train your own: `classifier.train_production_model('train_data.csv')`
- Ensure model files are in the project root directory

**Issue 2: XGBoost installation fails**
```
ERROR: Could not build wheels for xgboost
```

**Solution:**
```bash
# Try pre-built binary
pip install xgboost --no-build-isolation

# Or use conda
conda install -c conda-forge xgboost
```

**Issue 3: NumPy version conflict**
```
numpy 2.0.0 is not supported
```

**Solution:**
```bash
pip install "numpy<2.0.0"
```

**Issue 4: Low accuracy on your data**

**Possible causes:**
- Different domain (not university notices)
- Different language (model trained on English)
- Extremely poor OCR quality
- Different notice formatting conventions

**Solutions:**
- Fine-tune on your own labeled data
- Adjust confidence thresholds
- Implement custom preprocessing for your domain

### Getting Help

- **Email:** tamalc@gmail.com

---

## Examples

### Example 1: Single Document Classification

```python
from src.ocr_classifier import OCRClassifier

classifier = OCRClassifier()
classifier.load_model()

notice = """
Merit list for PhD admission 2024-25 has been published. 
Selected candidates must attend counseling on 20th November.
"""

result = classifier.predict(notice)
print(f"→ {result['category']} ({result['confidence']})")
# Output: → admission (93.5%)
```

### Example 2: Batch Processing

```python
notices = [
    "Final exam schedule published...",
    "Workshop on AI/ML on 25th November...",
    "Circular No. 45/2024 regarding attendance policy..."
]

results = classifier.predict_batch(notices)

for notice, result in zip(notices, results):
    print(f"{notice[:30]}... → {result['category']}")

# Output:
# Final exam schedule published... → examination
# Workshop on AI/ML on 25th Nov... → event  
# Circular No. 45/2024 regarding... → circular
```

### Example 3: Confidence-Based Filtering

```python
result = classifier.predict(ambiguous_notice)

CONFIDENCE_THRESHOLD = 0.80

if result['confidence_score'] < CONFIDENCE_THRESHOLD:
    print("Low confidence - flag for manual review")
    print(f"Top alternatives:")
    
    for cat, prob in sorted(result['probabilities_raw'].items(), 
                           key=lambda x: x[1], reverse=True)[:3]:
        print(f"  - {cat}: {prob:.1%}")
else:
    print(f" High confidence - auto-route to {result['category']}")
```

### Example 4: Custom Integration

```python
# Define routing logic
ROUTING = {
    'examination': {'dept': 'Exam Cell', 'priority': 'HIGH'},
    'admission': {'dept': 'Admissions', 'priority': 'HIGH'},
    'circular': {'dept': 'Administration', 'priority': 'MEDIUM'},
    'event': {'dept': 'Student Affairs', 'priority': 'LOW'}
}

result = classifier.predict(notice_text)
routing_info = ROUTING[result['category']]

print(f"Route to: {routing_info['dept']}")
print(f"Priority: {routing_info['priority']}")
print(f"Confidence: {result['confidence']}")
```

For complete working examples, see [`examples/demo_usage.py`](examples/demo_usage.py).

---

## Overview

This repository contains the implementation of an ensemble-based machine learning system for classifying OCR-extracted university administrative notices into four categories: **Examination**, **Admission**, **Circular**, and **Event**.

### Key Features

- **91.89% Accuracy** on held-out test set
- **35 Domain-Specific Features** engineered for university notices
- **Voting Ensemble** (RF + GB + XGBoost + LR)
- **CPU-Only Operation** (no GPU required)
- **Fast Inference** (356ms per document)
- **Production-Ready** codebase

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/chakrabt/OCR-Notice-Classifier.git
cd OCR-Notice-Classifier

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Train Your Own Model**
```python
from src.ocr_classifier import OCRClassifier

classifier = OCRClassifier()
classifier.train_production_model('path/to/your/train_data.csv')
```

### Basic Usage

```python
from src.ocr_classifier import OCRClassifier

# Load pre-trained model
classifier = OCRClassifier()
classifier.load_model()

# Classify a notice
notice_text = """
Hall ticket available for semester final examination. 
Admit card can be downloaded from the student portal.
"""

result = classifier.predict(notice_text)
print(f"Category: {result['category']}")
print(f"Confidence: {result['confidence']}")
print(f"Probabilities: {result['probabilities']}")
```

**Output:**
```
Category: examination
Confidence: 96.6%
Probabilities: {'admission': '1.9%', 'circular': '0.7%', 'event': '0.8%', 'examination': '96.6%'}
```

### Run Demo Examples

```bash
# Run all usage examples
python examples/demo_usage.py

# This will demonstrate:
# 1. Basic single document classification
# 2. Batch processing multiple documents
# 3. Confidence-based filtering
# 4. Processing from file
# 5. Custom integration with routing logic
```

---
## Sample Notices Performance
The included sample notices demonstrate realistic and well-calibrated classifier behavior across different administrative categories. 

| Notice Type                 | Predicted Category | Confidence | Analysis                                                                        |
| --------------------------- | ------------------ | ---------- | ------------------------------------------------------------------------------- |
| Exam Schedule               | Examination        | 96.6%      | **Excellent** — highly distinctive examination terminology and structural cues  |
| PG Admission Notification   | Admission          | 86.5%      | **Good** — multi-stage admission process with mixed procedural vocabulary       |
| Identity Card Circular      | Circular           | 77.0%      | **Moderate** — generic administrative language with limited lexical specificity |
| AI/ML Workshop Announcement | Event              | 86.5%      | **Very Good** — clear event-related indicators with minor academic overlap      |
| Hall Ticket Instructions    | Examination        | 96.2%      | **Excellent** — strong, unambiguous examination-specific terminology            |

**Note:**
Lower confidence values for certain notice types—particularly admission and circular notices—reflect the inherent linguistic characteristics of these categories rather than classification errors. Admission notices often describe multiple programs, eligibility rules, counseling procedures, and schedules, resulting in diverse vocabulary and reduced lexical purity. Simpler admission notices (e.g., “Merit List Published” or “Counseling Schedule Released”) typically achieve confidence scores in the 85–95% range.

Circular notices similarly exhibit lower confidence due to their short length and generic administrative phrasing, which overlaps with other notice types. Importantly, the classifier remains correct in all predictions, and the confidence values provide meaningful signals for downstream handling.

Overall, the results demonstrate that the proposed OCR Notice Classifier is accurate, confidence-aware, and suitable for automated routing, while appropriately flagging less distinctive notices for cautious interpretation or optional manual review.

**Confidence Thresholds:**
- ≥85%: Auto-route with high confidence
- 70-84%: Auto-route with logging
- <70%: Flag for manual review

Note: These are anonymized samples for demonstration purposes only. 
Real university notices may contain additional information and formatting.

## Model Architecture

### Feature Engineering (35 Features)

Our approach combines **12,000 TF-IDF features** with **35 handcrafted domain-specific features**:

| Feature Group | Count | Examples |
|--------------|-------|----------|
| **Examination** | 9 | Exam mentions, hall ticket indicators, assessment terminology |
| **Admission** | 9 | Merit list detection, counseling markers, application terms |
| **Circular** | 8 | Formal numbering, bureaucratic language, notification terms |
| **Event** | 6 | Workshop/seminar keywords, ceremonial markers |
| **Structural** | 3 | Document length, token count, numeric density |

**Most Important Features** (from ablation study):
1. Numeric density (0.92)
2. Document length (0.88)
3. Keyword: exam (0.84)
4. Keyword: admission (0.81)
5. Year tokens (0.78)

### Ensemble Configuration (Architecture: Weighted Soft Voting)

```
Base Learners (4 models):
├── Random Forest
│   ├── Trees: 1000
│   ├── Max depth: 20
│   ├── Min samples split: 3
│   ├── Feature sampling: √p per split
│   ├── Class weighting: balanced
│   └── Voting weight: 3
│
├── Gradient Boosting
│   ├── Iterations: 400
│   ├── Learning rate: 0.05
│   ├── Max depth: 5
│   ├── Subsample ratio: 0.8
│   ├── Loss: multinomial deviance
│   └── Voting weight: 2
│
├── XGBoost
│   ├── Iterations: 500
│   ├── Learning rate: 0.07
│   ├── Max depth: 5
│   ├── Subsample ratio: 0.8
│   ├── Objective: multi:softmax
│   └── Voting weight: 3
│
└── Logistic Regression
    ├── Regularization: L2 (C=2.0)
    ├── Solver: LBFGS
    ├── Max iterations: 2000
    ├── Class weighting: balanced
    └── Voting weight: 1

Voting Mechanism:
├── Type: Soft voting (probability averaging)
├── Weights: [3, 2, 1, 3] for [RF, GB, LR, XGB]
├── Optimization: 5-fold CV grid search over {1,2,3}⁴ (81 configs)
└── Criterion: Maximizes macro-F1 score

Training Pipeline:
├── Data Split: 80/20 stratified (444 train / 111 test)
├── SMOTE: k=3, balancing (444 → 636 samples)
├── Feature Extraction: 12,035-dim (TF-IDF + custom features)
├── Base Learner Training: Each model trained independently
├── Weight Optimization: 5-fold CV to select [3,2,1,3]
└── Final Prediction: Weighted probability aggregation
```
**Key Difference from Stacking:**
- ✓ No meta-learner (single-stage ensemble)
- ✓ Direct probability aggregation with fixed weights
- ✓ Faster inference (356ms vs 583ms for stacking)
- ✓ Better performance (+1.8 pp over stacking)
---

## Performance

### Test Set Results (n=111)

| Category | Precision | Recall | F1-Score | Samples |
|----------|-----------|--------|----------|---------|
| Admission | 100.0% | 89.3% | 94.3% | 28 |
| Circular | 95.7% | 84.6% | 89.8% | 26 |
| Event | 77.3% | 100.0% | 87.2% | 17 |
| Examination | 92.7% | 95.0% | 93.8% | 40 |
| **Overall** | **92.9%** | **91.9%** | **92.0%** | **111** |

**Overall Accuracy:** 91.89%
**Macro-F1:** 0.9129
**95% CI:** [89.2%, 94.6%]

### Cross-Validation
- **Mean Accuracy:** 91.2% ± 1.3% (SD)
- **Statistical Significance:** p < 0.001 (McNemar's test vs BERT)
- **Stability:** Low variance indicates robust generalization

### Comparison with Baselines

| Model | Accuracy | Macro-F1 | Inference (ms) |
|-------|----------|----------|----------------|
| **Voting Ensemble (Ours)** | **91.89%** | **0.9129** | **356.3** |
| Stacking Ensemble | 90.09% | 0.8926 | 582.9 |
| BERT-base (fine-tuned) | 81.08% | 0.8004 | 639.4 |
| CNN-BiLSTM | 66.67% | 0.6533 | 63.2 |
| TF-IDF + Logistic Regression | 78.38% | 0.7843 | 8.1 |

**Advantages over baselines:**
- vs BERT: +10.8 pp accuracy, 44% faster, CPU-only
- vs Stacking: +1.8 pp accuracy, 39% faster, simpler architecture
- vs CNN-BiLSTM: +25.2 pp accuracy, avoids severe overfitting
- 
### Ablation Study

| Configuration | Accuracy | Δ Accuracy (pp) | Analysis |
|---------------|----------|-----------------|----------|
| **Voting Ensemble (Full)** | **91.89%** | **—** | **Proposed method** |
| TF-IDF+Features + XGBoost | 90.99% | -0.90 | Best single learner |
| TF-IDF + Gradient Boosting | 90.09% | -1.80 | Matches stacking |
| Stacking Ensemble | 90.09% | -1.80 | 2-stage architecture |
| TF-IDF+Features + GB | 89.19% | -2.70 | — |
| **TF-IDF + XGBoost** | **88.29%** | **-3.60** | **No domain features** |
| TF-IDF+Features+SMOTE + RF | 87.39% | -4.50 | — |
| TF-IDF+Features + RF | 86.49% | -5.40 | — |
| TF-IDF + Random Forest | 81.08% | -10.81 | — |
| TF-IDF + Logistic Regression | 78.38% | -13.51 | Baseline |

**Key Insights:**
- **Feature engineering contribution:** Comparing TF-IDF+XGB (88.29%) vs TF-IDF+Features+XGB (90.99%) → **+2.70 pp**
- **Voting advantage:** Voting (91.89%) vs best single learner (90.99%) → **+0.90 pp**
- **Voting vs Stacking:** Voting (91.89%) vs Stacking (90.09%) → **+1.80 pp** with 39% faster inference

---

## Reproducibility 

All experiments are fully reproducible:

- **Dataset:** 555 notices across 4 categories (Examination, Admission, Circular, Event)
  - Original collection: 1,316 PDFs from Indian universities
  - After OCR cleaning: 1,228 usable texts
  - After filtering: 690 documents (5 categories)
  - Final dataset: 555 documents (4 categories, Holiday excluded)
- **Train/Test Split:** 80/20 stratified (444 train, 111 test)
- **SMOTE Balancing:** 444 → 636 training samples
- **Random Seed:** 42 (for all operations)
- **Hardware:** Intel i7-12700, 16GB RAM, CPU-only
- **Cross-Validation:** 5-fold stratified
- **Training Time:** ~12 minutes per full training run

### Training Configuration
```python
# Exact configuration from paper
config = {
    'random_forest': {
        'n_estimators': 1000,
        'max_depth': 20,
        'min_samples_split': 3,
        'class_weight': 'balanced'
    },
    'gradient_boosting': {
        'n_estimators': 400,
        'learning_rate': 0.05,
        'max_depth': 5,
        'subsample': 0.8
    },
    'xgboost': {
        'n_estimators': 500,
        'learning_rate': 0.07,
        'max_depth': 5,
        'subsample': 0.8
    },
    'logistic_regression': {
        'C': 2.0,
        'solver': 'lbfgs',
        'max_iter': 2000,
        'class_weight': 'balanced'
    },
    'voting_weights': [3, 2, 1, 3]  # [RF, GB, LR, XGB]
}
```

## Training Your Own Model

### Prepare Your Dataset

Your CSV should have two columns:
```csv
extracted_text,category
"Hall ticket available for...",examination
"Merit list published for...",admission
"Circular No. 123/2024...",circular
"Workshop on AI/ML...",event
```

### Train the Model

```python
from src.ocr_classifier import OCRClassifier

classifier = OCRClassifier()
classifier.train_production_model('path/to/train_data.csv')
```

**Training time:** ~12 minutes (Intel i7-12700, 16GB RAM, CPU-only)

---

## Repository Structure

```
├── src/
│   ├── ocr_classifier.py          # Main classifier (OCRClassifier)
│
├── examples/
│   ├── demo_usage.py              # Quick start examples
│   └── sample_notices.txt         # Anonymized sample data
|
├── requirements.txt               # Dependencies
├── LICENSE                        # MIT License
├── deployment.md                  # Deployment Readiness Analysis
├── fatures.md                     # Details of 35 handcrafted features for Notice Classification
└── README.md                      # This file
```

---

## Dependencies

```
python >= 3.8
scikit-learn>=1.0.0,<1.3.0
xgboost>=1.5.0,<3.0.0
imbalanced-learn>=0.9.0,<1.0.0
pandas>=1.3.0,<3.0.0
numpy>=1.21.0,<2.0.0
joblib>=1.1.0,<2.0.
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{chakraborty2025ocr,
  title={Ensemble-Based Classification of OCR-Extracted University Notices:
         A Weighted Voting Approach with Domain-Aware Feature Engineering},
  author={Chakraborty, Tamal},
  journal={Journal of Computer Science and Technology},
  year={2025},
  note={Submitted}
}
```

---

## Why Weighted Voting Over Stacking?

Our experiments compared both approaches:

| Aspect | Weighted Voting (Ours) | Stacking |
|--------|------------------------|----------|
| **Architecture** | Single-stage | Two-stage (meta-learner) |
| **Training** | Independent base learners | Meta-learner requires CV |
| **Inference** | 356ms | 583ms (+64%) |
| **Accuracy** | 91.89% | 90.09% (-1.8 pp) |
| **Macro-F1** | 0.9129 | 0.8926 (-2.03 pp) |
| **Interpretability** | Direct weight influence | Black-box meta-learner |
| **Risk** | No error propagation | Cascading errors possible |

**Decision:** Weighted voting provides superior performance with simpler architecture and faster inference, making it ideal for production deployment.

---

## Performance Benchmarks 

| System | Accuracy | Inference Time | Hardware | Notes |
|--------|----------|----------------|----------|-------|
| **Ours (Voting)** | **91.89%** | **356ms/doc** | **CPU** | **Production-ready** |
| Stacking | 90.09% | 583ms/doc | CPU | Our comparison |
| BERT-base | 81.08% | 639ms/doc | GPU | Fine-tuned 3 epochs |
| CNN-BiLSTM | 66.67% | 63ms/doc | GPU | Severe overfitting |
| TF-IDF + LR | 78.38% | 8ms/doc | CPU | Baseline |

**Key Advantages:**
- ✓ Highest accuracy among all tested methods
- ✓ CPU-only operation (no GPU required)
- ✓ Real-time processing capability (<400ms)
- ✓ Production-stable (no overfitting)
- ✓ Interpretable feature importance

---

## Model Files

Pre-trained model files are available upon request:
- `ocr_model.pkl` (450MB) - Trained ensemble
- `ocr_vectorizer.pkl` - TF-IDF vectorizer
- `ocr_encoder.pkl` - Label encoder
- `ocr_metadata.pkl` - Model metadata

**Note:** Due to size constraints, model files are not included in the repository. Please contact the author to obtain them.

---

## Deployment Options

### Option 1: REST API
```python
from flask import Flask, request, jsonify
from src.ocr_classifier import OCRClassifier

app = Flask(__name__)
classifier = OCRClassifier()
classifier.load_model()

@app.route('/classify', methods=['POST'])
def classify():
    text = request.json['text']
    result = classifier.predict(text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Option 2: Batch Processing
```python
import pandas as pd
from src.ocr_classifier import OCRClassifier

classifier = OCRClassifier()
classifier.load_model()

# Read notices from CSV
df = pd.read_csv('notices.csv')
results = classifier.predict_batch(df['text'].tolist())

# Save results
df['predicted_category'] = [r['category'] for r in results]
df.to_csv('classified_notices.csv', index=False)
```

---

## Limitations

- **Test set size:** 111 samples (modest but cross-validated)
- **Language:** English-only (multilingual support planned)
- **Domain:** University notices (generalizable to similar admin documents)
- **OCR quality:** Tested on moderate-quality scans (not severely degraded)

---

## Future Work

- [ ] Multilingual support (Hindi, Bengali)
- [ ] Layout-aware features (bounding boxes, font sizes)
- [ ] Active learning for error correction
- [ ] Open-set detection for novel notice types
- [ ] Web-based demo interface

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## Frequently Asked Questions (FAQ)

### General Questions

**Q: Can I use this for non-university documents?**  
A: The model is specifically trained for university administrative notices. Performance on other document types (government circulars, corporate memos) may vary but could work with fine-tuning.

**Q: Does this work with handwritten notices?**  
A: No. The system is designed for typed/printed text that has been digitized via OCR. Handwriting recognition requires different approaches.

**Q: What languages are supported?**  
A: Currently only English. Multilingual support (Hindi, Bengali) is planned for future releases.

**Q: Can I use this commercially?**  
A: Yes! The code is released under MIT License, which permits commercial use. Please cite the paper if you use it in your work.

### Technical Questions

**Q: Why 91.89% accuracy and not higher?**  
A: The remaining 8.11% errors are primarily from genuinely ambiguous notices (e.g., "Circular regarding exam schedule" could be both). 

**Q: Can I add more categories?**  
A: Yes, but you'll need to retrain the model with labeled data for your new categories. The feature engineering approach is generalizable.

**Q: Do I need a GPU?**  
A: No! The system runs efficiently on CPU-only hardware. Training takes ~12 minutes on a standard laptop.

**Q: Can I integrate this with my existing system?**  
A: Yes! See [Deployment Options](#deployment-options) for REST API, batch processing, and direct Python integration examples.

### Data and Privacy Questions

**Q: Can I get access to your training data?**  
A: Unfortunately no. The dataset contains institutional notices that may include personally identifiable information. However, the methodology is fully documented for replication on your own data.

**Q: How do I prepare my own dataset?**  
A: You need a CSV with two columns: `extracted_text` (OCR-extracted notice text) and `category` (examination/admission/circular/event). 

**Q: What about data privacy in deployment?**  
A: All processing is local - no data is sent to external services. If deploying as a web service, implement appropriate authentication and encryption.

---

## Contact

**Tamal Chakraborty**  
Department of Computer Science  
Mrinalini Datta Mahavidyapith, Kolkata, West Bengal, India  
Email: tamalc@gmail.com

---

## Acknowledgments

- Dataset collected from public university websites
- Inspired by real administrative challenges in university document management
- Thanks to reviewers and colleagues for valuable feedback

---

**⭐ If you find this work useful, please consider starring the repository!**
