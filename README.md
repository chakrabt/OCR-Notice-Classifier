# OCR-Notice-Classifier
Ensemble-Based 4-Class OCR Document Classification for University Notices

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GitHub stars](https://img.shields.io/github/stars/chakrabt/OCR-Notice-Classifier?style=social)](https://github.com/chakrabt/OCR-Notice-Classifier)

> **Paper:** "Ensemble-Based 4-Class OCR Document Classification for University Notices"  
> **Authors:** Tamal Chakraborty  
> **Institution:** Department of Computer Science, Mrinalini Datta Mahavidyapith, Kolkata, India  

---

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
- **Stacking Ensemble** (RF + GB + XGBoost + LR)
- **CPU-Only Operation** (no GPU required)
- **Fast Inference** (42ms per document)
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
================================================================================
SAMPLE UNIVERSITY NOTICES - ANONYMIZED EXAMPLES
================================================================================

These are anonymized sample notices representing the four categories used in
the OCR Notice Classification system. All names, dates, and identifying
information have been modified to protect privacy.

Category Distribution:
- Examination: 2 samples
- Admission: 1 sample  
- Circular: 1 sample
- Event: 1 sample

================================================================================

===== NOTICE 1: EXAMINATION =====

NOTICE - SEMESTER END EXAMINATION SCHEDULE

Date: 15th November 2024
Reference No: EXAM/2024/156

This is to inform all students that the semester end examinations for the 
academic year 2024-25 (Odd Semester) will be conducted from 5th December 2024 
to 20th December 2024.

IMPORTANT INSTRUCTIONS:

1. Hall tickets will be available for download from the student portal from 
   28th November 2024 onwards.

2. Students must carry their admit cards along with a valid photo ID card to 
   the examination hall.

3. Answer sheets will be provided at the examination venue. Students should 
   not bring their own answer sheets.

4. The detailed examination timetable with dates, timings, and venue 
   information has been uploaded to the university website.

5. Students are advised to check their examination schedule carefully and 
   report any discrepancies to the Examination Cell immediately.

6. Grading and results will be published within 30 days of the completion 
   of examinations.

For any queries regarding hall tickets, examination schedules, or other 
examination-related matters, please contact:

Examination Cell
Email: exams@university.edu.in
Phone: +91-XXXX-XXXXXX

Controller of Examinations
University Name

===== NOTICE 2: ADMISSION =====

ADMISSION NOTIFICATION - POSTGRADUATE PROGRAMS 2025-26

Notification No: ADM/PG/2024/089
Date: 10th October 2024

The University invites applications for admission to various Postgraduate 
programs (MA, MSc, MCom, MCA) for the academic session 2025-26.

PROGRAMS OFFERED:
- Master of Arts (English, History, Political Science)
- Master of Science (Physics, Chemistry, Mathematics, Computer Science)
- Master of Commerce
- Master of Computer Applications

ELIGIBILITY CRITERIA:
Candidates must have completed their Bachelor's degree in the relevant 
discipline with a minimum of 55% marks (50% for SC/ST/OBC candidates) from 
a recognized university.

APPLICATION PROCESS:
1. Online application forms are available on the university admission portal
2. Application fee: Rs. 500/- (Rs. 250/- for SC/ST candidates)
3. Last date for submission: 30th November 2024

ENTRANCE EXAMINATION:
An entrance test will be conducted on 15th December 2024 at designated 
examination centers across the state.

MERIT LIST AND COUNSELING:
- Merit list will be published on 5th January 2025
- Counseling sessions for seat allocation: 10-15 January 2025
- Selected candidates must report for document verification and fee payment

RESERVATION POLICY:
Seats are reserved as per government guidelines:
- SC: 15%, ST: 7.5%, OBC: 27%, EWS: 10%

For detailed information, eligibility requirements, and program-specific 
guidelines, please visit our website or contact:

Admissions Office
Email: admissions@university.edu.in
Helpline: +91-XXXX-XXXXXX (10 AM - 5 PM, Monday to Friday)

Registrar
University Name

===== NOTICE 3: CIRCULAR =====

CIRCULAR NO. 234/2024 - STUDENT IDENTITY CARD POLICY

Date: 1st September 2024
Ref: FACULT/STUD/2024/234

To: All Students, Faculty Members, and Heads of Departments

Subject: Circular - Revised Policy for Student Identity Cards : Academic Year 2024-25

This is to notify all concerned that the following revised guidelines for 
student identity cards have been approved by the administration and will be 
implemented with immediate effect.

All students must obtain their official university identity cards within 
one month of enrollment. Identity cards are mandatory for accessing campus 
 facilities including library, laboratories, and administrative offices.

This circular supersedes all previous notifications regarding student 
identity cards.

For any queries or clarifications, please contact the Administration Office.

By Order,
Registrar
University Name

===== NOTICE 4: EVENT =====

WORKSHOP ANNOUNCEMENT - ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING

Event ID: WORKSHOP/CS/2024/12
Date: 20th November 2024

The Department of Computer Science is organizing a Two-Day National Workshop 
on "Artificial Intelligence and Machine Learning: Current Trends and Future 
Directions" on 5-6 December 2024.

WORKSHOP DETAILS:

Date: 5th & 6th December 2024 (Tuesday & Wednesday)
Time: 9:00 AM to 5:00 PM
Venue: Seminar Hall, Department of Computer Science
Mode: Hybrid (In-person and Online)

TOPICS TO BE COVERED:
- Introduction to Machine Learning and Deep Learning
- Natural Language Processing and Computer Vision
- Reinforcement Learning and Generative AI
- Real-world Applications and Case Studies
- Hands-on Sessions with Python and Popular ML Libraries

RESOURCE PERSONS:
Distinguished speakers from academia and industry, including researchers 
from leading institutions and AI companies.

WHO CAN ATTEND:
- Undergraduate and Postgraduate students (all disciplines)
- Research scholars and faculty members
- Industry professionals and enthusiasts

REGISTRATION:
- Registration is FREE for all participants
- Limited seats available (150 for in-person, 500 for online)
- Register online at: workshop.university.edu.in
- Registration deadline: 30th November 2024

CERTIFICATES:
Participation certificates will be provided to all registered attendees who 
attend both days of the workshop.

For more information, contact:

Workshop Coordinator
Dr. Name Redacted, Associate Professor
Department of Computer Science
Email: workshop2024@university.edu.in
Phone: +91-XXXX-XXXXXX

All are invited to participate and benefit from this learning opportunity!

Head, Department of Computer Science
University Name

===== NOTICE 5: EXAMINATION (HALL TICKET) =====

EXAMINATION HALL TICKET - IMPORTANT INSTRUCTIONS

Examination Roll Number: 2024XXXX
Student Name: [REDACTED]
Program: Bachelor of Science (Computer Science)
Semester: V (Odd Semester 2024-25)

This is your official Hall Ticket for the semester examinations. Please read 
the following instructions carefully:

GENERAL INSTRUCTIONS:

1. This hall ticket must be brought to the examination hall along with a 
   valid photo identity card (Student ID/Aadhar Card/Driving License).

2. Students should report to the examination venue 30 minutes before the 
   scheduled start time.

3. Mobile phones, smartwatches, calculators (except where permitted), and 
   any electronic devices are strictly prohibited inside the examination hall.

4. Answer sheets are provided by the university. Writing on answer sheets 
   before the examination starts is prohibited.

5. Students must write their examination roll number on the answer sheet. 
   Do NOT write your name on the answer sheet.

6. Use of unfair means will result in cancellation of the examination and 
   disciplinary action.

EXAMINATION SCHEDULE:

Date       | Subject Name              | Time        | Venue
-----------|---------------------------|-------------|------------------
05-Dec-24  | Data Structures          | 10:00 AM    | Examination Hall A
08-Dec-24  | Database Management      | 10:00 AM    | Examination Hall A
12-Dec-24  | Operating Systems        | 10:00 AM    | Examination Hall B
15-Dec-24  | Computer Networks        | 10:00 AM    | Examination Hall B
18-Dec-24  | Software Engineering     | 10:00 AM    | Examination Hall A

IMPORTANT NOTES:

- Verify your examination schedule and report any discrepancies immediately
- Check the examination hall location before the examination day
- Results will be declared within 30 days after completion of exams
- Grade sheets will be available on the student portal

For queries, contact the Examination Cell during office hours.

This is a computer-generated hall ticket and does not require a signature.

================================================================================
END OF SAMPLE NOTICES
================================================================================

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

### Ensemble Configuration

```
Level 1 Base Learners:
├── Random Forest (n=800 trees, depth=15)
├── Gradient Boosting (n=300, lr=0.08)
└── XGBoost (n=400, lr=0.10)

Level 2 Meta-Learner:
└── Logistic Regression (C=0.5, balanced)

Training:
├── SMOTE balancing (444 → 636 samples)
├── 3-fold stacking
└── 5-fold cross-validation
```

---

## Performance

### Test Set Results (n=111)

| Category | Precision | Recall | F1-Score | Samples |
|----------|-----------|--------|----------|---------|
| Admission | 100.0% | 89.3% | 94.3% | 28 |
| Circular | 95.7% | 84.6% | 89.8% | 26 |
| Event | 77.3% | 100.0% | 87.2% | 17 |
| Examination | 92.7% | 95.0% | 93.8% | 40 |
| **Overall** | **92.86%** | **91.89%** | **91.99%** | **111** |

### Cross-Validation
- **Mean Accuracy:** 91.2% ± 1.3%
- **Statistical Significance:** p < 0.001 (McNemar's test)
- **95% Confidence Interval:** [89.2%, 94.6%]

### Ablation Study

| Component Removed | Accuracy | Δ |
|------------------|----------|---|
| **None (Full Model)** | 91.89% | — |
| Engineered Features | 86.5% | -5.39% |
| Stacking Ensemble | 87.1% | -4.79% |
| SMOTE Balancing | 88.4% | -3.49% |
| 5-class (with Holiday) | 84.7% | -7.19% |

---

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
  title={Ensemble-Based 4-Class OCR Document Classification for University Notices},
  author={Chakraborty, Tamal},
  year={2026},
  note={Submitted}
}
```

---

## Reproducibility

All experiments are fully reproducible:

- **Dataset:** 683 notices from Indian universities
- **Train/Test Split:** 80/20 stratified (444 train, 111 test)
- **Random Seed:** 42 (for all operations)
- **Hardware:** Intel i7-12700, 16GB RAM, CPU-only
- **Cross-Validation:** 5-fold stratified

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

## Performance Benchmarks

| System | Accuracy | Inference Time | Hardware |
|--------|----------|---------------|----------|
| **Ours** | **91.89%** | **42ms/doc** | **CPU** |
| TF-IDF + LR | 76.6% | 15ms/doc | CPU |
| BERT-base | ~83-86% | 180ms/doc | GPU |
| TrOCR + XGBoost | ~85-87% | 250ms/doc | GPU |

---

## Limitations

- **Test set size:** 111 samples (modest but cross-validated)
- **Language:** English-only (multilingual support planned)
- **Domain:** University notices (generalizable to similar admin documents)
- **OCR quality:** Tested on moderate-quality scans (not severely degraded)

See paper Section 4.10 and 5.4 for detailed discussion.

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
A: The remaining 8.11% errors are primarily from genuinely ambiguous notices (e.g., "Circular regarding exam schedule" could be both). See Section 4.7 of the paper for detailed error analysis.

**Q: Can I add more categories?**  
A: Yes, but you'll need to retrain the model with labeled data for your new categories. The feature engineering approach is generalizable.

**Q: How much training data do I need?**  
A: Our learning curve analysis (Figure 3 in paper) suggests stable performance with 60-70% of our dataset (~300-350 samples). Minimum recommended: 50 samples per category.

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
