# Feature Engineering Documentation

**Project:** Ensemble-Based OCR Document Classification for University Notices  
**Author:** Tamal Chakraborty  
**Repository:** https://github.com/chakrabt/OCR-Notice-Classifier

---

## Overview

This document provides complete specifications for the 35 handcrafted domain-specific features used in the classification system. These features were designed to address specific confusion patterns observed in pilot experiments with university administrative notices.

The feature engineering strategy targets three primary confusion patterns:
1. **Circular-Examination overlap** - Resolved via procedural phrase ratios and scheduling markers
2. **Admission-Event intersection** - Disambiguated through counselling markers and orientation signals
3. **Circular identification** - Enhanced via formal numbering conventions and bureaucratic register detection

---

## Feature Categories

### 📋 Summary Table

| Category | Feature Count | Primary Function |
|----------|---------------|------------------|
| Examination | 9 | Distinguish exam notices from circulars and admissions |
| Admission | 9 | Separate admission processes from events and exams |
| Circular | 8 | Identify administrative directives vs. operational notices |
| Event | 6 | Recognize programmatic announcements |
| Structural | 3 | Provide document-level discriminative signals |
| **Total** | **35** | |

---

## 1. Examination Features (9 features)

### Purpose
Mitigate circular-examination confusion by capturing examination-specific lexical and procedural signals.

### Feature Specifications

#### 1.1 Explicit Term Frequency
**Type:** Continuous (count)  
**Implementation:**
```python
exam_count = text.lower().count('exam') + text.lower().count('examination')
```
**Rationale:** Direct topical indicator; examinations explicitly mentioned in relevant notices.

---

#### 1.2 Domain Vocabulary Presence
**Type:** Continuous (aggregate count)  
**Keywords:**
- Assessment components: `hall ticket`, `admit card`, `answer sheet`, `question paper`
- Grading terms: `marks`, `grade`, `result`, `score`
- Procedural elements: `internal assessment`, `external examination`

**Implementation:**
```python
exam_vocab = ['hall ticket', 'admit card', 'answer sheet', 'question paper',
              'marks', 'grade', 'result', 'score', 'assessment']
vocab_count = sum(text.lower().count(term) for term in exam_vocab)
```
**Rationale:** Distinguishes examination logistics from general administrative text.

---

#### 1.3 Category Disambiguation Ratio
**Type:** Continuous (ratio)  
**Implementation:**
```python
exam_mentions = text.lower().count('exam') + text.lower().count('examination')
admission_mentions = text.lower().count('admission') + text.lower().count('admit')
ratio = exam_mentions / (admission_mentions + 1)  # +1 to avoid division by zero
```
**Rationale:** Resolves ambiguous notices like "examination for admission" or "admission examination" by determining dominant category through relative frequency.

---

#### 1.4 Procedural Phrase Indicators
**Type:** Binary (0/1)  
**Keywords:** `hall ticket`, `admit card`  
**Implementation:**
```python
has_procedural = int(any(phrase in text.lower() for phrase in ['hall ticket', 'admit card']))
```
**Rationale:** These phrases signal examination logistics and entry protocols; rarely appear outside examination contexts.

---

#### 1.5 Assessment Material Indicators
**Type:** Binary (0/1)  
**Keywords:** `answer sheet`, `answer book`, `question paper`  
**Implementation:**
```python
has_materials = int(any(phrase in text.lower() for phrase in ['answer sheet', 'answer book', 'question paper']))
```
**Rationale:** Captures notices about exam administration, material distribution, and submission protocols.

---

#### 1.6 Term Density Normalization
**Type:** Continuous (normalized frequency)  
**Implementation:**
```python
exam_terms = ['exam', 'examination', 'test', 'assessment']
term_count = sum(text.lower().count(term) for term in exam_terms)
density = term_count / max(len(text), 1)  # Normalize by document length
```
**Rationale:** Accounts for verbosity differences; longer circulars may mention exams incidentally without being exam-focused.

---

#### 1.7 Category Purity Indicator
**Type:** Binary (0/1)  
**Implementation:**
```python
has_exam_terms = any(term in text.lower() for term in ['exam', 'examination'])
has_competing_terms = any(term in text.lower() for term in ['circular', 'admission', 'workshop'])
is_pure = int(has_exam_terms and not has_competing_terms)
```
**Rationale:** Identifies unambiguous examination notices without competing category signals.

---

#### 1.8 Assessment Terminology
**Type:** Binary (0/1)  
**Keywords:** `marks`, `grades`, `results`, `evaluation`, `performance`  
**Implementation:**
```python
assessment_terms = ['marks', 'grade', 'result', 'evaluation', 'performance']
has_assessment = int(any(term in text.lower() for term in assessment_terms))
```
**Rationale:** Signals examination contexts focused on academic performance rather than administrative announcements.

---

#### 1.9 Temporal Scheduling Markers
**Type:** Binary (0/1)  
**Implementation:**
```python
has_exam = any(term in text.lower() for term in ['exam', 'examination'])
has_schedule = any(term in text.lower() for term in ['timetable', 'schedule', 'date'])
has_both = int(has_exam and has_schedule)
```
**Rationale:** Distinguishes exam timetables from result announcements or policy circulars.

---

## 2. Admission Features (9 features)

### Purpose
Resolve admission-event confusion, particularly for notices like "Workshop for admission candidates" or "Orientation for newly admitted students."

### Feature Specifications

#### 2.1 Explicit Admission Mentions
**Type:** Continuous (count)  
**Implementation:**
```python
admission_count = text.lower().count('admission') + text.lower().count('admit')
```
**Rationale:** Primary topical indicator for admission-related content.

---

#### 2.2 Process-Specific Vocabulary
**Type:** Continuous (aggregate count)  
**Keywords:**
- Procedural terms: `apply`, `application`, `eligibility`, `entrance`
- Counselling activities: `counselling`, `counseling`, `merit list`, `seat allotment`
- Program references: `PhD`, `MPhil`, `research program`

**Implementation:**
```python
admission_vocab = ['apply', 'application', 'eligibility', 'entrance',
                   'counselling', 'counseling', 'merit list', 'seat allotment',
                   'phd', 'mphil', 'research program']
vocab_count = sum(text.lower().count(term) for term in admission_vocab)
```
**Rationale:** Captures admission-specific processes distinct from general events.

---

#### 2.3 Merit and Selection Indicators
**Type:** Binary (0/1)  
**Keywords:** `merit list`, `selected candidates`, `rank holders`, `waitlist`  
**Implementation:**
```python
merit_phrases = ['merit list', 'selected candidates', 'rank holder', 'waitlist']
has_merit = int(any(phrase in text.lower() for phrase in merit_phrases))
```
**Rationale:** Distinguishes admission results from event announcements; strong admission signal.

---

#### 2.4 Program-Level Admission Signals
**Type:** Binary (0/1)  
**Keywords:** `PhD`, `MPhil`, `research admission`, `doctoral`  
**Implementation:**
```python
program_terms = ['phd', 'mphil', 'research admission', 'doctoral']
has_program = int(any(term in text.lower() for term in program_terms))
```
**Rationale:** Graduate admissions employ distinct vocabulary compared to undergraduate processes.

---

#### 2.5 Counselling Activity Markers
**Type:** Binary (0/1)  
**Keywords:** `counselling`, `counseling`, `counselling session`  
**Implementation:**
```python
has_counselling = int('counsell' in text.lower())  # Catches both spellings
```
**Rationale:** Strong admission signal; rarely appears in other categories.

---

#### 2.6 Admission Density
**Type:** Continuous (normalized frequency)  
**Implementation:**
```python
admission_terms = ['admission', 'admit', 'admissions']
term_count = sum(text.lower().count(term) for term in admission_terms)
density = term_count / max(len(text), 1)
```
**Rationale:** Normalizes for document length variations.

---

#### 2.7 Eligibility Vocabulary
**Type:** Binary (0/1)  
**Keywords:** `eligibility`, `eligible`, `qualification`, `criteria`  
**Implementation:**
```python
eligibility_terms = ['eligibility', 'eligible', 'qualification', 'criteria']
has_eligibility = int(any(term in text.lower() for term in eligibility_terms))
```
**Rationale:** Characteristic of admission notices specifying entry requirements.

---

#### 2.8 Application Process Indicators
**Type:** Binary (0/1)  
**Keywords:** `apply`, `application form`, `registration`, `submission`  
**Implementation:**
```python
application_terms = ['apply', 'application form', 'registration', 'submission']
has_application = int(any(term in text.lower() for term in application_terms))
```
**Rationale:** Signals procedural admission content about application mechanics.

---

#### 2.9 Entrance Examination References
**Type:** Binary (0/1)  
**Keywords:** `entrance`, `entrance exam`, `entrance test`  
**Implementation:**
```python
has_entrance = int('entrance' in text.lower())
```
**Rationale:** Bridges examination and admission contexts; helps resolve category boundaries.

---

## 3. Circular Features (8 features)

### Purpose
Capture bureaucratic and administrative signals that distinguish circulars from operational notices.

### Feature Specifications

#### 3.1 Circular Term Frequency
**Type:** Continuous (count)  
**Implementation:**
```python
circular_count = text.lower().count('circular') + text.lower().count('notification')
```
**Rationale:** Direct detection of circular/notification terminology.

---

#### 3.2 Formal Numbering Conventions
**Type:** Binary (0/1)  
**Pattern:** Matches formats like "Circular No. 123/2024" or "Notification No. XYZ"  
**Implementation:**
```python
import re
numbering_pattern = r'(circular|notification)\s+(no\.|number)\s*[\d/\-]+'
has_numbering = int(bool(re.search(numbering_pattern, text.lower())))
```
**Rationale:** Official circulars use standardized numbering; absent in other categories.

---

#### 3.3 Bureaucratic Register Markers
**Type:** Binary (0/1)  
**Keywords:** `hereby`, `herewith`, `aforementioned`, `undersigned`  
**Implementation:**
```python
bureaucratic_terms = ['hereby', 'herewith', 'aforementioned', 'undersigned']
has_bureaucratic = int(any(term in text.lower() for term in bureaucratic_terms))
```
**Rationale:** Formal administrative language signals official pronouncements.

---

#### 3.4 Notification Terminology
**Type:** Binary (0/1)  
**Keywords:** `notified`, `notification`, `inform`, `announce`  
**Implementation:**
```python
notification_terms = ['notified', 'notification', 'inform', 'announce']
has_notification = int(any(term in text.lower() for term in notification_terms))
```
**Rationale:** Formal notification vocabulary characteristic of administrative circulars.

---

#### 3.5 Formal Language Co-occurrence
**Type:** Binary (0/1)  
**Implementation:**
```python
has_circular = any(term in text.lower() for term in ['circular', 'notification'])
has_bureaucratic = any(term in text.lower() for term in ['hereby', 'herewith'])
has_both = int(has_circular and has_bureaucratic)
```
**Rationale:** Composite indicator capturing distinctive formal register of administrative circulars.

---

#### 3.6 Category Dominance Signal
**Type:** Binary (0/1)  
**Implementation:**
```python
has_circular = 'circular' in text.lower()
has_competing = any(term in text.lower() for term in ['exam', 'admission', 'workshop'])
is_dominant = int(has_circular and not has_competing)
```
**Rationale:** Identifies pure administrative circulars without operational content.

---

#### 3.7 Addressee Formality
**Type:** Binary (0/1)  
**Keywords:** `all concerned`, `whom it may concern`, `to all faculty`, `to all students`  
**Implementation:**
```python
formal_address = ['all concerned', 'whom it may concern', 'to all faculty', 'to all students']
has_formal = int(any(phrase in text.lower() for phrase in formal_address))
```
**Rationale:** Formal addressing conventions typical of circular broadcasts.

---

#### 3.8 Circular Density
**Type:** Continuous (normalized frequency)  
**Implementation:**
```python
circular_terms = ['circular', 'notification', 'hereby', 'notified']
term_count = sum(text.lower().count(term) for term in circular_terms)
density = term_count / max(len(text), 1)
```
**Rationale:** Normalized frequency accounts for document length variations.

---

## 4. Event Features (6 features)

### Purpose
Recognize programmatic announcements using action-oriented vocabulary that distinguishes events from administrative circulars or procedural notices.

### Feature Specifications

#### 4.1 Event Vocabulary Richness
**Type:** Continuous (aggregate count)  
**Keywords:**
- Activity types: `workshop`, `seminar`, `webinar`, `training`, `competition`
- Program descriptors: `orientation`, `inauguration`, `ceremony`
- Participation frameworks: `program`, `session`, `meet`

**Implementation:**
```python
event_vocab = ['workshop', 'seminar', 'webinar', 'training', 'competition',
               'orientation', 'inauguration', 'ceremony', 'program', 'session']
vocab_count = sum(text.lower().count(term) for term in event_vocab)
```
**Rationale:** Captures action-oriented nature of event announcements.

---

#### 4.2 Common Event Type Indicators
**Type:** Binary (0/1)  
**Keywords:** `workshop`, `seminar`, `webinar`  
**Implementation:**
```python
common_events = ['workshop', 'seminar', 'webinar']
has_common = int(any(term in text.lower() for term in common_events))
```
**Rationale:** Most frequent event types; strong discriminative signal.

---

#### 4.3 Training and Orientation Signals
**Type:** Binary (0/1)  
**Keywords:** `training`, `orientation`, `induction`  
**Implementation:**
```python
training_terms = ['training', 'orientation', 'induction']
has_training = int(any(term in text.lower() for term in training_terms))
```
**Rationale:** Indicates structured educational event activities.

---

#### 4.4 Event Category Purity
**Type:** Binary (0/1)  
**Implementation:**
```python
has_event = any(term in text.lower() for term in ['workshop', 'seminar', 'program'])
has_exam = 'exam' in text.lower()
has_admission = 'admission' in text.lower()
is_pure = int(has_event and not (has_exam or has_admission))
```
**Rationale:** Identifies pure event announcements vs. admission workshops or exam preparation seminars.

---

#### 4.5 Event Density Normalization
**Type:** Continuous (normalized frequency)  
**Implementation:**
```python
event_terms = ['workshop', 'seminar', 'webinar', 'program', 'event']
term_count = sum(text.lower().count(term) for term in event_terms)
density = term_count / max(len(text), 1)
```
**Rationale:** Accounts for verbosity differences across notices.

---

#### 4.6 Ceremonial and Celebratory Markers
**Type:** Binary (0/1)  
**Keywords:** `inauguration`, `ceremony`, `celebration`, `observance`  
**Implementation:**
```python
ceremonial_terms = ['inauguration', 'ceremony', 'celebration', 'observance']
has_ceremonial = int(any(term in text.lower() for term in ceremonial_terms))
```
**Rationale:** Distinguishes formal events from routine academic operations.

---

## 5. Structural Features (3 features)

### Purpose
Provide document-level statistics that remain robust under lexical variation and OCR noise.

### Feature Specifications

#### 5.1 Document Length
**Type:** Continuous (character count)  
**Implementation:**
```python
doc_length = len(text)
```
**Rationale:** Circulars and examination schedules tend to be longer; event announcements typically brief.

---

#### 5.2 Token Count
**Type:** Continuous (word count)  
**Implementation:**
```python
token_count = len(text.split())
```
**Rationale:** Length measure less sensitive to whitespace/formatting artifacts than character count.

---

#### 5.3 Mean Token Length
**Type:** Continuous (average characters per token)  
**Implementation:**
```python
tokens = text.split()
mean_length = sum(len(token) for token in tokens) / max(len(tokens), 1)
```
**Rationale:** Reflects vocabulary complexity; administrative circulars contain longer bureaucratic terminology than event announcements.

---

## Feature Extraction Implementation

### Complete Feature Extraction Function

```python
def extract_features(text):
    """
    Extract all 35 handcrafted features from OCR-extracted text.
    
    Args:
        text (str): Cleaned OCR-extracted notice text
        
    Returns:
        dict: Dictionary containing all 35 feature values
    """
    import re
    
    features = {}
    text_lower = text.lower()
    tokens = text.split()
    
    # ========== EXAMINATION FEATURES (9) ==========
    features['exam_term_freq'] = text_lower.count('exam') + text_lower.count('examination')
    
    exam_vocab = ['hall ticket', 'admit card', 'answer sheet', 'question paper',
                  'marks', 'grade', 'result', 'score', 'assessment']
    features['exam_vocab_presence'] = sum(text_lower.count(term) for term in exam_vocab)
    
    exam_count = text_lower.count('exam') + text_lower.count('examination')
    admission_count = text_lower.count('admission') + text_lower.count('admit')
    features['exam_admission_ratio'] = exam_count / (admission_count + 1)
    
    features['procedural_phrases'] = int(any(phrase in text_lower 
                                            for phrase in ['hall ticket', 'admit card']))
    
    features['assessment_materials'] = int(any(phrase in text_lower 
                                              for phrase in ['answer sheet', 'question paper']))
    
    exam_terms = ['exam', 'examination', 'test', 'assessment']
    features['exam_density'] = sum(text_lower.count(t) for t in exam_terms) / max(len(text), 1)
    
    has_exam = any(t in text_lower for t in ['exam', 'examination'])
    has_competing = any(t in text_lower for t in ['circular', 'admission', 'workshop'])
    features['exam_purity'] = int(has_exam and not has_competing)
    
    features['assessment_terminology'] = int(any(t in text_lower 
                                                for t in ['marks', 'grade', 'result', 'evaluation']))
    
    has_schedule = any(t in text_lower for t in ['timetable', 'schedule', 'date'])
    features['temporal_scheduling'] = int(has_exam and has_schedule)
    
    # ========== ADMISSION FEATURES (9) ==========
    features['admission_mentions'] = text_lower.count('admission') + text_lower.count('admit')
    
    admission_vocab = ['apply', 'application', 'eligibility', 'entrance',
                       'counselling', 'counseling', 'merit list', 'seat allotment']
    features['admission_vocab'] = sum(text_lower.count(term) for term in admission_vocab)
    
    features['merit_indicators'] = int(any(phrase in text_lower 
                                          for phrase in ['merit list', 'selected candidates', 'rank holder']))
    
    features['program_admission'] = int(any(t in text_lower 
                                           for t in ['phd', 'mphil', 'research admission']))
    
    features['counselling_markers'] = int('counsell' in text_lower)
    
    features['admission_density'] = (text_lower.count('admission') + 
                                     text_lower.count('admit')) / max(len(text), 1)
    
    features['eligibility_vocab'] = int(any(t in text_lower 
                                           for t in ['eligibility', 'eligible', 'qualification']))
    
    features['application_process'] = int(any(t in text_lower 
                                             for t in ['apply', 'application form', 'registration']))
    
    features['entrance_references'] = int('entrance' in text_lower)
    
    # ========== CIRCULAR FEATURES (8) ==========
    features['circular_freq'] = text_lower.count('circular') + text_lower.count('notification')
    
    numbering_pattern = r'(circular|notification)\s+(no\.|number)\s*[\d/\-]+'
    features['formal_numbering'] = int(bool(re.search(numbering_pattern, text_lower)))
    
    features['bureaucratic_markers'] = int(any(t in text_lower 
                                              for t in ['hereby', 'herewith', 'aforementioned']))
    
    features['notification_terminology'] = int(any(t in text_lower 
                                                  for t in ['notified', 'notification', 'inform']))
    
    has_circular = 'circular' in text_lower or 'notification' in text_lower
    has_bureaucratic = any(t in text_lower for t in ['hereby', 'herewith'])
    features['formal_cooccurrence'] = int(has_circular and has_bureaucratic)
    
    has_competing_circ = any(t in text_lower for t in ['exam', 'admission', 'workshop'])
    features['circular_dominance'] = int(has_circular and not has_competing_circ)
    
    features['addressee_formality'] = int(any(phrase in text_lower 
                                             for phrase in ['all concerned', 'to all faculty']))
    
    circular_terms = ['circular', 'notification', 'hereby', 'notified']
    features['circular_density'] = sum(text_lower.count(t) for t in circular_terms) / max(len(text), 1)
    
    # ========== EVENT FEATURES (6) ==========
    event_vocab = ['workshop', 'seminar', 'webinar', 'training', 'competition',
                   'orientation', 'inauguration', 'ceremony', 'program']
    features['event_vocab_richness'] = sum(text_lower.count(term) for term in event_vocab)
    
    features['common_event_types'] = int(any(t in text_lower 
                                            for t in ['workshop', 'seminar', 'webinar']))
    
    features['training_signals'] = int(any(t in text_lower 
                                          for t in ['training', 'orientation', 'induction']))
    
    has_event = any(t in text_lower for t in ['workshop', 'seminar', 'program'])
    has_exam_event = 'exam' in text_lower or 'admission' in text_lower
    features['event_purity'] = int(has_event and not has_exam_event)
    
    event_terms = ['workshop', 'seminar', 'webinar', 'program', 'event']
    features['event_density'] = sum(text_lower.count(t) for t in event_terms) / max(len(text), 1)
    
    features['ceremonial_markers'] = int(any(t in text_lower 
                                            for t in ['inauguration', 'ceremony', 'celebration']))
    
    # ========== STRUCTURAL FEATURES (3) ==========
    features['document_length'] = len(text)
    features['token_count'] = len(tokens)
    features['mean_token_length'] = sum(len(token) for token in tokens) / max(len(tokens), 1)
    
    return features
```

### Usage Example

```python
# Extract features from a notice
from src.ocr_classifier import OCRClassifier

text = """
Notification No. 45/2024
This is to inform all concerned that the final examination schedule 
for semester exams has been published. Students can download their 
hall tickets from the student portal.
"""

features = extract_features(text)
print(f"Total features: {len(features)}")
print(f"Examination signals: {features['exam_term_freq']}, {features['procedural_phrases']}")
print(f"Document structure: {features['document_length']} chars, {features['token_count']} tokens")
```

---

## Feature Importance Analysis

Based on aggregated importance scores from Random Forest, Gradient Boosting, and XGBoost models:

### Top 15 Most Influential Features

| Rank | Feature Name | Normalized Importance | Category |
|------|--------------|----------------------|----------|
| 1 | Numeric density (derived) | 0.92 | Structural |
| 2 | Document length | 0.88 | Structural |
| 3 | Exam term frequency | 0.84 | Examination |
| 4 | Admission mentions | 0.81 | Admission |
| 5 | Circular frequency | 0.78 | Circular |
| 6 | Event vocabulary richness | 0.76 | Event |
| 7 | Year tokens (derived) | 0.73 | Structural |
| 8 | Token count | 0.71 | Structural |
| 9 | Formal numbering | 0.68 | Circular |
| 10 | Counselling markers | 0.65 | Admission |
| 11 | Merit indicators | 0.63 | Admission |
| 12 | Common event types | 0.61 | Event |
| 13 | Procedural phrases | 0.59 | Examination |
| 14 | Assessment terminology | 0.57 | Examination |
| 15 | Bureaucratic markers | 0.55 | Circular |

**Note:** Numeric density and year tokens are derived features computed during TF-IDF vectorization but included here for context.

---

## Design Rationale

### Why 35 Features?

The feature count reflects a balance between:
1. **Coverage:** Addressing observed confusion patterns comprehensively
2. **Computational efficiency:** Keeping feature extraction fast (<5ms per document)
3. **Interpretability:** Each feature has clear semantic meaning
4. **Ablation results:** Removing these features reduces accuracy by 5.4 percentage points

### Why Domain-Specific vs. Generic?

Generic TF-IDF features alone achieved 86.5% accuracy. Adding these 35 domain features improved performance to 91.89% by:
- Capturing procedural distinctions (e.g., "hall ticket" vs. "admission ticket")
- Resolving ambiguous phrases (e.g., "circular regarding exam" - circular or exam?)
- Providing robustness to OCR noise (structural features remain stable despite token corruption)

---

## Extending the Feature Set

### For Other Domains

The feature engineering approach can be adapted to other administrative document types:

**Government Circulars:**
- Add features for legislative references (e.g., "Section X of Act Y")
- Policy numbering conventions
- Ministerial terminology

**Corporate Memos:**
- Department-specific keywords
- Urgency markers ("immediate", "urgent")
- Meeting references

**Medical Records:**
- Diagnosis terminology
- Prescription indicators
- Appointment scheduling markers

### For Multilingual Settings

Many features are language-agnostic:
- Document length, token count (universal)
- Numeric patterns (dates, numbers)
- Structural markers (punctuation, formatting)

Language-specific adaptations needed:
- Keyword lists (translate to target language)
- Bureaucratic register markers (identify formal language patterns)

---

## References

For complete methodology and ablation studies, see:
- **Paper:** "Ensemble-Based Four-Class OCR Document Classification for University Notices"
- **Repository:** https://github.com/chakrabt/OCR-Notice-Classifier
- **Contact:** tamalc@gmail.com

---

**Last Updated:** January 2025  
**Version:** 1.0
