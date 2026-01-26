"""
35 Precision Features + Optimized Voting Ensemble
"""

import joblib
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')


class OCRClassifier:
    """
    Production OCR Notice Classifier
    Categories: Examination, Admission, Circular, Event (4 categories, no Holiday)
    """

    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.le = None
        self.version = "1.0 - 35 Features (Voting)"

    # ------------------------------------------------------------------
    # FEATURE EXTRACTION (UNCHANGED)
    # ------------------------------------------------------------------
    def extract_35_features(self, text):
        text = str(text).lower()
        words = text.split()
        word_count = len(words)

        feat = {}

        # Examination (9)
        feat['exam_explicit'] = text.count('exam') + text.count('examination')
        feat['exam_keywords'] = sum(1 for w in
                                    ['hall', 'ticket', 'admit', 'card', 'answer', 'sheet',
                                     'question', 'paper', 'marks', 'grade'] if w in text)
        feat['exam_vs_admission_ratio'] = feat['exam_explicit'] / max(text.count('admission') + 1, 1)
        feat['has_hall_admit'] = int('hall ticket' in text or 'admit card' in text or 'hall-ticket' in text)
        feat['has_answer_question'] = int('answer sheet' in text or 'question paper' in text)
        feat['exam_density'] = feat['exam_explicit'] / max(word_count, 1)
        feat['is_pure_exam'] = int(feat['exam_explicit'] > 0 and 'admission' not in text and 'circular' not in text)
        feat['has_marks_grade'] = int('marks' in text or 'grade' in text or 'result' in text)
        feat['has_exam_schedule'] = int(('exam' in text or 'examination' in text) and 'schedule' in text)

        # Admission (9)
        feat['admission_explicit'] = text.count('admission')
        feat['admission_keywords'] = sum(1 for w in
                                         ['merit', 'list', 'counseling', 'counselling',
                                          'eligibility', 'apply', 'application', 'phd',
                                          'entrance', 'seat'] if w in text)
        feat['has_merit_list'] = int('merit' in text and 'list' in text)
        feat['has_phd_admission'] = int(('phd' in text or 'ph.d' in text) and 'admission' in text)
        feat['has_counseling'] = int('counseling' in text or 'counselling' in text)
        feat['admission_density'] = feat['admission_explicit'] / max(word_count, 1)
        feat['has_eligibility'] = int('eligibility' in text or 'eligible' in text)
        feat['has_apply'] = int('apply' in text or 'application' in text)
        feat['has_entrance'] = int('entrance' in text)

        # Circular (8)
        feat['circular_explicit'] = text.count('circular')
        feat['circular_no'] = int('circular no' in text or 'notification no' in text)
        feat['has_hereby'] = int('hereby' in text)
        feat['has_notified'] = int('notified' in text or 'notification' in text)
        feat['circular_formal'] = int(
            ('circular' in text or 'notification' in text) and
            ('hereby' in text or 'notified' in text)
        )
        feat['is_pure_circular'] = int('circular' in text and 'exam' not in text and 'admission' not in text)
        feat['has_all_concerned'] = int('all concerned' in text or 'whom it may concern' in text)
        feat['circular_density'] = feat['circular_explicit'] / max(word_count, 1)

        # Event (6)
        feat['event_keywords'] = sum(1 for w in
                                     ['workshop', 'seminar', 'webinar', 'training',
                                      'program', 'programme', 'orientation',
                                      'inauguration', 'competition'] if w in text)
        feat['has_workshop_seminar'] = int('workshop' in text or 'seminar' in text or 'webinar' in text)
        feat['has_training'] = int('training' in text or 'orientation' in text)
        feat['is_pure_event'] = int(feat['event_keywords'] > 0 and 'exam' not in text and 'admission' not in text)
        feat['event_density'] = feat['event_keywords'] / max(word_count, 1)
        feat['has_inauguration'] = int('inauguration' in text or 'ceremony' in text)

        # Structural (3)
        feat['text_length'] = len(text)
        feat['word_count'] = word_count
        feat['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0

        return np.array(list(feat.values())).reshape(1, -1)

    # ------------------------------------------------------------------
    # TRAINING (ENSEMBLE SWITCHED TO VOTING)
    # ------------------------------------------------------------------
    def train_production_model(self, train_path):
        print("=" * 80)
        print("TRAINING OCR CLASSIFIER (VOTING)")
        print("Version:", self.version)
        print("=" * 80)

        df = pd.read_csv(train_path, encoding='utf-8-sig')

        if 'holiday' in df['category'].values:
            df = df[df['category'] != 'holiday']

        feats = np.vstack([self.extract_35_features(t)[0] for t in df['extracted_text']])

        vectorizer = TfidfVectorizer(
            max_features=12000,
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.9,
            sublinear_tf=True,
            norm='l2'
        )
        X_tfidf = vectorizer.fit_transform(df['extracted_text'])

        X = hstack([X_tfidf, feats]).toarray()

        le = LabelEncoder()
        y = le.fit_transform(df['category'])

        smote = SMOTE(
            random_state=42,
            k_neighbors=min(5, min(pd.Series(y).value_counts()) - 1)
        )
        X_bal, y_bal = smote.fit_resample(X, y)

        rf = RandomForestClassifier(
            n_estimators=1000,
            max_depth=20,
            min_samples_split=3,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        gb = GradientBoostingClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=42
        )

        lr = LogisticRegression(
            C=2.0,
            class_weight='balanced',
            max_iter=2000,
            random_state=42
        )

        try:
            from xgboost import XGBClassifier
            xgb = XGBClassifier(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.07,
                subsample=0.8,
                random_state=42,
                eval_metric='mlogloss'
            )

            ensemble = VotingClassifier(
                estimators=[('rf', rf), ('gb', gb), ('lr', lr), ('xgb', xgb)],
                voting='soft',
                weights=[3, 2, 1, 3],
                n_jobs=1
            )
        except ImportError:
            ensemble = VotingClassifier(
                estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
                voting='soft',
                weights=[3, 2, 1],
                n_jobs=1
            )

        ensemble.fit(X_bal, y_bal)

        joblib.dump(ensemble, 'ocr_model.pkl')
        joblib.dump(vectorizer, 'ocr_vectorizer.pkl')
        joblib.dump(le, 'ocr_encoder.pkl')

        print("✔ Voting-based production model saved")

    # ------------------------------------------------------------------
    # PREDICTION (UNCHANGED)
    # ------------------------------------------------------------------
    def load_model(self):
        self.model = joblib.load('ocr_model.pkl')
        self.vectorizer = joblib.load('ocr_vectorizer.pkl')
        self.le = joblib.load('ocr_encoder.pkl')

    def predict(self, text):
        if self.model is None:
            self.load_model()

        feats = self.extract_35_features(text)
        tfidf = self.vectorizer.transform([text])
        X = hstack([tfidf, feats]).toarray()

        pred = self.model.predict(X)[0]
        probs = self.model.predict_proba(X)[0]

        raw_probs = self.model.predict_proba(X)[0]
        all_probs = dict(zip(self.le.classes_, raw_probs))

        pred_idx = int(np.argmax(raw_probs))
        predicted_label = self.le.classes_[pred_idx]
        confidence = float(raw_probs[pred_idx])

        return {
            'category': predicted_label,
            'confidence': f"{confidence:.1%}",
            'confidence_score': confidence,
            'probabilities': {k: f"{v:.1%}" for k, v in all_probs.items()},
            'probabilities_raw': all_probs
        }

    def predict_batch(self, texts):
        """
        Batch prediction for multiple OCR notices
        Returns a list of prediction dictionaries
        """
        if self.model is None:
            self.load_model()

        results = []
        for text in texts:
            results.append(self.predict(text))

        return results

if __name__ == "__main__":
    print("\n" + "="*80)
    print("OCR NOTICE CLASSIFIER - VOTING ENSEMBLE")
    print("="*80)

    classifier = OCRClassifier()
    classifier.train_production_model(
        'final_dataset/train_data_no_holiday.csv'
    )

    print("\n✅ Training finished. Model files created.")
