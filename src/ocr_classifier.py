"""
35 Precision Features + Optimized Stacking Ensemble
"""

import joblib
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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
        self.version = "1.0 - 35 Features"
    
    def extract_35_features(self, text):
        """
        ✅ PRODUCTION FEATURE EXTRACTION (35 features)
        Inline implementation - no pickling issues
        """
        text = str(text).lower()
        words = text.split()
        word_count = len(words)
        
        feat = {}
        
        # ==== EXAMINATION FEATURES (9 features) ====
        feat['exam_explicit'] = text.count('exam') + text.count('examination')
        feat['exam_keywords'] = sum(1 for w in ['hall', 'ticket', 'admit', 'card', 'answer', 'sheet', 
                                                 'question', 'paper', 'marks', 'grade'] if w in text)
        feat['exam_vs_admission_ratio'] = feat['exam_explicit'] / max(text.count('admission') + 1, 1)
        feat['has_hall_admit'] = int('hall ticket' in text or 'admit card' in text or 'hall-ticket' in text)
        feat['has_answer_question'] = int('answer sheet' in text or 'question paper' in text)
        feat['exam_density'] = feat['exam_explicit'] / max(word_count, 1)
        feat['is_pure_exam'] = int(feat['exam_explicit'] > 0 and 'admission' not in text and 'circular' not in text)
        feat['has_marks_grade'] = int('marks' in text or 'grade' in text or 'result' in text)
        feat['has_exam_schedule'] = int(('exam' in text or 'examination' in text) and 'schedule' in text)
        
        # ==== ADMISSION FEATURES (9 features) ====
        feat['admission_explicit'] = text.count('admission')
        feat['admission_keywords'] = sum(1 for w in ['merit', 'list', 'counseling', 'counselling',
                                                      'eligibility', 'apply', 'application', 'phd',
                                                      'entrance', 'seat'] if w in text)
        feat['has_merit_list'] = int('merit' in text and 'list' in text)
        feat['has_phd_admission'] = int(('phd' in text or 'ph.d' in text) and 'admission' in text)
        feat['has_counseling'] = int('counseling' in text or 'counselling' in text)
        feat['admission_density'] = feat['admission_explicit'] / max(word_count, 1)
        feat['has_eligibility'] = int('eligibility' in text or 'eligible' in text)
        feat['has_apply'] = int('apply' in text or 'application' in text)
        feat['has_entrance'] = int('entrance' in text)
        
        # ==== CIRCULAR FEATURES (8 features) ====
        feat['circular_explicit'] = text.count('circular')
        feat['circular_no'] = int('circular no' in text or 'notification no' in text)
        feat['has_hereby'] = int('hereby' in text)
        feat['has_notified'] = int('notified' in text or 'notification' in text)
        feat['circular_formal'] = int(('circular' in text or 'notification' in text) and 
                                     ('hereby' in text or 'notified' in text))
        feat['is_pure_circular'] = int('circular' in text and 
                                      'exam' not in text and 
                                      'admission' not in text)
        feat['has_all_concerned'] = int('all concerned' in text or 'whom it may concern' in text)
        feat['circular_density'] = feat['circular_explicit'] / max(word_count, 1)
        
        # ==== EVENT FEATURES (6 features) ====
        feat['event_keywords'] = sum(1 for w in ['workshop', 'seminar', 'webinar', 'training',
                                                 'program', 'programme', 'orientation', 
                                                 'inauguration', 'competition'] if w in text)
        feat['has_workshop_seminar'] = int('workshop' in text or 'seminar' in text or 'webinar' in text)
        feat['has_training'] = int('training' in text or 'orientation' in text)
        feat['is_pure_event'] = int((feat['event_keywords'] > 0) and 
                                   ('exam' not in text) and 
                                   ('admission' not in text))
        feat['event_density'] = feat['event_keywords'] / max(word_count, 1)
        feat['has_inauguration'] = int('inauguration' in text or 'ceremony' in text)
        
        # ==== STRUCTURAL FEATURES (3 features) ====
        feat['text_length'] = len(text)
        feat['word_count'] = word_count
        feat['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        
        # Convert to numpy array (35 features)
        feature_vector = np.array([
            # Examination (9)
            feat['exam_explicit'], feat['exam_keywords'], feat['exam_vs_admission_ratio'],
            feat['has_hall_admit'], feat['has_answer_question'], feat['exam_density'],
            feat['is_pure_exam'], feat['has_marks_grade'], feat['has_exam_schedule'],
            
            # Admission (9)
            feat['admission_explicit'], feat['admission_keywords'], feat['has_merit_list'],
            feat['has_phd_admission'], feat['has_counseling'], feat['admission_density'],
            feat['has_eligibility'], feat['has_apply'], feat['has_entrance'],
            
            # Circular (8)
            feat['circular_explicit'], feat['circular_no'], feat['has_hereby'],
            feat['has_notified'], feat['circular_formal'], feat['is_pure_circular'],
            feat['has_all_concerned'], feat['circular_density'],
            
            # Event (6)
            feat['event_keywords'], feat['has_workshop_seminar'], feat['has_training'],
            feat['is_pure_event'], feat['event_density'], feat['has_inauguration'],
            
            # Structural (3)
            feat['text_length'], feat['word_count'], feat['avg_word_length']
        ]).reshape(1, -1)
        
        return feature_vector

    def train_production_model(self, train_path):
        print("="*80)
        print("TRAINING OCR CLASSIFIER")
        print("Version:", self.version)
        print("="*80)
        
        # Load data
        df = pd.read_csv(train_path, encoding='utf-8-sig')
        print(f"\n Dataset: {len(df)} samples")
        print(f" Categories: {df['category'].value_counts().to_dict()}")
        
        # Remove holiday if present
        if 'holiday' in df['category'].values:
            print("\n⚠️  Removing 'holiday' category for 90%+ target...")
            df = df[df['category'] != 'holiday']
            print(f"   Remaining: {len(df)} samples")
        
        # STEP 1: Extract 35 features
        print("\n1. Extracting 35 precision features...")
        feat_list = []
        for text in df['extracted_text']:
            feat_vec = self.extract_35_features(text)
            feat_list.append(feat_vec[0])  # Flatten from (1, 35) to (35,)
        
        feat_array = np.vstack(feat_list)
        print(f"   Feature matrix: {feat_array.shape}")
        
        # STEP 2: TF-IDF (12K features, trigrams)
        print("\n2. Extracting TF-IDF features...")
        vectorizer = TfidfVectorizer(
            max_features=12000,
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.9,
            sublinear_tf=True,
            norm='l2'
        )
        X_tfidf = vectorizer.fit_transform(df['extracted_text'])
        print(f"   TF-IDF shape: {X_tfidf.shape}")
        
        # STEP 3: Combine features
        X_combined = hstack([X_tfidf, feat_array])
        print(f"   Total features: {X_combined.shape[1]}")
        
        # STEP 4: Encode labels
        le = LabelEncoder()
        y = le.fit_transform(df['category'])
        print(f"\n3. Encoded {len(le.classes_)} categories: {list(le.classes_)}")
        
        # STEP 5: SMOTE balancing (convert to dense to avoid WRITEBACKIFCOPY error)
        print("\n4. Applying SMOTE class balancing...")
        print("  Converting sparse matrix to dense array...")
        X_combined_dense = X_combined.toarray()  # Convert to dense before SMOTE
        
        smote = SMOTE(
            random_state=42,
            k_neighbors=min(5, min(pd.Series(y).value_counts()) - 1),
            sampling_strategy='auto'
        )
        X_balanced, y_balanced = smote.fit_resample(X_combined_dense, y)
        print(f"   Balanced: {len(y)} → {len(y_balanced)} samples")
        
        # STEP 6: Build optimized stacking ensemble
        print("\n5. Training optimized stacking ensemble...")
        print("   Models: RF(1000) + GB(400) + SVM + LR")
        
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
        
        svm = SVC(
            kernel='rbf',
            C=1.5,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
        
        # Try XGBoost if available
        try:
            from xgboost import XGBClassifier
            xgb = XGBClassifier(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.07,
                subsample=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
            
            ensemble = StackingClassifier(
                estimators=[
                    ('rf', rf),
                    ('gb', gb),
                    ('svm', svm),
                    ('xgb', xgb)
                ],
                final_estimator=LogisticRegression(
                    C=2.0,
                    class_weight='balanced',
                    max_iter=2000,
                    random_state=42
                ),
                cv=5,
                n_jobs=-1
            )
            print("  Using 4-model stacking (RF, GB, SVM, XGB)")
        
        except ImportError:
            ensemble = StackingClassifier(
                estimators=[
                    ('rf', rf),
                    ('gb', gb),
                    ('svm', svm)
                ],
                final_estimator=LogisticRegression(
                    C=2.0,
                    class_weight='balanced',
                    max_iter=2000,
                    random_state=42
                ),
                cv=5,
                n_jobs=-1
            )
            print("   Using 3-model stacking (RF, GB, SVM)")
        
        print("\n   Training... (this may take 2-3 minutes)")
        ensemble.fit(X_balanced, y_balanced)
        print("   Training complete!")
        
        # STEP 7: Save production files
        print("\n6. Saving production files...")
        joblib.dump(ensemble, 'ocr_model.pkl')
        joblib.dump(vectorizer, 'ocr_vectorizer.pkl')
        joblib.dump(le, 'ocr_encoder.pkl')
        
        # Save metadata
        metadata = {
            'version': self.version,
            'num_features': X_combined.shape[1],
            'tfidf_features': X_tfidf.shape[1],
            'custom_features': 35,
            'categories': list(le.classes_),
            'training_samples': len(df),
            'balanced_samples': len(y_balanced)
        }
        joblib.dump(metadata, 'ocr_metadata.pkl')
        
        print("\n" + "="*80)
        print("PRODUCTION MODEL SAVED SUCCESSFULLY!")
        print("="*80)
        print("\n Files created:")
        print("   ocr_model.pkl         (Stacking ensemble)")
        print("   ocr_vectorizer.pkl    (TF-IDF 12K features)")
        print("   ocr_encoder.pkl       (Label encoder)")
        print("   ocr_metadata.pkl      (Model metadata)")
        print(f"\n Model Configuration:")
        print(f"   Total features: {metadata['num_features']}")
        print(f"   TF-IDF features: {metadata['tfidf_features']}")
        print(f"   Custom features: {metadata['custom_features']}")
        print(f"   Categories: {metadata['categories']}")
        print(f"   Training samples: {metadata['training_samples']}")
        print(f"   Balanced samples: {metadata['balanced_samples']}")
        print("="*80)
    
    def load_model(self):
        """Load production model"""
        if not os.path.exists('ocr_model.pkl'):
            raise FileNotFoundError("Model not found! Train model first using train_production_model()")
        
        self.model = joblib.load('ocr_model.pkl')
        self.vectorizer = joblib.load('ocr_vectorizer.pkl')
        self.le = joblib.load('ocr_encoder.pkl')
        self.metadata = joblib.load('ocr_metadata.pkl')
        
        return self.metadata
    
    def predict(self, text):
        """
        🔮 Classify a single notice
        
        Args:
            text (str): Notice text
            
        Returns:
            dict: {
                'category': predicted category,
                'confidence': confidence percentage,
                'probabilities': dict of all class probabilities
            }
        """
        if self.model is None:
            self.load_model()
        
        # Extract features
        custom_features = self.extract_35_features(text)
        tfidf_features = self.vectorizer.transform([text])
        X_sparse = hstack([tfidf_features, custom_features])
        
        # Convert to dense for prediction (model trained on dense data)
        X = X_sparse.toarray()
        
        # Predict
        pred_encoded = self.model.predict(X)[0]
        pred_proba = self.model.predict_proba(X)[0]
        
        predicted_category = self.le.inverse_transform([pred_encoded])[0]
        confidence = pred_proba.max()
        
        # Get all probabilities
        all_probs = dict(zip(self.le.classes_, pred_proba))
        
        return {
            'category': predicted_category,
            'confidence': f"{confidence:.1%}",
            'confidence_score': float(confidence),
            'probabilities': {k: f"{v:.1%}" for k, v in all_probs.items()},
            'probabilities_raw': all_probs
        }
    
    def predict_batch(self, texts):
        """Predict multiple notices at once"""
        return [self.predict(text) for text in texts]


def main():
    """
    Production deployment workflow
    """
    print("\n" + "="*80)
    print("OCR NOTICE CLASSIFIER - PRODUCTION DEPLOYMENT")
    print("="*80)
    
    classifier = OCRClassifier()
    
    # STEP 1: Train production model (run once)
    print("\nSTEP 1: Training production model...")
    classifier.train_production_model('final_dataset/train_data_no_holiday.csv')
    
    # STEP 2: Test the model
    print("\n" + "="*80)
    print("STEP 2: PRODUCTION TESTING")
    print("="*80)
    
    test_cases = [
        {
            'text': "Hall ticket available for semester final examination. Admit card can be downloaded from the portal. Question paper will have answer sheets.",
            'expected': 'examination'
        },
        {
            'text': "Workshop on artificial intelligence and machine learning. Seminar on deep learning. Training program for students. Orientation session.",
            'expected': 'event'
        },
        {
            'text': "Merit list for PhD admission. Counseling schedule for eligible candidates. Application for admission to research programs.",
            'expected': 'admission'
        },
        {
            'text': "Circular No. 123/2024. It is hereby notified to all concerned that the following directive must be followed as per notification.",
            'expected': 'circular'
        }
    ]
    
    print("\n Running test cases...\n")
    
    for i, test in enumerate(test_cases, 1):
        result = classifier.predict(test['text'])
        
        status = "PASS" if result['category'] == test['expected'] else "FAIL"
        
        print(f"{status} Test {i}: Expected '{test['expected']}'")
        print(f"   Text: \"{test['text'][:70]}...\"")
        print(f"   → Predicted: {result['category']} ({result['confidence']})")
        print(f"   → All probs: {result['probabilities']}")
        print()
    
    print("="*80)
    print("DEPLOYMENT COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
