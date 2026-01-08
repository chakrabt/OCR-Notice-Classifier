"""
Production OCR Notice Classifier

Trains and serves a stacked ensemble classifier
with TF-IDF and engineered features.
"""

from __future__ import annotations

import joblib
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy.sparse import hstack
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


class OCRClassifier:
    """Production-ready OCR notice classifier."""

    def __init__(self) -> None:
        self.model = None
        self.vectorizer = None
        self.encoder = None

    def train(self, csv_path: str) -> None:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        df = df[df["category"] != "holiday"]

        self.vectorizer = TfidfVectorizer(
            max_features=12000,
            ngram_range=(1, 3),
            sublinear_tf=True,
        )
        X = self.vectorizer.fit_transform(df["extracted_text"])

        self.encoder = LabelEncoder()
        y = self.encoder.fit_transform(df["category"])

        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X.toarray(), y)

        rf = RandomForestClassifier(
            n_estimators=1000,
            max_depth=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        gb = GradientBoostingClassifier(
            n_estimators=400,
            learning_rate=0.05,
            random_state=42,
        )

        svm = SVC(
            kernel="rbf",
            C=1.5,
            gamma="scale",
            probability=True,
            class_weight="balanced",
        )

        meta = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )

        self.model = StackingClassifier(
            estimators=[("rf", rf), ("gb", gb), ("svm", svm)],
            final_estimator=meta,
            cv=5,
            n_jobs=-1,
        )

        self.model.fit(X, y)

        joblib.dump(self.model, "model.pkl")
        joblib.dump(self.vectorizer, "vectorizer.pkl")
        joblib.dump(self.encoder, "encoder.pkl")

    def load(self) -> None:
        self.model = joblib.load("model.pkl")
        self.vectorizer = joblib.load("vectorizer.pkl")
        self.encoder = joblib.load("encoder.pkl")

    def predict(self, text: str) -> Dict:
        if self.model is None:
            self.load()

        X = self.vectorizer.transform([text]).toarray()
        proba = self.model.predict_proba(X)[0]
        idx = proba.argmax()

        return {
            "category": self.encoder.inverse_transform([idx])[0],
            "confidence": float(proba[idx]),
            "probabilities": dict(zip(self.encoder.classes_, proba)),
        }
