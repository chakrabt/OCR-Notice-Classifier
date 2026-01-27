# Production Deployment Guide

**Project:** Ensemble-Based OCR Document Classification for University Notices  
**Author:** Tamal Chakraborty  
**Repository:** https://github.com/chakrabt/OCR-Notice-Classifier

---

## Overview

This guide provides comprehensive instructions for deploying the OCR-based notice classification system in production university environments. The system has been validated on held-out test data (91.89% accuracy) and is ready for institutional integration.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Deployment-Ready Components](#deployment-ready-components)
3. [Integration Pathways](#integration-pathways)
4. [Phased Deployment Roadmap](#phased-deployment-roadmap)
5. [Pre-Deployment Checklist](#pre-deployment-checklist)
6. [Monitoring and Maintenance](#monitoring-and-maintenance)
7. [Troubleshooting](#troubleshooting)
8. [Scaling Considerations](#scaling-considerations)

---

## System Requirements

### Hardware Requirements

**Minimum:**
- CPU: 2 cores, 2.0 GHz
- RAM: 4 GB
- Storage: 2 GB (model artifacts + dependencies)
- No GPU required

**Recommended for Production:**
- CPU: 4+ cores, 2.5+ GHz (for parallel processing)
- RAM: 8 GB
- Storage: 10 GB (includes logs, monitoring data)
- SSD preferred for faster model loading

### Software Requirements

```
Python >= 3.8
scikit-learn >= 1.0.0, < 1.3.0
xgboost >= 1.5.0, < 3.0.0
imbalanced-learn >= 0.9.0, < 1.0.0
pandas >= 1.3.0, < 3.0.0
numpy >= 1.21.0, < 2.0.0
joblib >= 1.1.0, < 2.0.0
```

**Install via:**
```bash
pip install -r requirements.txt
```

### Operating System

- Linux (Ubuntu 20.04+, CentOS 7+) - Recommended
- Windows 10/11 - Supported
- macOS 11+ - Supported

---

## Deployment-Ready Components

The following components have been developed and tested:

### ✅ Core Components

1. **Trained Ensemble Model**
   - File: `ocr_model.pkl` (~450 MB)
   - Accuracy: 91.89% on test set
   - Components: Random Forest, Gradient Boosting, XGBoost, Logistic Regression meta-learner

2. **TF-IDF Vectorizer**
   - File: `ocr_vectorizer.pkl`
   - Features: 12,000 TF-IDF features + 35 handcrafted indicators
   - Frozen vocabulary for consistency

3. **Label Encoder**
   - File: `ocr_encoder.pkl`
   - Categories: Examination, Admission, Circular, Event

4. **Metadata**
   - File: `ocr_metadata.pkl`
   - Training configuration, feature names, version info

5. **Python API**
   - Module: `src/ocr_classifier.py`
   - Class: `OCRClassifier`
   - Methods: `predict()`, `predict_batch()`, `load_model()`, `train_production_model()`

---

## Integration Pathways

### Option A: REST API Service

**Best for:** Web-based systems, distributed architectures, microservices

#### A.1 Flask-Based Deployment

```python
# app.py
from flask import Flask, request, jsonify
from src.ocr_classifier import OCRClassifier
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load model once at startup
classifier = OCRClassifier()
classifier.load_model()
logging.info("Model loaded successfully")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": True}), 200

@app.route('/classify', methods=['POST'])
def classify_notice():
    """
    Classify a single notice
    
    Request JSON:
    {
        "text": "Notice text here...",
        "return_probabilities": true  // optional
    }
    
    Response JSON:
    {
        "category": "examination",
        "confidence": "96.6%",
        "confidence_score": 0.966,
        "probabilities": {...},  // if requested
        "processing_time_ms": 42
    }
    """
    try:
        data = request.get_json()
        
        if 'text' not in data:
            return jsonify({"error": "Missing 'text' field"}), 400
        
        import time
        start = time.time()
        
        result = classifier.predict(data['text'])
        
        processing_time = (time.time() - start) * 1000  # ms
        result['processing_time_ms'] = round(processing_time, 2)
        
        if not data.get('return_probabilities', False):
            result.pop('probabilities_raw', None)
        
        logging.info(f"Classified as {result['category']} ({result['confidence']})")
        
        return jsonify(result), 200
        
    except Exception as e:
        logging.error(f"Classification error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/classify/batch', methods=['POST'])
def classify_batch():
    """
    Classify multiple notices
    
    Request JSON:
    {
        "texts": ["Notice 1...", "Notice 2...", ...]
    }
    
    Response JSON:
    {
        "results": [...],
        "total": 10,
        "processing_time_ms": 420
    }
    """
    try:
        data = request.get_json()
        
        if 'texts' not in data or not isinstance(data['texts'], list):
            return jsonify({"error": "Missing or invalid 'texts' field"}), 400
        
        import time
        start = time.time()
        
        results = classifier.predict_batch(data['texts'])
        
        processing_time = (time.time() - start) * 1000
        
        return jsonify({
            "results": results,
            "total": len(results),
            "processing_time_ms": round(processing_time, 2)
        }), 200
        
    except Exception as e:
        logging.error(f"Batch classification error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

#### A.2 Docker Containerization

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY app.py .
COPY ocr_model.pkl ocr_vectorizer.pkl ocr_encoder.pkl ocr_metadata.pkl ./

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Run application
CMD ["python", "app.py"]
```

**Build and run:**
```bash
# Build image
docker build -t ocr-classifier:latest .

# Run container
docker run -d \
  --name ocr-classifier \
  -p 5000:5000 \
  --memory=2g \
  --cpus=2 \
  ocr-classifier:latest

# Test
curl -X POST http://localhost:5000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Final examination schedule published..."}'
```

**Estimated implementation time:** 2-3 weeks

---

### Option B: Scheduled Batch Processing

**Best for:** Nightly processing, periodic bulk classification, existing document pipelines

#### B.1 Cron Job Integration

```python
# batch_classifier.py
import sys
import pandas as pd
from datetime import datetime
from src.ocr_classifier import OCRClassifier
import logging

logging.basicConfig(
    filename='/var/log/ocr_classifier/batch.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def process_notices(input_csv, output_csv):
    """
    Process all notices from input CSV and save results
    
    Args:
        input_csv: Path to CSV with 'id' and 'text' columns
        output_csv: Path to save results with 'id', 'text', 'category', 'confidence'
    """
    try:
        # Load model
        logging.info(f"Loading model...")
        classifier = OCRClassifier()
        classifier.load_model()
        
        # Read input
        logging.info(f"Reading {input_csv}...")
        df = pd.read_csv(input_csv)
        
        if 'text' not in df.columns:
            raise ValueError("Input CSV must have 'text' column")
        
        # Classify
        logging.info(f"Classifying {len(df)} notices...")
        results = classifier.predict_batch(df['text'].tolist())
        
        # Add results to dataframe
        df['predicted_category'] = [r['category'] for r in results]
        df['confidence'] = [r['confidence'] for r in results]
        df['confidence_score'] = [r['confidence_score'] for r in results]
        
        # Flag low-confidence predictions
        df['needs_review'] = df['confidence_score'] < 0.80
        
        # Save results
        df.to_csv(output_csv, index=False)
        logging.info(f"Results saved to {output_csv}")
        
        # Summary statistics
        summary = df['predicted_category'].value_counts()
        low_conf_count = df['needs_review'].sum()
        
        logging.info(f"Classification summary:")
        for category, count in summary.items():
            logging.info(f"  {category}: {count}")
        logging.info(f"Low-confidence predictions: {low_conf_count} ({low_conf_count/len(df)*100:.1f}%)")
        
        return True
        
    except Exception as e:
        logging.error(f"Batch processing failed: {str(e)}")
        return False

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python batch_classifier.py input.csv output.csv")
        sys.exit(1)
    
    success = process_notices(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)
```

#### B.2 Crontab Configuration

```bash
# Add to crontab (crontab -e)

# Run every night at 2 AM
0 2 * * * /usr/bin/python3 /path/to/batch_classifier.py \
  /data/notices/pending.csv \
  /data/notices/classified_$(date +\%Y\%m\%d).csv \
  >> /var/log/ocr_classifier/cron.log 2>&1

# Archive old classified files monthly
0 3 1 * * find /data/notices/classified_*.csv -mtime +30 -exec gzip {} \;
```

**Estimated implementation time:** 1-2 weeks

---

### Option C: Direct Python Integration

**Best for:** Existing Python applications, data processing pipelines, custom workflows

#### C.1 Integration Example

```python
# your_application.py
from src.ocr_classifier import OCRClassifier

# Initialize once at application startup
classifier = OCRClassifier()
classifier.load_model()

def process_uploaded_notice(notice_file):
    """
    Process a newly uploaded notice PDF
    """
    # Extract text via OCR (your existing pipeline)
    text = extract_text_from_pdf(notice_file)
    
    # Classify
    result = classifier.predict(text)
    
    # Route based on category
    if result['confidence_score'] >= 0.80:
        # Auto-route high-confidence predictions
        route_to_department(
            notice_file,
            department=get_department_for_category(result['category']),
            priority=get_priority(result['category'])
        )
        log_classification(notice_file, result, auto_routed=True)
    else:
        # Flag low-confidence for manual review
        flag_for_review(
            notice_file,
            predicted_category=result['category'],
            confidence=result['confidence'],
            probabilities=result['probabilities']
        )
        log_classification(notice_file, result, auto_routed=False)

def get_department_for_category(category):
    """Map categories to departments"""
    routing = {
        'examination': 'Exam Cell',
        'admission': 'Admissions Office',
        'circular': 'Administration',
        'event': 'Student Affairs'
    }
    return routing.get(category, 'General Administration')

def get_priority(category):
    """Assign priority levels"""
    priority_map = {
        'examination': 'HIGH',
        'admission': 'HIGH',
        'circular': 'MEDIUM',
        'event': 'LOW'
    }
    return priority_map.get(category, 'MEDIUM')
```

**Estimated implementation time:** 1 week

---

## Phased Deployment Roadmap

### Phase 1: Shadow Mode Deployment (Month 1-2)

**Goal:** Validate accuracy in real operational conditions without affecting current workflows

#### Activities

1. **Setup parallel classification**
   - System classifies all incoming notices
   - Predictions logged but NOT used for routing
   - Manual classification continues as normal

2. **Monitor performance**
   ```python
   # monitoring.py
   def compare_predictions_to_manual():
       """Compare model predictions to manual classifications"""
       results = load_shadow_mode_results()
       manual = load_manual_classifications()
       
       agreement = calculate_agreement(results, manual)
       confusion = build_confusion_matrix(results, manual)
       
       print(f"Agreement: {agreement:.2%}")
       print(f"Confusion matrix:\n{confusion}")
       
       # Identify systematic errors
       errors = identify_disagreements(results, manual)
       analyze_error_patterns(errors)
   ```

3. **Weekly review meetings**
   - Review classification agreement
   - Analyze error patterns
   - Gather administrator feedback

#### Success Criteria

- ✅ **>88% agreement** with manual classification
- ✅ **<5% severe misclassifications** (e.g., exam notice routed to events)
- ✅ **Stable performance** across 2 months

#### Outputs

- Performance report
- Error pattern analysis
- Recommended confidence thresholds
- Administrator trust assessment

---

### Phase 2: Partial Deployment (Month 3-4)

**Goal:** Reduce manual workload while maintaining quality through selective automation

#### Activities

1. **Implement confidence-based routing**
   ```python
   def route_notice(text):
       result = classifier.predict(text)
       
       if result['confidence_score'] >= 0.85:
           # HIGH CONFIDENCE: Auto-route
           route_automatically(result['category'])
           log_decision('auto_routed', result)
       
       elif result['confidence_score'] >= 0.70:
           # MEDIUM CONFIDENCE: Flag for quick review
           flag_for_review(result, priority='medium')
           log_decision('flagged_medium', result)
       
       else:
           # LOW CONFIDENCE: Manual classification
           send_to_manual_queue()
           log_decision('manual', result)
   ```

2. **Daily monitoring dashboard**
   - Auto-routing rate
   - Error rate on auto-routed cases
   - Review queue size
   - Administrator override frequency

3. **Bi-weekly calibration**
   - Adjust confidence thresholds based on performance
   - Retrain on accumulated corrections

#### Success Criteria

- ✅ **>70% auto-routing rate** (high-confidence predictions)
- ✅ **<5% misrouting rate** for auto-routed cases
- ✅ **<3% administrator override** rate

#### Outputs

- Optimized confidence thresholds
- Auto-routing performance report
- Administrator satisfaction survey

---

### Phase 3: Full Deployment with Oversight (Month 5-6)

**Goal:** Achieve sustained automation with safety mechanisms

#### Activities

1. **Expand auto-routing to medium-confidence**
   - Gradually lower threshold from 0.85 to 0.75
   - Monitor error rates continuously

2. **Implement continuous monitoring**
   ```python
   # monitor_drift.py
   from datetime import datetime, timedelta
   
   def detect_concept_drift():
       """Monitor for distribution shifts"""
       recent = get_predictions(days=7)
       baseline = load_baseline_distribution()
       
       # Compare category distributions
       drift_score = calculate_kl_divergence(recent, baseline)
       
       if drift_score > 0.15:
           alert_administrators()
           recommend_retraining()
   ```

3. **Enable administrator feedback loop**
   - Easy override interface
   - Corrections fed into retraining pipeline

#### Success Criteria

- ✅ **>90% automation rate**
- ✅ **<3% error rate** overall
- ✅ **<2 hours/week** manual intervention time

#### Outputs

- Full production deployment
- Monitoring dashboards
- Feedback collection system

---

### Phase 4: Production Operation (Ongoing)

**Goal:** Maintain long-term reliability and continuous improvement

#### Activities

1. **Quarterly retraining**
   ```bash
   # retrain.sh
   #!/bin/bash
   
   # Collect corrections from last quarter
   python collect_corrections.py --since="3 months ago" --output=new_training_data.csv
   
   # Combine with original training data
   python merge_datasets.py original_data.csv new_training_data.csv combined.csv
   
   # Retrain model
   python -c "
   from src.ocr_classifier import OCRClassifier
   clf = OCRClassifier()
   clf.train_production_model('combined.csv')
   "
   
   # Validate on held-out test set
   python validate_model.py
   
   # If validation passed, deploy
   if [ $? -eq 0 ]; then
       cp ocr_model.pkl ocr_model_$(date +%Y%m%d).pkl.backup
       mv new_ocr_model.pkl ocr_model.pkl
       systemctl restart ocr-classifier
       echo "Model updated successfully"
   fi
   ```

2. **Performance degradation triggers**
   - **Alert trigger:** Accuracy drops below 88%
   - **Retraining trigger:** Accuracy drops below 85%
   - **Emergency trigger:** Accuracy drops below 80% (immediate manual review mode)

3. **Monthly reporting**
   - Classification volume
   - Accuracy trends
   - Category distribution shifts
   - Time/cost savings

#### Success Criteria

- ✅ **Sustained accuracy >88%** over 12 months
- ✅ **<5% drift** in category distributions
- ✅ **Administrator satisfaction >80%**

---

## Pre-Deployment Checklist

### Infrastructure Setup

- [ ] **Hardware provisioned** (CPU, RAM, storage per requirements)
- [ ] **Python environment** configured (version 3.8+)
- [ ] **Dependencies installed** (`pip install -r requirements.txt`)
- [ ] **Model artifacts downloaded** (contact author if needed)
- [ ] **Log directories created** (`/var/log/ocr_classifier/`)
- [ ] **Data directories created** (`/data/notices/{pending,classified,archive}`)

### Security & Access

- [ ] **API authentication** implemented (if using REST API)
- [ ] **SSL/TLS certificates** configured (for HTTPS)
- [ ] **Network firewall rules** configured
- [ ] **User permissions** set correctly
- [ ] **PII handling** procedures documented

### Testing

- [ ] **Unit tests** passing (`pytest tests/`)
- [ ] **Integration tests** passing
- [ ] **Load testing** completed (target: 1,000+ docs/hour)
- [ ] **Failover scenarios** tested

### Monitoring

- [ ] **Logging configured** (application + system logs)
- [ ] **Monitoring dashboard** deployed
- [ ] **Alert rules** configured (accuracy drops, errors, resource usage)
- [ ] **Backup procedures** established

### Documentation

- [ ] **Administrator training** completed
- [ ] **User guide** distributed
- [ ] **Runbooks** created (common issues, troubleshooting)
- [ ] **Contact procedures** documented

### Compliance

- [ ] **Data privacy review** completed
- [ ] **Security audit** passed
- [ ] **Institutional approval** obtained
- [ ] **SLA defined** (uptime, response time)

---

## Monitoring and Maintenance

### Key Metrics to Track

#### Performance Metrics

```python
# metrics.py
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ClassificationMetrics:
    timestamp: datetime
    total_classified: int
    accuracy: float
    precision_by_category: dict
    recall_by_category: dict
    avg_confidence: float
    low_confidence_rate: float
    processing_time_p50: float
    processing_time_p95: float

def log_metrics(metrics):
    """Log metrics to monitoring system"""
    # Send to your monitoring platform (Prometheus, CloudWatch, etc.)
    pass
```

**Track daily:**
- Classification volume
- Accuracy (if ground truth available)
- Confidence score distribution
- Processing latency (P50, P95, P99)
- Error rate

**Track weekly:**
- Category distribution shifts
- Low-confidence prediction rate
- Administrator override rate
- System resource usage (CPU, memory, disk)

**Track monthly:**
- Accuracy trend
- Retraining frequency
- Time/cost savings vs. manual classification

### Monitoring Dashboard

**Recommended tools:**
- **Grafana** + **Prometheus** (open-source)
- **CloudWatch** (AWS)
- **Azure Monitor** (Azure)
- **Custom dashboard** (Flask + Plotly)

**Dashboard panels:**
1. Real-time classification rate
2. Accuracy over time
3. Confidence score distribution
4. Category breakdown (pie chart)
5. Error rate trend
6. Processing latency histogram
7. Resource usage (CPU, memory)
8. Alert status

### Logging Best Practices

```python
# logging_config.py
import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    """Configure application logging"""
    logger = logging.getLogger('ocr_classifier')
    logger.setLevel(logging.INFO)
    
    # File handler (rotating, max 10MB, keep 5 backups)
    file_handler = RotatingFileHandler(
        '/var/log/ocr_classifier/app.log',
        maxBytes=10*1024*1024,
        backupCount=5
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(levelname)s: %(message)s'
    ))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
```

**Log events to track:**
- Model loading/reloading
- Classification requests (text hash, category, confidence)
- Errors and exceptions
- Performance degradation alerts
- Administrator overrides
- Retraining events

---

## Troubleshooting

### Common Issues

#### Issue 1: Model Loading Fails

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'ocr_model.pkl'
```

**Solutions:**
1. Verify model files exist:
   ```bash
   ls -lh ocr_*.pkl
   ```
2. Check file permissions:
   ```bash
   chmod 644 ocr_*.pkl
   ```
3. Ensure correct working directory:
   ```python
   import os
   print(os.getcwd())
   ```

---

#### Issue 2: Slow Inference

**Symptoms:** Latency >100ms per document

**Diagnostics:**
```python
import time

def profile_inference(text):
    start = time.time()
    
    # Feature extraction
    t1 = time.time()
    features = extract_features(text)
    feature_time = time.time() - t1
    
    # Prediction
    t2 = time.time()
    prediction = classifier.predict(text)
    predict_time = time.time() - t2
    
    print(f"Feature extraction: {feature_time*1000:.1f}ms")
    print(f"Prediction: {predict_time*1000:.1f}ms")
    print(f"Total: {(time.time()-start)*1000:.1f}ms")
```

**Solutions:**
1. **Batch processing:** Process multiple documents together
2. **Parallel processing:** Use multiprocessing for CPU-bound tasks
3. **Model optimization:** Consider model compression
4. **Hardware upgrade:** Add more CPU cores or RAM

---

#### Issue 3: Memory Usage Spikes

**Symptoms:** Memory usage >2GB

**Solutions:**
1. **Batch size reduction:** Process smaller batches
2. **Garbage collection:** Force GC after large batches
   ```python
   import gc
   results = classifier.predict_batch(large_batch)
   gc.collect()
   ```
3. **Model reloading:** Reload model periodically to clear memory

---

#### Issue 4: Accuracy Degradation

**Symptoms:** Accuracy drops below 85%

**Diagnostic steps:**
1. Check category distribution shift:
   ```python
   recent_predictions = get_recent_predictions(days=30)
   baseline = load_baseline_distribution()
   
   print("Category distribution comparison:")
   for category in baseline:
       recent_pct = recent_predictions[category] / sum(recent_predictions.values())
       baseline_pct = baseline[category] / sum(baseline.values())
       drift = abs(recent_pct - baseline_pct)
       print(f"{category}: {drift:.2%} drift")
   ```

2. Analyze recent errors:
   ```python
   errors = get_misclassified_samples(days=30)
   error_patterns = analyze_error_types(errors)
   print(error_patterns)
   ```

3. Check for data quality issues:
   - OCR quality degradation
   - New notice formats
   - Terminology changes

**Solutions:**
1. **Immediate:** Increase manual review threshold
2. **Short-term:** Analyze errors, adjust confidence thresholds
3. **Long-term:** Retrain with new data

---

## Scaling Considerations

### Horizontal Scaling

**For >10,000 documents/month:**

```yaml
# docker-compose.yml
version: '3.8'

services:
  classifier-1:
    image: ocr-classifier:latest
    deploy:
      replicas: 3
    ports:
      - "5001-5003:5000"
    environment:
      - WORKER_ID=1
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - classifier-1
```

**Load balancer (nginx.conf):**
```nginx
upstream classifiers {
    least_conn;
    server classifier-1:5000;
    server classifier-2:5000;
    server classifier-3:5000;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://classifiers;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Parallel Processing

```python
# parallel_batch.py
from multiprocessing import Pool, cpu_count
from src.ocr_classifier import OCRClassifier

def process_chunk(texts_chunk):
    """Process a chunk of texts in parallel"""
    classifier = OCRClassifier()
    classifier.load_model()
    return classifier.predict_batch(texts_chunk)

def parallel_classify(all_texts, n_workers=None):
    """Classify large batches using multiprocessing"""
    if n_workers is None:
        n_workers = cpu_count()
    
    # Split into chunks
    chunk_size = len(all_texts) // n_workers
    chunks = [all_texts[i:i+chunk_size] 
              for i in range(0, len(all_texts), chunk_size)]
    
    # Process in parallel
    with Pool(n_workers) as pool:
        results_chunks = pool.map(process_chunk, chunks)
    
    # Flatten results
    return [r for chunk in results_chunks for r in chunk]

# Usage
texts = load_10000_notices()
results = parallel_classify(texts, n_workers=4)
print(f"Processed {len(results)} notices using 4 workers")
```

**Expected speedup:** 4-8x on multi-core systems

---

## Support and Contact

**Issues and Questions:**
- GitHub Issues: https://github.com/chakrabt/OCR-Notice-Classifier/issues
- Email: tamalc@gmail.com

**Documentation:**
- Feature specifications: `FEATURES.md`
- Repository: https://github.com/chakrabt/OCR-Notice-Classifier
- Paper: "Ensemble-Based Four-Class OCR Document Classification for University Notices"

---

**Last Updated:** January 2025  
**Version:** 1.0  
**License:** MIT
