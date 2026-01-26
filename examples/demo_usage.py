Demo Usage Examples for OCR Notice Classifier
==============================================

This script demonstrates various usage patterns for the OCR-based
university notice classification system.

Paper: "Ensemble-Based Classification of OCR-Extracted University Notices: A Weighted Voting Approach with Domain-Aware Feature Engineering"
Author: Tamal Chakraborty
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ocr_classifier import OCRClassifier


def example_1_basic_classification():
    """Example 1: Basic single document classification"""

    print("=" * 80)
    print("EXAMPLE 1: Basic Classification")
    print("=" * 80)

    # Initialize classifier
    classifier = OCRClassifier()

    # Load pre-trained model
    print("\n[1] Loading pre-trained model...")
    classifier.load_model()
    print("    ✓ Model loaded successfully")

    # Sample notice text
    notice_text = """
    NOTIFICATION

    This is to notify all students that the semester final examination
    will be conducted from 15th December 2024. Hall tickets can be
    downloaded from the student portal starting 10th December 2024.

    Students must bring their admit cards and valid ID cards to the
    examination hall. Answer sheets will be provided at the venue.

    For any queries, contact the Examination Cell.
    """

    # Classify
    print("\n[2] Classifying notice...")
    result = classifier.predict(notice_text)

    # Display results
    print("\n[3] Results:")
    print(f"    Category: {result['category']}")
    print(f"    Confidence: {result['confidence']}")
    print(f"    All probabilities:")
    for cat, prob in result['probabilities'].items():
        print(f"      - {cat:12s}: {prob}")

    print("\n" + "=" * 80 + "\n")
    return result


def example_2_batch_classification():
    """Example 2: Batch processing multiple documents"""

    print("=" * 80)
    print("EXAMPLE 2: Batch Classification")
    print("=" * 80)

    classifier = OCRClassifier()
    classifier.load_model()

    # Multiple notices
    notices = [
        {
            "id": "NOTICE_001",
            "text": """Merit list for PhD admission 2024-25 has been published.
                      Selected candidates must attend the counseling session on
                      20th November 2024. Please bring all original documents."""
        },
        {
            "id": "NOTICE_002",
            "text": """Circular No. 45/2024. It is hereby notified to all faculty
                      members that the new attendance policy will be implemented
                      from 1st December 2024."""
        },
        {
            "id": "NOTICE_003",
            "text": """Workshop on Artificial Intelligence and Machine Learning
                      will be organized on 25th November 2024. Registration is
                      open for all students. Certificates will be provided."""
        }
    ]

    print("\n[1] Processing batch of notices...\n")

    # Classify all notices
    texts = [n["text"] for n in notices]
    results = classifier.predict_batch(texts)

    # Display results
    print("[2] Batch Results:\n")
    print(f"{'Notice ID':<15} {'Category':<15} {'Confidence':<12}")
    print("-" * 50)

    for notice, result in zip(notices, results):
        print(f"{notice['id']:<15} {result['category']:<15} {result['confidence']:<12}")

    print("\n" + "=" * 80 + "\n")
    return results


def example_3_confidence_filtering():
    """Example 3: Filtering low-confidence predictions"""

    print("=" * 80)
    print("EXAMPLE 3: Confidence-Based Filtering")
    print("=" * 80)

    classifier = OCRClassifier()
    classifier.load_model()

    # Ambiguous notice (mentions multiple categories)
    ambiguous_notice = """
    Circular No. 123/2024

    This is to inform all students that a workshop on examination
    preparation strategies will be conducted on 5th December 2024.
    Admission is free for all registered students.
    """

    print("\n[1] Classifying ambiguous notice...")
    result = classifier.predict(ambiguous_notice)

    # Check confidence threshold
    confidence_threshold = 0.80  # 80%

    print(f"\n[2] Result with confidence filtering (threshold: {confidence_threshold:.0%}):")
    print(f"    Predicted Category: {result['category']}")
    print(f"    Confidence: {result['confidence']}")

    if result['confidence_score'] < confidence_threshold:
        print(f"\n    ⚠️  WARNING: Low confidence prediction!")
        print(f"    ⚠️  Consider manual review for this notice.")
        print(f"\n    Alternative categories:")

        # Sort probabilities
        sorted_probs = sorted(
            result['probabilities_raw'].items(),
            key=lambda x: x[1],
            reverse=True
        )

        for cat, prob in sorted_probs[:3]:
            print(f"      - {cat:12s}: {prob:.1%}")
    else:
        print(f"\n    ✓ High confidence prediction - can be auto-routed")

    print("\n" + "=" * 80 + "\n")
    return result


def example_4_from_file():
    """Example 4: Processing notices from a file"""

    print("=" * 80)
    print("EXAMPLE 4: Processing from File")
    print("=" * 80)

    classifier = OCRClassifier()
    classifier.load_model()

    # Read sample notices file
    sample_file = os.path.join(os.path.dirname(__file__), 'sample_notices.txt')

    if not os.path.exists(sample_file):
        print(f"\n⚠️  Sample file not found: {sample_file}")
        print("   Please ensure sample_notices.txt exists in the examples/ directory")
        return

    print(f"\n[1] Reading notices from: {sample_file}")

    with open(sample_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by separator pattern (===== NOTICE N: CATEGORY =====)
    import re
    notice_pattern = r'={5,}\s*NOTICE\s+\d+:.*?={5,}'
    parts = re.split(notice_pattern, content)

    # Filter out header (first part), empty parts, and very short fragments
    # Skip first element which is always the header before first notice
    notices = [
        n.strip() for n in parts[1:]
        if n.strip() and len(n.strip()) > 100  # Minimum 100 chars
    ]

    print(f"    Found {len(notices)} notices\n")

    print("[2] Classification Results:\n")

    # Classify each notice
    for i, notice in enumerate(notices, 1):
        result = classifier.predict(notice)

        # Extract first line as title
        title = notice.split('\n')[0][:50]

        print(f"Notice {i}: {title}...")
        print(f"  → Category: {result['category']}")
        print(f"  → Confidence: {result['confidence']}")
        print()

    print("=" * 80 + "\n")


def example_5_custom_integration():
    """Example 5: Custom integration with routing logic"""

    print("=" * 80)
    print("EXAMPLE 5: Custom Integration with Auto-Routing")
    print("=" * 80)

    classifier = OCRClassifier()
    classifier.load_model()

    # Routing configuration
    ROUTING_CONFIG = {
        'examination': {
            'email': 'exam-cell@university.edu',
            'department': 'Examination Department',
            'priority': 'HIGH'
        },
        'admission': {
            'email': 'admissions@university.edu',
            'department': 'Admissions Office',
            'priority': 'HIGH'
        },
        'circular': {
            'email': 'admin@university.edu',
            'department': 'Administration',
            'priority': 'MEDIUM'
        },
        'event': {
            'email': 'events@university.edu',
            'department': 'Student Affairs',
            'priority': 'LOW'
        }
    }

    # Sample notice
    notice = """
    Final Year Examination Schedule - Spring 2024

    The final examinations for all undergraduate programs will be
    held from 10th to 25th December 2024. Detailed timetables
    will be published on the university website.
    """

    print("\n[1] Classifying and routing notice...")
    result = classifier.predict(notice)

    category = result['category']
    routing_info = ROUTING_CONFIG.get(category, {})

    print(f"\n[2] Classification Result:")
    print(f"    Category: {category}")
    print(f"    Confidence: {result['confidence']}")

    print(f"\n[3] Auto-Routing Information:")
    print(f"    Department: {routing_info.get('department', 'Unknown')}")
    print(f"    Email: {routing_info.get('email', 'Unknown')}")
    print(f"    Priority: {routing_info.get('priority', 'NORMAL')}")

    # Simulated routing action
    if result['confidence_score'] >= 0.85:
        print(f"\n[4] Action: ✓ Auto-routed to {routing_info.get('department')}")
    else:
        print(f"\n[4] Action: ⚠️  Flagged for manual review (low confidence)")

    print("\n" + "=" * 80 + "\n")


def main():
    """Run all examples"""

    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  OCR Notice Classifier - Demo Usage Examples".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    try:
        # Run examples
        example_1_basic_classification()
        example_2_batch_classification()
        example_3_confidence_filtering()
        example_4_from_file()
        example_5_custom_integration()

        print("\n✓ All examples completed successfully!")
        print("\nFor more information, see:")
        print("  - README.md: General documentation")
        print("  - Paper: Full methodology and evaluation\n")

    except FileNotFoundError as e:
        print(f"\n Error: Model files not found!")
        print(f"   {str(e)}")
        print("\n   Please ensure you have:")
        print("   1. Trained the model using train.py, OR")
        print("   2. Downloaded pre-trained model files")
        print("\n   Model files required:")
        print("   - ocr_model.pkl")
        print("   - ocr_vectorizer.pkl")
        print("   - ocr_encoder.pkl")
        print("   - ocr_metadata.pkl\n")

    except Exception as e:
        print(f"\n Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
