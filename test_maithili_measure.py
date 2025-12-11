"""
Test script for Maithili measure tagger enhancements
Tests Arabic digit support, range patterns, per-unit patterns, and other new features
"""

from nemo_text_processing.text_normalization import Normalizer

# Initialize Maithili normalizer
print("Initializing Maithili normalizer...")
normalizer = Normalizer(input_case='cased', lang='mai')

# Test cases
test_cases = [
    # Arabic digits
    ("12kg", "Arabic digit with unit"),
    ("25m", "Arabic digit with meter"),
    ("100km", "Arabic digit with kilometer"),
    
    # Range patterns
    ("2-3kg", "Range with dash"),
    ("5x10cm", "Range with x"),
    ("2*3m", "Range with asterisk"),
    
    # Per-unit patterns
    ("100km/h", "Kilometer per hour"),
    ("50m/s", "Meter per second"),
    
    # Decimal with Arabic digits
    ("12.5kg", "Decimal with Arabic digits"),
    ("2.5m", "Decimal measurement"),
    
    # Maithili digits (should still work)
    ("१२kg", "Maithili digit with unit"),
    ("२५m", "Maithili digit with meter"),
]

print("\n" + "="*80)
print("Testing Maithili Measure Tagger Enhancements")
print("="*80 + "\n")

for text, description in test_cases:
    try:
        result = normalizer.normalize(text)
        print(f"✓ {description:30s} | Input: {text:15s} | Output: {result}")
    except Exception as e:
        print(f"✗ {description:30s} | Input: {text:15s} | Error: {str(e)[:50]}")

print("\n" + "="*80)
print("Test completed!")
print("="*80)
