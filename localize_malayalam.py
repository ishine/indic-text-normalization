#!/usr/bin/env python3
"""
Comprehensive script to localize Malayalam text normalization from Hindi to Malayalam.
This script replaces Hindi Devanagari text with Malayalam equivalents across all files.
"""

import os
import re
from pathlib import Path

# Common Hindi to Malayalam replacements
COMMON_REPLACEMENTS = {
    # Negative/minus
    'ऋणात्मक': 'നെഗറ്റീവ്',
    
    # Decimal
    'दशमलव': 'ദശാംശം',
    'बिंदु': 'ബിന്ദു',
    
    # Currency
    'रुपये': 'രൂപ',
    'रुपया': 'രൂപ',
    'पैसे': 'പൈസ',
    'पैसा': 'പൈസ',
    
    # Time words
    'बजे': 'മണി',
    'घंटे': 'മണിക്കൂർ',
    'मिनट': 'മിനിറ്റ്',
    'सेकंड': 'സെക്കൻഡ്',
    
    # Ordinal
    'वां': 'ാം',
    'वीं': 'ാം',
    'वें': 'ാം',
    
    # Fractions
    'आधा': 'പകുതി',
    'चौथाई': 'കാൽ',
    'तिहाई': 'മൂന്നിലൊന്ന്',
    
    # Common words
    'और': 'ഉം',
    'से': 'മുതൽ',
    'तक': 'വരെ',
    'का': 'ന്റെ',
    'की': 'ന്റെ',
    'के': 'ന്റെ',
}

# Hindi digits to Malayalam digits
DIGIT_REPLACEMENTS = {
    '०': '൦',
    '१': '൧',
    '२': '൨',
    '३': '൩',
    '४': '൪',
    '५': '൫',
    '६': '൬',
    '७': '൭',
    '८': '൮',
    '९': '൯',
}

def find_hindi_text(text):
    """Find all Hindi Devanagari text in a string."""
    # Devanagari Unicode range: U+0900 to U+097F
    hindi_pattern = re.compile(r'[\u0900-\u097F]+')
    matches = hindi_pattern.findall(text)
    return matches

def replace_text(content, replacements):
    """Replace text using the provided dictionary."""
    for hindi, malayalam in replacements.items():
        content = content.replace(hindi, malayalam)
    return content

def process_file(filepath, replacements, dry_run=False):
    """Process a single file and apply replacements."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Apply replacements
        new_content = replace_text(original_content, replacements)
        
        if new_content != original_content:
            if not dry_run:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                return True, "Updated"
            else:
                # Find what Hindi text remains
                remaining_hindi = find_hindi_text(new_content)
                if remaining_hindi:
                    return True, f"Would update (remaining Hindi: {set(remaining_hindi)})"
                else:
                    return True, "Would update (all Hindi replaced)"
        else:
            # Check if there's any Hindi text
            hindi_found = find_hindi_text(original_content)
            if hindi_found:
                return False, f"No changes (unmapped Hindi: {set(hindi_found)})"
            else:
                return False, "No Hindi text found"
    except Exception as e:
        return False, f"Error: {e}"

def process_directory(base_path, replacements, dry_run=False):
    """Process all Python and TSV files in a directory."""
    results = {
        'updated': [],
        'unchanged': [],
        'errors': [],
        'unmapped_hindi': set()
    }
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(('.py', '.tsv')):
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, base_path)
                
                changed, message = process_file(filepath, replacements, dry_run)
                
                if changed:
                    results['updated'].append((rel_path, message))
                    # Extract unmapped Hindi if present
                    if 'unmapped Hindi' in message or 'remaining Hindi' in message:
                        hindi_match = re.search(r'\{([^}]+)\}', message)
                        if hindi_match:
                            hindi_words = hindi_match.group(1).replace("'", "").split(', ')
                            results['unmapped_hindi'].update(hindi_words)
                else:
                    if 'unmapped Hindi' in message:
                        results['unchanged'].append((rel_path, message))
                        # Extract unmapped Hindi
                        hindi_match = re.search(r'\{([^}]+)\}', message)
                        if hindi_match:
                            hindi_words = hindi_match.group(1).replace("'", "").split(', ')
                            results['unmapped_hindi'].update(hindi_words)
                    elif 'Error' in message:
                        results['errors'].append((rel_path, message))
    
    return results

def main():
    # Base path for Malayalam normalization
    ma_path = r'nemo_text_processing\text_normalization\ma'
    
    if not os.path.exists(ma_path):
        print(f"Error: Path {ma_path} does not exist!")
        return
    
    # Combine all replacements
    all_replacements = {**COMMON_REPLACEMENTS, **DIGIT_REPLACEMENTS}
    
    print("=" * 80)
    print("Malayalam Localization Script")
    print("=" * 80)
    print(f"\nProcessing directory: {ma_path}")
    print(f"Total replacements defined: {len(all_replacements)}")
    print("\n" + "=" * 80)
    
    # First, do a dry run to see what would change
    print("\n🔍 DRY RUN - Analyzing files...")
    print("-" * 80)
    dry_results = process_directory(ma_path, all_replacements, dry_run=True)
    
    print(f"\n📊 Analysis Results:")
    print(f"  Files to update: {len(dry_results['updated'])}")
    print(f"  Files unchanged: {len(dry_results['unchanged'])}")
    print(f"  Errors: {len(dry_results['errors'])}")
    
    if dry_results['updated']:
        print(f"\n📝 Files that will be updated:")
        for filepath, message in dry_results['updated'][:10]:  # Show first 10
            print(f"  ✓ {filepath}")
            if 'remaining Hindi' in message or 'unmapped Hindi' in message:
                print(f"    {message}")
    
    if dry_results['unmapped_hindi']:
        print(f"\n⚠️  Unmapped Hindi words found ({len(dry_results['unmapped_hindi'])}):")
        for word in sorted(dry_results['unmapped_hindi'])[:20]:  # Show first 20
            print(f"  • {word}")
    
    # Now do the actual replacement
    print("\n" + "=" * 80)
    print("✏️  APPLYING CHANGES...")
    print("-" * 80)
    
    results = process_directory(ma_path, all_replacements, dry_run=False)
    
    print(f"\n✅ Completed!")
    print(f"  Files updated: {len(results['updated'])}")
    print(f"  Files unchanged: {len(results['unchanged'])}")
    print(f"  Errors: {len(results['errors'])}")
    
    if results['updated']:
        print(f"\n📝 Updated files:")
        for filepath, message in results['updated']:
            print(f"  ✓ {filepath}")
    
    if results['errors']:
        print(f"\n❌ Errors:")
        for filepath, message in results['errors']:
            print(f"  ✗ {filepath}: {message}")
    
    if results['unmapped_hindi']:
        print(f"\n⚠️  Remaining unmapped Hindi words ({len(results['unmapped_hindi'])}):")
        print("These words need manual translation and should be added to COMMON_REPLACEMENTS:")
        for word in sorted(results['unmapped_hindi']):
            print(f"  '{word}': 'MALAYALAM_TRANSLATION',")
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)

if __name__ == '__main__':
    main()
