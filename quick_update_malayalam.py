import os

# Simple script to replace Hindi with Malayalam in specific files
files_and_replacements = {
    r'nemo_text_processing\text_normalization\ma\verbalizers\decimal.py': {
        'दशमलव': 'ദശാംശം',
    },
    r'nemo_text_processing\text_normalization\ma\verbalizers\cardinal.py': {
        # Already uses MINUS from graph_utils, so should be fine
    },
}

print("Updating Malayalam verbalizer files...")

for filepath, replacements in files_and_replacements.items():
    if not replacements:
        print(f"  Skipping {filepath} (no replacements needed)")
        continue
    
    if not os.path.exists(filepath):
        print(f"  ✗ File not found: {filepath}")
        continue
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        for hindi, malayalam in replacements.items():
            content = content.replace(hindi, malayalam)
        
        if content != original:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✓ Updated: {filepath}")
        else:
            print(f"  - No changes: {filepath}")
    except Exception as e:
        print(f"  ✗ Error: {filepath} - {e}")

print("\nDone!")
