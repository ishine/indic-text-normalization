import json

source_path = 'test_tamil.ipynb'
dest_path = 'test_malyalam.ipynb'

# Tamil to Malayalam digit map
digit_map = {
    '௦': '൦',
    '௧': '൧',
    '௨': '൨',
    '௩': '൩',
    '௪': '൪',
    '௫': '௫',
    '௬': '൬',
    '௭': '൭',
    '௮': '൮',
    '௯': '൯'
}

# Simple word replacements (optional but good for context)
# Note: This is a best-effort replacement for specific patterns found in test_tamil.ipynb
word_mapp = {
    'வது': 'ാം', # Ordinal suffix
    'ரூ': 'രൂപ',   # Rupee symbol/text
}

def replace_text(text):
    # Replace digits
    for t, m in digit_map.items():
        text = text.replace(t, m)
    # Replace words
    for t, m in word_mapp.items():
        text = text.replace(t, m)
    
    # Replace Tamil specific headers/text
    text = text.replace('Tamil', 'Malayalam')
    text = text.replace('TAMIL', 'MALAYALAM')
    text = text.replace("lang='ta'", "lang='ma'")
    text = text.replace('normalizer_ta', 'normalizer_ma')
    
    return text

try:
    with open(source_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    # Process cells
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                new_source.append(replace_text(line))
            cell['source'] = new_source
            
            # Clear outputs as they will be different
            cell['outputs'] = []
            cell['execution_count'] = None
            
        elif cell['cell_type'] == 'markdown':
            new_source = []
            for line in cell['source']:
                new_source.append(replace_text(line))
            cell['source'] = new_source

    with open(dest_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print("Successfully created test_malyalam.ipynb with mixed cases")

except Exception as e:
    print(f"Error: {e}")
