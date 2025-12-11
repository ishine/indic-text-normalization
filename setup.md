# Setup Guide for Indic Text Normalization

This guide provides two methods to set up the environment for the Indic Text Normalization project.

> [!NOTE]
> **Conda is the recommended method** for setting up this project. UV is provided as an alternative option.

## Prerequisites

- Python 3.7 or higher
- Conda (recommended) or UV package manager installed

---

## Method 1: Using Conda (Recommended)

### Step 1: Create Virtual Environment

```bash
conda create -n venv python=3.8
```

### Step 2: Activate Environment

```bash
conda activate venv
```

### Step 3: Install Pynini

```bash
conda install -c conda-forge pynini=2.1.6.post1
```

### Step 4: Install Indic Text Normalization

```bash
pip install git+https://github.com/Kenpath/indic-text-normalization.git@main
```

### Step 5: Verify Installation

```bash
python -c "import pynini; print('Pynini version:', pynini.__version__)"
python -c "import nemo_text_processing; print('Installation successful!')"
```

---

## Method 2: Using UV (Alternative)

> [!IMPORTANT]
> After installing dependencies with UV, you should run your scripts normally using `python main.py`, **NOT** `uv run main.py` or any other UV-specific commands.

### Step 1: Initialize UV Project

```bash
uv init
```

### Step 2: Install Pynini

```bash
uv pip install pynini==2.1.6.post1
```

### Step 3: Install Indic Text Normalization

```bash
uv pip install git+https://github.com/Kenpath/indic-text-normalization.git@main
```

### Step 4: Verify Installation

```bash
python -c "import pynini; print('Pynini version:', pynini.__version__)"
python -c "import nemo_text_processing; print('Installation successful!')"
```

### Step 5: Running Your Scripts

After installation, run your scripts normally:

```bash
python main.py
python test_tamil.py
# Or any other script
```

**Do NOT use** `uv run main.py` - just use the standard `python` command.

---

## Troubleshooting

### Common Issues

1. **Pynini installation fails**: Ensure you're using the conda-forge channel (Conda method) or the exact version specified (UV method).

2. **Git installation fails**: Make sure you have git installed and accessible from your command line.

3. **Import errors**: Verify that your virtual environment is activated before running Python scripts.

### Platform-Specific Notes

- **Windows**: You may need to install Microsoft Visual C++ Build Tools for some dependencies.
- **Linux**: Ensure you have `python3-dev` and `build-essential` packages installed.
- **macOS**: Xcode Command Line Tools may be required.

---

## Next Steps

After successful installation, you can:

1. Run the test notebooks (e.g., `test_tamil.ipynb`, `test_hindi.ipynb`)
2. Import the normalizer in your Python scripts:
   ```python
   from nemo_text_processing.text_normalization.normalize import Normalizer
   normalizer = Normalizer(input_case='cased', lang='hi')
   ```

---

## Deactivating Environment

### Conda
```bash
conda deactivate
```

### UV
Simply close your terminal or switch to a different project directory.
