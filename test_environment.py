#!/usr/bin/env python3
"""
Test script to verify the ModernFinBERT environment is working correctly.
This tests the core functionality needed by Data.ipynb.
"""

import sys
print(f"Python version: {sys.version}")

# Test core imports from the notebook
try:
    import numpy as np
    print("✅ numpy imported successfully")
    
    from datasets import load_dataset
    print("✅ datasets imported successfully")
    
    import pandas as pd
    print("✅ pandas imported successfully")
    
    import matplotlib.pyplot as plt
    print("✅ matplotlib imported successfully")
    
    import torch
    print("✅ torch imported successfully")
    
    import transformers
    print("✅ transformers imported successfully")
    
    from huggingface_hub import HfApi
    print("✅ huggingface_hub imported successfully")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test basic functionality
print("\n=== Testing basic functionality ===")

# Test numpy
NUM_CLASSES = 3
label_dict = {'NEUTRAL/MIXED': 1, 'NEGATIVE': 0, 'POSITIVE': 2}
test_labels = np.eye(NUM_CLASSES)[label_dict['POSITIVE']]
print(f"✅ Numpy one-hot encoding works: {test_labels}")

# Test dataset loading (just check connection, don't download full dataset)
try:
    print("✅ Testing HuggingFace connection...")
    # This will just check if we can connect to HF hub
    from huggingface_hub import list_models
    models = list(list_models(limit=1))
    print("✅ HuggingFace Hub connection successful")
except Exception as e:
    print(f"⚠️  HuggingFace connection warning: {e}")

# Test torch
print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ PyTorch MPS available: {torch.backends.mps.is_available()}")
print(f"✅ PyTorch MPS built: {torch.backends.mps.is_built()}")

print("\n=== Environment Test Complete ===")
print("🎉 Your environment is ready for Data.ipynb!")
print("\nTo activate this environment in the future:")
print("  source venv/bin/activate")
print("\nTo run Jupyter:")
print("  source venv/bin/activate && jupyter notebook")