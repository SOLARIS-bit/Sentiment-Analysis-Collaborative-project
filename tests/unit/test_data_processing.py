# tests/unit/test_data_processing.py
import os
import sys

# When this test file is executed directly (python tests/unit/test_data_processing.py),
# ensure the project root is on sys.path so `from src...` imports work.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest
import torch
from transformers import AutoTokenizer
import pandas as pd
from src.data_processing import clean_text, preprocess_texts, tokenize_data

def test_clean_text():
    text = "Great movie! <br /> I love it!!!"
    cleaned = clean_text(text)
    assert cleaned == "great movie i love it"
    assert len(cleaned) > 0

def test_preprocess_texts():
    df = pd.DataFrame({'review': ['Test <br /> HTML', ''], 'sentiment': [1, 0]})
    df_clean = preprocess_texts(df)
    assert len(df_clean) == 1  # Filtre le vide
    assert 'test html' in df_clean['review'].iloc[0]

def test_tokenize_data():
    df = pd.DataFrame({'review': ['Good movie'], 'sentiment': [1]})
    datasets = tokenize_data(df, test_size=0.0)  # Pas de split pour test
    assert datasets['train']['input_ids'].shape[0] == 1
    assert datasets['train']['labels'].shape == torch.Size([1])
    # Vérifier token IDs (CLS et SEP)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    expected = tokenizer('good movie')['input_ids']
    assert torch.equal(datasets['train']['input_ids'][0, :len(expected)], torch.tensor(expected))