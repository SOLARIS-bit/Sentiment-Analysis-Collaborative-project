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
    """Test que le nettoyage du texte fonctionne comme dans l'approche Kaggle:
    - Suppression des balises HTML
    - Conversion en minuscules
    - Suppression de la ponctuation
    - Normalisation des espaces
    """
    # Test avec HTML, majuscules, ponctuation et espaces multiples
    text = "Great   Movie!!!<br />I   LOVE  it...  "
    cleaned = clean_text(text)
    assert cleaned == "great movie i love it"
    
    # Test avec caractères spéciaux et HTML complexe
    text = "<div>Special &amp; chars @ #$% </div><br/>"
    cleaned = clean_text(text)
    assert cleaned == "special chars"
    
    # Test avec texte vide ou None
    assert clean_text("") == ""
    assert clean_text(None) == ""

def test_preprocess_texts():
    df = pd.DataFrame({'content': ['Test <br /> HTML content longer than 10 chars', ''], 'score': [4, 2]})
    df_clean = preprocess_texts(df)
    assert len(df_clean) == 1  # Filtre le vide
    assert 'test html content longer than chars' in df_clean['content'].iloc[0]

def test_tokenize_data():
    df = pd.DataFrame({
        'content': [
            'Good movie longer than ten chars',
            'Another good movie review',
            'Bad movie review not good',
            'Terrible movie very bad'
        ], 
        'sentiment': [1, 1, 0, 0]  # 2 positifs, 2 négatifs
    })
    datasets = tokenize_data(df, text_col='content', test_size=0.5)  # Split 50/50
    # Vérifier les dimensions (2 exemples dans train, 2 dans val)
    assert datasets['train']['input_ids'].shape[0] == 2
    assert datasets['train']['labels'].shape == torch.Size([2])
    assert datasets['val']['input_ids'].shape[0] == 2
    assert datasets['val']['labels'].shape == torch.Size([2])
    # Vérifier format des tokens (commence par CLS=[101] et finit par SEP=[102])
    assert datasets['train']['input_ids'][0, 0].item() == 101  # CLS token
    assert 102 in datasets['train']['input_ids'][0]  # SEP token quelque part dans la séquence


if __name__ == "__main__":
    # When this file is executed directly, run pytest on this file so the test
    # functions actually execute and produce output.
    raise SystemExit(pytest.main([__file__]))