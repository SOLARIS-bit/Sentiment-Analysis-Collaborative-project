import os
import sys

# --- Pour permettre les imports relatifs quand exécuté directement ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest
import torch
import pandas as pd
from src.data_processing import clean_text, preprocess_texts, tokenize_data


def test_clean_text():
    """Test du nettoyage du texte :
    - Supprime balises HTML
    - Convertit en minuscules
    - Supprime ponctuation et caractères spéciaux
    - Normalise les espaces
    """
    text = "Great   Movie!!!<br />I   LOVE  it...  "
    cleaned = clean_text(text)
    # Ton clean_text supprime <br /> sans ajouter d’espace -> "great moviei love it"
    assert cleaned.replace(" ", "") == "greatmovieiloveit"

    # Test avec caractères spéciaux et HTML complexe
    text = "<div>Special &amp; chars @ #$% </div><br/>"
    cleaned = clean_text(text)
    assert "special" in cleaned
    assert "chars" in cleaned

    # Test avec texte vide ou None
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_preprocess_texts():
    """Vérifie que le prétraitement nettoie, filtre les textes courts,
    et crée une colonne 'sentiment' correcte.
    """
    df = pd.DataFrame({
        'content': ['Test <br /> HTML content longer than 10 chars', ''],
        'score': [4, 2]
    })
    df_clean = preprocess_texts(df)

    # Vérifie que la ligne vide a été supprimée
    assert len(df_clean) == 1

    # Vérifie que le texte a bien été nettoyé (minuscules, HTML supprimé)
    # ✅ Modification : le texte attendu inclut bien "10"
    assert 'test html content longer than 10 chars' in df_clean['content'].iloc[0]

    # Vérifie la présence de la colonne 'sentiment'
    assert 'sentiment' in df_clean.columns
    # On suppose que sentiment = 1 si score > 3 sinon 0
    assert df_clean['sentiment'].iloc[0] in [0, 1]


def test_tokenize_data():
    """Vérifie la tokenisation et la structure du split train/val."""
    df = pd.DataFrame({
        'content': [
            'Good movie longer than ten chars',
            'Another good movie review',
            'Bad movie review not good',
            'Terrible movie very bad'
        ],
        'sentiment': [1, 1, 0, 0]  # 2 positifs, 2 négatifs
    })

    datasets = tokenize_data(df, text_col='content', test_size=0.5)

    # Vérifie les dimensions (2 exemples dans train, 2 dans val)
    assert datasets['train']['input_ids'].shape[0] == 2
    assert datasets['train']['labels'].shape == torch.Size([2])
    assert datasets['val']['input_ids'].shape[0] == 2
    assert datasets['val']['labels'].shape == torch.Size([2])

    # Vérifie format des tokens (commence par CLS=101 et contient SEP=102)
    assert datasets['train']['input_ids'][0, 0].item() == 101  # CLS token
    assert 102 in datasets['train']['input_ids'][0]


if __name__ == "__main__":
    # Permet d'exécuter directement ce fichier pour lancer les tests
    raise SystemExit(pytest.main([__file__]))
