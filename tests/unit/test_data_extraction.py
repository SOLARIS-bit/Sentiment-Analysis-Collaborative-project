# tests/unit/test_data_extraction.py
import pytest
import pandas as pd
import tempfile
import os
from test_data_extraction import load_sentiment_data  # Importe ta fonction

@pytest.fixture
def sample_csv(tmp_path):
    """
    Crée un CSV de test temporaire avec données valides.
    """
    df_test = pd.DataFrame({
        'review': ['Great movie!', 'Terrible plot.'],
        'sentiment': ['positive', 'negative']
    })
    file_path = tmp_path / "test.csv"
    df_test.to_csv(file_path, index=False)
    yield str(file_path)
    # Cleanup automatique via tmp_path, mais on peut forcer
    os.remove(file_path)

def test_load_data_success(sample_csv):
    """
    Teste le chargement réussi : vérifie longueur, colonnes, et conversion labels.
    """
    df = load_sentiment_data(sample_csv)
    assert len(df) == 2
    assert list(df.columns) == ['review', 'sentiment']
    assert df['sentiment'].tolist() == [1, 0]  # Convertis : positive=1, negative=0

def test_load_missing_file():
    """
    Teste gestion d'erreur : fichier inexistant → DataFrame vide.
    """
    df = load_sentiment_data("nonexistent.csv")
    assert df.empty

def test_load_wrong_columns(tmp_path):
    """
    Teste colonnes manquantes/inattendues → DataFrame vide.
    """
    wrong_df = pd.DataFrame({'wrong_col': ['test']})
    file_path = tmp_path / "wrong.csv"
    wrong_df.to_csv(file_path, index=False)
    df = load_sentiment_data(str(file_path))
    assert df.empty
    os.remove(file_path)  # Cleanup

def test_load_empty_csv(tmp_path):
    """
    Teste CSV vide → DataFrame vide.
    """
    empty_df = pd.DataFrame()
    file_path = tmp_path / "empty.csv"
    empty_df.to_csv(file_path, index=False)
    df = load_sentiment_data(str(file_path))
    assert df.empty
    os.remove(file_path)  # Cleanup

def test_load_with_null_sentiment(tmp_path):
    """
    Teste sentiment inattendu (NaN) → Avertissement, mais lignes supprimées.
    """
    df_test = pd.DataFrame({
        'review': ['Test review'],
        'sentiment': ['unknown']  # Mappera à NaN
    })
    file_path = tmp_path / "null.csv"
    df_test.to_csv(file_path, index=False)
    df = load_sentiment_data(str(file_path))
    assert len(df) == 0  # Supprimé car NaN
    os.remove(file_path)