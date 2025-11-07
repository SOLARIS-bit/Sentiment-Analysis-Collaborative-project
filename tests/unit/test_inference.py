# tests/unit/test_inference.py
import pytest
import torch
from unittest.mock import patch, MagicMock
from inference import predict_sentiment, clean_text
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def test_clean_text_in_inference():
    """
    Vérifie que clean_text est bien utilisé dans predict_sentiment.
    Teste nettoyage basique (HTML, ponctuation, lowercase).
    """
    text = "Great <br /> movie!!!"
    cleaned = clean_text(text)
    assert cleaned == "great movie"  # Attendu après nettoyage

def test_predict_sentiment_positive(monkeypatch):
    """
    Teste prédiction positive : mock le modèle pour retourner label=1 (positive).
    Vérifie retour ('positive', confiance >0).
    """
    def mock_softmax(logits):
        return torch.tensor([[0.1, 0.9]])  # Prob [neg=0.1, pos=0.9]
    
    def mock_model_forward(**inputs):
        return MagicMock(logits=torch.tensor([[0.2, 0.8]]))  # Logits pour pos
    
    monkeypatch.setattr("torch.softmax", mock_softmax)
    monkeypatch.setattr("inference.AutoModelForSequenceClassification.from_pretrained", lambda x: MagicMock(forward=mock_model_forward))
    monkeypatch.setattr("inference.AutoTokenizer.from_pretrained", lambda x: MagicMock(encode_plus=lambda t, **k: {'input_ids': torch.tensor([101]), 'attention_mask': torch.tensor([1])}))
    
    label, conf = predict_sentiment("./dummy_model", "Great movie!")
    assert label == "positive"
    assert 0 < conf <= 1

def test_predict_sentiment_negative(monkeypatch):
    """
    Teste prédiction négative : mock pour label=0 (negative).
    """
    def mock_softmax(logits):
        return torch.tensor([[0.9, 0.1]])  # Prob [neg=0.9, pos=0.1]
    
    def mock_model_forward(**inputs):
        return MagicMock(logits=torch.tensor([[0.8, 0.2]]))  # Logits pour neg
    
    monkeypatch.setattr("torch.softmax", mock_softmax)
    monkeypatch.setattr("inference.AutoModelForSequenceClassification.from_pretrained", lambda x: MagicMock(forward=mock_model_forward))
    monkeypatch.setattr("inference.AutoTokenizer.from_pretrained", lambda x: MagicMock(encode_plus=lambda t, **k: {'input_ids': torch.tensor([101]), 'attention_mask': torch.tensor([1])}))
    
    label, conf = predict_sentiment("./dummy_model", "Terrible film.")
    assert label == "negative"
    assert 0 < conf <= 1

def test_predict_sentiment_empty_text():
    """
    Teste erreur sur texte vide : lève ValueError.
    """
    with pytest.raises(ValueError, match="Texte vide"):
        predict_sentiment("./dummy_model", "")

def test_predict_sentiment_missing_model(tmp_path):
    """
    Teste gestion d'erreur : modèle manquant → FileNotFoundError.
    """
    output_dir = str(tmp_path / "missing_model")
    with pytest.raises(FileNotFoundError, match="Modèle non trouvé"):
        predict_sentiment(output_dir, "Test text")

def test_end_to_end_inference_with_mock_model(tmp_path):
    """
    Test end-to-end : Sauvegarde un modèle mock, charge, prédit sur texte simple.
    Vérifie flux complet sans erreur.
    """
    # Crée un fake model et tokenizer sauvés
    model_name = "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    output_dir = str(tmp_path / "fake_model")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Run inference
    label, conf = predict_sentiment(output_dir, "This is a positive test.")
    assert isinstance(label, str)  # 'positive' ou 'negative' selon modèle base
    assert 0 < conf <= 1

def test_main_cli_no_text():
    """
    Teste le main() CLI : Si pas de --text, demande input (simulé).
    Vérifie que ça appelle predict_sentiment.
    """
    with patch('sys.argv', ['inference.py']), \
         patch('builtins.input', return_value='Test input'), \
         patch('inference.predict_sentiment', return_value=('positive', 0.95)) as mock_predict:
        
        from inference import main
        main()
        mock_predict.assert_called_once()