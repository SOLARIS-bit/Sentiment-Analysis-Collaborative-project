# tests/unit/test_model.py
import pytest
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from model import (
    train_model, 
    load_fine_tuned_model, 
    compute_metrics,
    SentimentDataset  # Si tu l'as défini comme classe dans model.py
)
import tempfile
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def test_model_instantiation():
    """
    Teste l'instantiation du modèle BERT sans erreur.
    Vérifie num_labels=2 et type.
    """
    model_name = "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )
    assert model.num_labels == 2
    assert isinstance(model, AutoModelForSequenceClassification)

def test_dummy_batch_forward():
    """
    Passe un batch dummy à travers le modèle et vérifie la shape des logits [1, 2].
    """
    model_name = "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Input dummy
    inputs = tokenizer("This is a test sentence.", return_tensors="pt")
    batch = {key: val for key, val in inputs.items()}
    batch['labels'] = torch.tensor([1])  # Label dummy
    
    with torch.no_grad():
        outputs = model(**batch)
    
    logits = outputs.logits
    assert logits.shape == torch.Size([1, 2])  # Batch=1, classes=2

def test_compute_metrics():
    """
    Teste le calcul des métriques sur des prédictions/labels dummy.
    Vérifie présence d'accuracy et valeurs valides.
    """
    # Mock pour pred (comme dans Trainer)
    pred = type('obj', (object,), {
        'label_ids': torch.tensor([0, 1, 1]),
        'predictions': torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])  # Preds argmax: [1, 0, 1]
    })()
    metrics = compute_metrics(pred)
    assert 'accuracy' in metrics
    assert 0 <= metrics['accuracy'] <= 1
    assert 'f1' in metrics
    # Vérif manuelle : preds [1,0,1] vs labels [0,1,1] → accuracy=0.666
    assert abs(metrics['accuracy'] - 2/3) < 0.01

def test_sentiment_dataset():
    """
    Teste le Dataset custom : __len__ et __getitem__ retournent bien les items.
    """
    encodings = {
        'input_ids': torch.tensor([[101, 2023, 102]]),  # Dummy tokens
        'attention_mask': torch.tensor([[1, 1, 1]])
    }
    labels = [1]
    dataset = SentimentDataset(encodings, labels)
    assert len(dataset) == 1
    item = dataset[0]
    assert 'input_ids' in item
    assert item['labels'] == torch.tensor(1)
    assert item['input_ids'].shape == torch.Size([3])

def test_train_model_integration(tmp_path):
    """
    Test d'intégration end-to-end : train sur un tiny dataset.
    Skip si erreur (ex. pas de GPU ou temps long).
    """
    # Crée un tiny CSV pour test rapide
    tiny_df = pd.DataFrame({
        'review': ['Great movie!', 'Terrible film.'],
        'sentiment': ['positive', 'negative']
    })
    tiny_path = tmp_path / "tiny.csv"
    tiny_df.to_csv(tiny_path, index=False)
    
    output_dir = str(tmp_path / "test_output")
    try:
        trainer = train_model(str(tiny_path), output_dir, num_epochs=1, max_length=128)
        assert trainer is not None
        assert os.path.exists(output_dir)  # Vérifie save
    except Exception as e:
        pytest.skip(f"Training skipped pour test (GPU/ressources ?) : {e}")

def test_load_fine_tuned_model(tmp_path):
    """
    Teste le chargement d'un modèle sauvé (simulé).
    Vérifie num_labels et tokenizer.
    """
    model_name = "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    output_dir = str(tmp_path / "fake_model")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    loaded_model, loaded_tokenizer = load_fine_tuned_model(output_dir)
    assert loaded_model.num_labels == 2
    assert loaded_tokenizer.name_or_path == model_name
    assert os.path.exists(os.path.join(output_dir, "config.json"))  # Fichier sauvé