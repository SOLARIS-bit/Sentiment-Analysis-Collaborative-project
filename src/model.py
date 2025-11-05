# model.py
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from data_extraction import load_sentiment_data
from data_processing import preprocess_texts, tokenize_data
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

logger = logging.getLogger(__name__)

class SentimentDataset(torch.utils.data.Dataset):
    """
    Dataset custom pour Hugging Face Trainer.
    Prend input_ids, attention_mask, labels.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    """
    Métriques pour Trainer : accuracy, precision, recall, f1.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_model(data_path: str = "data/IMDB Dataset.csv", output_dir: str = "./model_output", num_epochs: int = 3, max_length: int = 512):
    """
    Fine-tune BERT sur le dataset sentiment.
    Étapes : Load → Clean → Tokenize → Train avec Trainer API.
    
    Args:
        data_path (str): Chemin CSV.
        output_dir (str): Dossier sauvegarde modèle.
        num_epochs (int): Nombre d'époques.
        max_length (int): Longueur max tokens.
    
    Returns:
        Trainer: Instance du trainer fine-tuné.
    """
    # 1. Load data
    df = load_sentiment_data(data_path)
    if df.empty:
        raise ValueError("Impossible de charger les données.")

    # 2. Preprocess (clean)
    df_clean = preprocess_texts(df)

    # 3. Tokenize & split
    datasets = tokenize_data(df_clean, max_length=max_length)
    train_encodings = {k: v for k, v in datasets['train'].items() if k != 'labels'}
    train_labels = datasets['train']['labels'].numpy()
    val_encodings = {k: v for k, v in datasets['val'].items() if k != 'labels'}
    val_labels = datasets['val']['labels'].numpy()

    # 4. Créer datasets HF
    train_dataset = SentimentDataset(train_encodings, train_labels)
    val_dataset = SentimentDataset(val_encodings, val_labels)

    # 5. Load model & tokenizer
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,  # Binaire : neg/pos
        ignore_mismatched_sizes=True  # Si mismatch
    )

    # 6. Data collator (padding dynamique)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 7. Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=8,  # Ajuste selon GPU (16 si fort)
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none"  # Pas de wandb/tensorboard si pas installé
    )

    # 8. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 9. Train !
    logger.info("Début du fine-tuning...")
    trainer.train()

    # 10. Éval finale & save
    eval_results = trainer.evaluate()
    logger.info(f"Résultats éval : {eval_results}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Modèle sauvé dans {output_dir}")

    return trainer

def load_fine_tuned_model(model_path: str = "./model_output"):
    """
    Charge le modèle fine-tuné pour inférence.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle non trouvé : {model_path}")
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()  # Mode inférence
    return model, tokenizer

# Exemple d'usage
if __name__ == "__main__":
    # Train (subsample si test : df = df.sample(1000) dans preprocess)
    trainer = train_model()
    
    # Load pour test
    model, tokenizer = load_fine_tuned_model()
    print("Modèle chargé !")