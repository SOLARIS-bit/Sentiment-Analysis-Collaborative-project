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

def train_model(data_path: str =  "dataset.csv", output_dir: str = "./model_output", num_epochs: int = 3, max_length: int = 512, fast_dev_run: bool = False):
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
    # 1. Load data (option fast_dev_run pour tests rapides sans dataset)
    if fast_dev_run:
        # Tiny toy dataset for fast local runs / CI
        df = pd.DataFrame({
            'content': [
                'I loved this movie, it was fantastic and thrilling!',
                'Terrible film, I hated it and it was boring.',
                'It was okay, not great but watchable.',
                'Amazing performance and great story.'
            ],
            # Use scores on 1-5 scale so preprocess_texts maps correctly
            'score': [5, 1, 3, 5]
        })
        logger.info("fast_dev_run activé : dataset minimal créé pour tests locaux.")
    else:
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
        evaluation_strategy=True,
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

# Exemple d'usage / CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune BERT pour classification de sentiments (script d'aide).")
    parser.add_argument("--data_path", type=str, default="Dataset.csv", help="Chemin vers le CSV de données")
    parser.add_argument("--output_dir", type=str, default="./model_output", help="Dossier de sortie pour le modèle")
    parser.add_argument("--num_epochs", type=int, default=3, help="Nombre d'époques")
    parser.add_argument("--max_length", type=int, default=512, help="Max token length")
    parser.add_argument("--fast_dev_run", action='store_true', help="Utiliser un tiny dataset pour un test local rapide (CPU)")

    args = parser.parse_args()

    csv_path = r"C:\Users\jeora\Downloads\dataset.csv"
    args.data_path = csv_path

    print(f"✅ Chargement du dataset depuis : {args.data_path}")

    # Lance l'entraînement (fast_dev_run utile si pas de dataset ou pas de GPU)
    trainer = train_model(
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        max_length=args.max_length,
        fast_dev_run=args.fast_dev_run
    )

    # Exemple de chargement post-train (optionnel)
    try:
        model, tokenizer = load_fine_tuned_model(args.output_dir)
        print("Modèle chargé depuis :", args.output_dir)
    except Exception:
        print("Modèle non trouvé dans le dossier de sortie (vérifiez que l'entraînement a bien sauvegardé le modèle).")