# data_processing.py
import re
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import pandas as pd
import logging
import torch

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    Nettoie le texte : enlève HTML, ponctuation/non-alpha, lowercase, normalisation basique.
    Inspiré du Kaggle : remove <br />, etc.
    """
    if pd.isna(text):
        return ""
    
    # Enlever HTML tags (<br />)
    text = re.sub(r'<.*?>', '', text)
    # Enlever caractères non-alpha (garder espaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Lowercase
    text = text.lower().strip()
    # Normalisation basique (supprimer espaces multiples)
    text = re.sub(r'\s+', ' ', text)
    
    return text

def preprocess_texts(df: pd.DataFrame, text_col: str = 'review') -> pd.DataFrame:
    """
    Applique le nettoyage à la colonne texte.
    """
    df[text_col] = df[text_col].apply(clean_text)
    df = df[df[text_col].str.len() > 10]  # Filtrer textes trop courts
    logger.info(f"Après nettoyage : {len(df)} lignes.")
    return df

def tokenize_data(df: pd.DataFrame, text_col: str = 'review', max_length: int = 512, test_size: float = 0.2) -> dict:
    """
    Tokenise avec BERT tokenizer. Split train/val.
    Retourne dict : {'train': tensors, 'val': tensors} avec input_ids, attention_mask, labels.
    """
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    texts = df[text_col].tolist()
    labels = df['sentiment'].tolist()
    
    # Tokeniser
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'  # PyTorch tensors pour BERT
    )
    
    # Split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    # Tokeniser splits séparément (pour éviter fuites, mais ici on tokenise tout d'abord pour simplicité)
    # Note : En prod, tokenise après split
    train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    val_enc = tokenizer(val_texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    
    train_dataset = {
        'input_ids': train_enc['input_ids'],
        'attention_mask': train_enc['attention_mask'],
        'labels': torch.tensor(train_labels)
    }
    val_dataset = {
        'input_ids': val_enc['input_ids'],
        'attention_mask': val_enc['attention_mask'],
        'labels': torch.tensor(val_labels)
    }
    
    logger.info(f"Split : Train {len(train_texts)}, Val {len(val_texts)}")
    return {'train': train_dataset, 'val': val_dataset}

# Exemple d'usage
if __name__ == "__main__":
    from data_extraction import load_sentiment_data
    df = load_sentiment_data(r"C:\Users\jeora\Downloads\dataset.csv")
    df_clean = preprocess_texts(df)
    datasets = tokenize_data(df_clean)
    print("Exemple token input_ids shape:", datasets['train']['input_ids'].shape)