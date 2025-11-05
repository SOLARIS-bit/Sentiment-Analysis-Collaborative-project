import re
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# ======================================
# 1️⃣ Nettoyage du texte
# ======================================

def clean_text(text: str) -> str:
    """
    Nettoyage inspiré de l'approche Kaggle :
    - Supprime les balises HTML
    - Convertit en minuscules
    - Supprime la ponctuation et les caractères spéciaux
    - Normalise les espaces
    """
    if not isinstance(text, str):
        return ""

    # Remplacer les balises HTML et entités par un espace
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"&\w+;", " ", text)

    # Tout en minuscules
    text = text.lower()

    # Supprimer tout sauf lettres, chiffres et espaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Réduire les espaces multiples
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ======================================
# 2️⃣ Prétraitement du DataFrame
# ======================================

def preprocess_texts(df: pd.DataFrame,
                     text_col: str = "content",
                     score_col: str = "score",
                     min_len: int = 10) -> pd.DataFrame:
    """
    - Nettoie le texte
    - Filtre les textes vides ou trop courts
    - Crée une colonne 'sentiment' à partir du score (0 = négatif, 1 = neutre, 2 = positif)
    """
    df = df.copy()

    # Nettoyage du texte
    df.loc[:, text_col] = df[text_col].fillna("").apply(clean_text)

    # Filtrer les textes trop courts
    df = df[df[text_col].str.len() > min_len].reset_index(drop=True)

    # Mapper le score (1-5) vers des classes sentiment
    if score_col in df.columns:
        df.loc[:, "sentiment"] = df[score_col].apply(
            lambda x: 0 if x <= 2 else (2 if x >= 4 else 1)
        )

    return df


# ======================================
# 3️⃣ Tokenisation et création des datasets
# ======================================

def tokenize_data(df: pd.DataFrame,
                  text_col: str = "content",
                  label_col: str = "sentiment",
                  tokenizer_name: str = "bert-base-uncased",
                  max_length: int = 128,
                  test_size: float = 0.2,
                  random_state: int = 42):
    """
    - Nettoie le texte (prétraitement)
    - Split en train/val
    - Tokenise avec AutoTokenizer (Hugging Face)
    - Retourne des dictionnaires de tenseurs torch :
      {'train': {...}, 'val': {...}}
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Nettoyage et filtrage
    df_clean = preprocess_texts(df, text_col=text_col)

    if len(df_clean) < 2:
        raise ValueError("Pas assez de données après nettoyage pour split train/val")

    # Split train/val
    train_df, val_df = train_test_split(
        df_clean,
        test_size=test_size,
        random_state=random_state,
        stratify=df_clean[label_col] if label_col in df_clean else None,
    )

    def tokenize_df(dataframe):
        encodings = tokenizer(
            dataframe[text_col].tolist(),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        labels = torch.tensor(dataframe[label_col].tolist(), dtype=torch.long)
        encodings["labels"] = labels
        return encodings

    train_enc = tokenize_df(train_df)
    val_enc = tokenize_df(val_df)

    return {"train": train_enc, "val": val_enc}
