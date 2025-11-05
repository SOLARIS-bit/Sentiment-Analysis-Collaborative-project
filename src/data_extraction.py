# data_extraction.py
import pandas as pd
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_sentiment_data(file_path: str) -> pd.DataFrame:
    """
    Charge les données depuis un fichier CSV (dataset d'avis Google Play).
    Ne fait pas d'analyse de sentiment. 
    Vérifie simplement la présence des colonnes et nettoie les données.
    
    Args:
        file_path (str): chemin vers le fichier CSV.
    
    Returns:
        pd.DataFrame: DataFrame nettoyé
    """
    if not os.path.exists(file_path):
        logger.error(f"Fichier manquant : {file_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path)
        logger.info(f"Données chargées : {len(df)} lignes, colonnes : {df.columns.tolist()}")

        # Vérifier la présence de la colonne de texte
        if 'content' not in df.columns:
            logger.error("Colonne 'content' manquante dans le dataset.")
            return pd.DataFrame()

        # Nettoyer le texte
        df['content'] = df['content'].astype(str).str.strip()
        df = df.dropna(subset=['content'])
        df = df[df['content'].str.len() > 0]

        logger.info(f"DataFrame nettoyé : {len(df)} lignes.")
        return df

    except pd.errors.EmptyDataError:
        logger.error("Fichier CSV vide.")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Erreur inattendue : {e}")
        return pd.DataFrame()

# Exemple d'usage
if __name__ == "__main__":
    file_path = r"C:\Users\jeora\Downloads\dataset.csv"
    df = load_sentiment_data(file_path)

    if not df.empty:
        print(df.head())
        print(f"\nNombre total de lignes : {len(df)}")
    else:
        print("Aucune donnée chargée.")
