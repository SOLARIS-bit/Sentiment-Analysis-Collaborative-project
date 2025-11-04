# data_extraction.py
import pandas as pd
import os
import logging

# Setup logging pour tracer les erreurs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_sentiment_data(file_path: str) -> pd.DataFrame:
    """
    Charge les données de sentiment depuis un fichier CSV.
    Colonnes attendues : 'review' (texte), 'sentiment' ('positive'/'negative').
    Convertit 'sentiment' en 0/1. Gère les erreurs gracefully.
    
    Args:
        file_path (str): Chemin vers le CSV.
    
    Returns:
        pd.DataFrame: DataFrame nettoyé, ou vide si erreur.
    """
    if not os.path.exists(file_path):
        logger.error(f"Fichier manquant : {file_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Données chargées : {len(df)} lignes.")
        
        # Vérifier colonnes
        required_cols = ['review', 'sentiment']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Colonnes manquantes. Attendues : {required_cols}")
            return pd.DataFrame()
        
        # Convertir sentiment en numérique
        df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
        if df['sentiment'].isnull().any():
            logger.warning("Valeurs inattendues dans 'sentiment' – remplies par NaN.")
        
        # Supprimer lignes vides
        df = df.dropna(subset=['review', 'sentiment'])
        df = df[df['review'].str.len() > 0]
        
        logger.info(f"DataFrame final : {len(df)} lignes, colonnes : {df.columns.tolist()}")
        return df
    
    except pd.errors.EmptyDataError:
        logger.error("Fichier CSV vide.")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Erreur inattendue : {e}")
        return pd.DataFrame()

# Exemple d'usage
if __name__ == "__main__":
    df = load_sentiment_data(r"C:\Users\jeora\Downloads\dataset.csv")
    print(df.head())
    print(df['sentiment'].value_counts())