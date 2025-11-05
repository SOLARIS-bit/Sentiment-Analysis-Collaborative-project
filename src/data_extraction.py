# data_extraction.py
import pandas as pd
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_sentiment_data(file_path: str) -> pd.DataFrame:
    """
    Charge les données depuis un fichier CSV contenant des avis et leurs sentiments.
    Vérifie la présence des colonnes, nettoie les données et convertit les sentiments.
    
    Args:
        file_path (str): chemin vers le fichier CSV.
    
    Returns:
        pd.DataFrame: DataFrame nettoyé avec sentiments convertis (1 pour positif, 0 pour négatif)
    """
    if not os.path.exists(file_path):
        logger.error(f"Fichier manquant : {file_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path)
        logger.info(f"Données chargées : {len(df)} lignes, colonnes : {df.columns.tolist()}")

        # Vérifier la présence des colonnes requises
        required_columns = ['review', 'sentiment']
        if not all(col in df.columns for col in required_columns):
            logger.error("Colonnes 'review' et 'sentiment' requises dans le dataset.")
            return pd.DataFrame()

        # Nettoyer le texte
        df['review'] = df['review'].astype(str).str.strip()
        df = df.dropna(subset=['review', 'sentiment'])
        df = df[df['review'].str.len() > 0]

        # Convertir les sentiments en valeurs numériques
        sentiment_map = {'positive': 1, 'negative': 0}
        df['sentiment'] = df['sentiment'].map(sentiment_map)
        df = df.dropna(subset=['sentiment'])

        logger.info(f"DataFrame nettoyé : {len(df)} lignes avec sentiments convertis.")
        return df

    except pd.errors.EmptyDataError:
        logger.error("Fichier CSV vide.")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Erreur inattendue : {e}")
        return pd.DataFrame()

# Exemple d'usage
if __name__ == "__main__":
    file_path = "dataset.csv"  # Mettre le chemin vers votre fichier de données
    df = load_sentiment_data(file_path)

    if not df.empty:
        print(df.head())
        print(f"\nNombre total de lignes : {len(df)}")
        print("\nDistribution des sentiments :")
        print(df['sentiment'].value_counts())
    else:
        print("Aucune donnée chargée.")
