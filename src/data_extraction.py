# data_extraction.py
import pandas as pd
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_sentiment_data(file_path: str, text_col: str = 'content', score_col: str = 'score') -> pd.DataFrame:
    """
    Charge les données depuis un fichier CSV contenant des avis et leurs scores.
    Vérifie la présence des colonnes et nettoie les données.
    
    Args:
        file_path (str): chemin vers le fichier CSV
        text_col (str): nom de la colonne contenant le texte (par défaut: 'content')
        score_col (str): nom de la colonne contenant le score (par défaut: 'score')
    
    Returns:
        pd.DataFrame: DataFrame nettoyé avec les colonnes demandées
    """
    if not os.path.exists(file_path):
        logger.error(f"Fichier manquant : {file_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path)
        logger.info(f"Données chargées : {len(df)} lignes, colonnes : {df.columns.tolist()}")

        # Vérifier la présence des colonnes requises
        required_columns = [text_col, score_col]
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Colonnes '{text_col}' et '{score_col}' requises dans le dataset.")
            return pd.DataFrame()

        # Nettoyer le texte
        df[text_col] = df[text_col].astype(str).str.strip()
        df = df.dropna(subset=[text_col, score_col])
        df = df[df[text_col].str.len() > 0]

        # Vérifier que les scores sont numériques
        df[score_col] = pd.to_numeric(df[score_col], errors='coerce')
        df = df.dropna(subset=[score_col])

        logger.info(f"DataFrame nettoyé : {len(df)} lignes avec scores valides.")
        return df

    except pd.errors.EmptyDataError:
        logger.error("Fichier CSV vide.")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Erreur inattendue : {e}")
        return pd.DataFrame()

# Exemple d'usage
if __name__ == "__main__":
    # Chemin vers votre fichier de données
    file_path = r"C:\Users\jeora\Downloads\dataset.csv"
    
    # Charger les données avec les noms de colonnes corrects
    df = load_sentiment_data(
        file_path,
        text_col='content',  # Nom de la colonne contenant le texte
        score_col='score'    # Nom de la colonne contenant le score
    )

    if not df.empty:
        print("\nAperçu des données :")
        print(df[['content', 'score']].head())
        print(f"\nNombre total de lignes : {len(df)}")
        print("\nDistribution des scores :")
        print(df['score'].value_counts().sort_index())
    else:
        print("Aucune donnée chargée.")
