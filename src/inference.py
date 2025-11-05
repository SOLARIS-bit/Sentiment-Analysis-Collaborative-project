# inference.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from data_processing import clean_text
import argparse
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def predict_sentiment(model_path: str = "./model_output", text: str = "", model=None, tokenizer=None) -> Tuple[str, float]:
    """
    Prédit le sentiment sur un nouveau texte.
    Étapes : Clean → Tokenize → Inférence avec modèle fine-tuné.
    
    Args:
        model_path (str): Chemin vers le modèle sauvé.
        text (str): Texte à analyser.
        model: Modèle déjà chargé (optionnel)
        tokenizer: Tokenizer déjà chargé (optionnel)
    
    Returns:
        Tuple[str, float]: ('positive'/'negative', probabilité max).
    """
    if not text.strip():
        raise ValueError("Texte vide – fournissez un texte valide.")
    
    # 1. Load model & tokenizer si non fournis
    if model is None or tokenizer is None:
        import os
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Modèle non trouvé : {model_path}\n"
                "Vous devez d'abord entraîner le modèle avec model.py, ou fournir un chemin valide."
            )
        
        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement du modèle : {e}")
    
    model.eval()
    
    # 2. Clean text
    cleaned_text = clean_text(text)
    logger.info(f"Texte nettoyé : {cleaned_text[:100]}...")
    
    # 3. Tokenize
    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # 4. Inférence
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # 5. Mapper à label
    label = "positive" if predicted_class == 1 else "negative"
    
    logger.info(f"Prédiction : {label} (confiance: {confidence:.2%})")
    return label, confidence

def main():
    parser = argparse.ArgumentParser(description="Interface CLI pour prédiction de sentiment avec BERT fine-tuné.")
    parser.add_argument("--text", type=str, default="", help="Texte à analyser (ex: 'I love this movie!')")
    parser.add_argument("--model_path", type=str, default="./model_output", help="Chemin vers le modèle")
    
    args = parser.parse_args()
    
    if not args.text:
        args.text = input("Entrez le texte à analyser : ").strip()
    
    try:
        label, conf = predict_sentiment(args.model_path, args.text)
        print(f"\nRésultat d'analyse :")
        print(f"  - Sentiment : {label.upper()}")
        print(f"  - Confiance : {conf:.2%}")
        print("\nNote : Un score > 50% indique que le modèle est plus confiant dans sa prédiction.")
    except Exception as e:
        logger.error(f"Erreur lors de l'inférence : {e}")
        print("\nErreur – Assurez-vous que :")
        print("1. Le modèle a été entraîné (utilisez model.py)")
        print("2. Le chemin du modèle est correct (par défaut: ./model_output)")
        print("3. Le texte n'est pas vide")
        print("\nMessage d'erreur complet :", str(e))

# Exemple d'usage
if __name__ == "__main__":
    main()