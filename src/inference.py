# inference.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from data_processing import clean_text
import argparse
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def predict_sentiment(model_path: str = "./model_output", text: str = "") -> Tuple[str, float]:
    """
    Prédit le sentiment sur un nouveau texte.
    Étapes : Clean → Tokenize → Inférence avec modèle fine-tuné.
    
    Args:
        model_path (str): Chemin vers le modèle sauvé.
        text (str): Texte à analyser.
    
    Returns:
        Tuple[str, float]: ('positive'/'negative', probabilité max).
    """
    if not text.strip():
        raise ValueError("Texte vide – fournissez un texte valide.")
    
    # 1. Load model & tokenizer (vérifier d'abord que le chemin existe pour des messages d'erreur clairs)
    import os
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle non trouvé : {model_path}")

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
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
        print(f"\nRésultat : Sentiment = {label.upper()} (Confiance : {conf:.2%})\n")
    except Exception as e:
        logger.error(f"Erreur lors de l'inférence : {e}")
        print("Erreur – vérifiez le modèle et le texte.")

# Exemple d'usage
if __name__ == "__main__":
    main()