import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from data_processing import clean_text  # Assume func pour nettoyer texte

# Chargement du modèle fine-tuné
model_path = './fine_tuned_bert'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Pipeline d'inférence (super simple !)
sentiment_pipeline = pipeline(
    'sentiment-analysis',
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True  # Retourne probs pour neg/pos
)

def predict_sentiment(text):
    # Nettoie le texte
    cleaned_text = clean_text(text)  # Utilise ta func de preprocessing
    # Prédit
    result = sentiment_pipeline(cleaned_text)
    # Ex: [{'label': 'LABEL_1', 'score': 0.95}] -> map à pos/neg
    score_pos = result[0][1]['score'] if len(result[0]) > 1 else result[0][0]['score']
    sentiment = 'POSitif' if score_pos > 0.5 else 'NÉGatif'
    return {'text': cleaned_text, 'sentiment': sentiment, 'confidence': score_pos}

# Exemple d'usage
if __name__ == '__main__':
    user_text = input("Entrez un texte à analyser : ")
    prediction = predict_sentiment(user_text)
    print(f"Texte : {prediction['text']}")
    print(f"Sentiment : {prediction['sentiment']} (confiance : {prediction['confidence']:.2f})")