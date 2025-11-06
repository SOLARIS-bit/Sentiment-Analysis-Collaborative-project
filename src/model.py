import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from data_extraction import load_data  # Assume tu as une func pour charger les raw data
from data_processing import preprocess_data  # Assume pour tokeniser et splitter

# Chargement des données (adapte selon ton setup)
df = load_data('path/to/your/sentiment_data.csv')  # Ex: colonnes 'text', 'label'
train_df, val_df = preprocess_data(df)  # Retourne DataFrames tokenisés (ajoute 'input_ids', 'attention_mask', 'labels')

# Conversion en Dataset Hugging Face
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Tokenizer (déjà fait dans preprocessing, mais on recharge pour cohérence)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    # Si pas déjà tokenisé, décommente
    # return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
    return examples  # Si déjà tokenisé, passe direct

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Chargement du modèle pour classification binaire (2 classes: neg/pos)
model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2,  # Binaire: 0=neg, 1=pos
    output_attentions=False,
    output_hidden_states=False,
)

# Args d'entraînement
training_args = TrainingArguments(
    output_dir='./results',  # Dossier pour checkpoints
    num_train_epochs=3,  # 3 epochs pour tester (adapte)
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy='epoch',  # Éval à chaque epoch
    save_strategy='epoch',
    load_best_model_at_end=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=lambda p: {'accuracy': (p.predictions.argmax(-1) == p.label_ids).mean()},  # Métrique simple
)

# Fine-tuning
trainer.train()

# Sauvegarde
model.save_pretrained('./fine_tuned_bert')
tokenizer.save_pretrained('./fine_tuned_bert')

print("Modèle fine-tuné sauvegardé !")