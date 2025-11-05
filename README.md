# Projet d'Analyse de Sentiments Collaboratif

## Organisation du Projet

Le projet est divisé en plusieurs phases avec une répartition claire des tâches :

### Phase 1 : Extraction de Données
- **Lead** : Student 1
- **Tâches** : Chargement et validation des données
- **Fichiers** : `src/data_extraction.py`, `tests/unit/test_data_extraction.py`

### Phase 2 : Traitement des Données
- **Lead** : Les deux étudiants
- **Tâches** : 
  - Nettoyage des textes
  - Tokenization
  - Préparation des données pour BERT
- **Fichiers** : `src/data_processing.py`, `tests/unit/test_data_processing.py`

### Phase 3 : Entraînement du Modèle
- **Lead** : Student 2
- **Tâches** : Fine-tuning de BERT pour la classification
- **Fichiers** : `src/model.py`, `tests/unit/test_model.py`

### Phase 4 : Inférence
- **Lead** : Student 2
- **Support** : Student 1 (documentation, tests)
- **Fichiers** : `src/inference.py`, `tests/unit/test_inference.py`

## Guide Rapide

### Installation

1. Créer l'environnement virtuel :
```bash
python -m venv sentiment-env
```

2. Activer l'environnement :
```bash
# Windows
sentiment-env\Scripts\activate
# Linux/Mac
source sentiment-env/bin/activate
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

### Entraînement du Modèle

Pour entraîner le modèle (nécessite un GPU) :
```bash
python src/model.py
```

Pour un entraînement rapide (test/développement) :
```bash
python src/model.py --fast_dev_run
```

### Inférence / Prédiction

Pour analyser un texte :
```bash
python src/inference.py --text "Your text here"
```

Ou en mode interactif :
```bash
python src/inference.py
```

### Tests

Lancer tous les tests :
```bash
pytest
```

Tests spécifiques :
```bash
# Tests du modèle uniquement
pytest tests/unit/test_model.py

# Tests d'inférence uniquement
pytest tests/unit/test_inference.py
```

## Notes Importantes

### Ressources Requises

- L'entraînement complet nécessite un GPU avec au moins 4GB de VRAM
- Pour développer/tester sans GPU :
  - Utilisez `--fast_dev_run` qui limite les données et époques
  - Les tests utilisent des mini-batches et peuvent s'exécuter sur CPU
  - L'inférence peut fonctionner sur CPU (plus lente)

### Gestion des Erreurs Courantes

1. `FileNotFoundError: Modèle non trouvé`
   - Solution : Exécutez d'abord `python src/model.py` pour entraîner

2. `CUDA out of memory`
   - Solution : Réduisez `batch_size` dans `model.py` ou utilisez `--fast_dev_run`

3. `ImportError: No module named 'src'`
   - Solution : Assurez-vous d'être dans le dossier racine du projet