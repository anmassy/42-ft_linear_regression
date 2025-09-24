# 🤖 ft_linear_regression

## 📋 Description
Implémentation complète d'une **régression linéaire from scratch** en Python, développée dans le cadre du cursus **École 42**.

Ce projet démontre la compréhension des algorithmes de Machine Learning fondamentaux sans utiliser de bibliothèques spécialisées comme scikit-learn.

## 🎯 Objectifs
- Implémenter l'algorithme de **descente de gradient**
- Prédire le prix d'une voiture en fonction du kilométrage
- Maîtriser la **normalisation des données**
- Créer une architecture modulaire avec 2 programmes distincts

## 🛠️ Technologies utilisées
![Python](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557c?style=flat)
![NumPy](https://img.shields.io/badge/-Math-013243?style=flat)

## 📁 Structure du projet
```
ft_linear_regression/
├── 📄 README.md           # Documentation
├── 🐍 train.py           # Programme d'entraînement
├── 🐍 predict.py         # Programme de prédiction
├── 📊 data.csv           # Dataset (kilométrage, prix)
├── 📈 theta.csv          # Paramètres sauvegardés
└── 🖼️ regression_graphique.png  # Visualisation
```

## 🚀 Installation & Utilisation

### Prérequis
```bash
# Installation de matplotlib (optionnel pour la visualisation)
pip install matplotlib
```

### Entraînement du modèle
```bash
python3 train.py
```
**Sortie attendue :**
- Affichage du progrès d'entraînement
- Génération du graphique de régression
- Sauvegarde des paramètres θ0 et θ1
- Tests de prédiction automatiques

### Prédiction de prix
```bash
python3 predict.py
```
**Utilisation :**
1. Saisissez le kilométrage souhaité
2. Le programme affiche le prix estimé
3. Répétez ou quittez

## 📊 Algorithme implémenté

### Formule de prédiction
```
Prix estimé = θ0 + (θ1 × kilométrage)
```

### Descente de gradient
```python
# Calcul des gradients
gradient_θ0 = (1/m) × Σ(prédiction - prix_réel)
gradient_θ1 = (1/m) × Σ((prédiction - prix_réel) × kilométrage)

# Mise à jour des paramètres
θ0 = θ0 - learning_rate × gradient_θ0
θ1 = θ1 - learning_rate × gradient_θ1
```

### Normalisation Z-score
```python
valeur_normalisée = (valeur - moyenne) / écart_type
```

## 📈 Exemple de résultats

```bash
=== TESTS DE PRÉDICTION ===
50,000 km → 7,427.15€
100,000 km → 6,354.70€
150,000 km → 5,282.26€
200,000 km → 4,209.81€
```

## 🎨 Visualisation
Le programme génère automatiquement un graphique montrant :
- Points de données réels (scatter plot)
- Droite de régression linéaire
- Équation de la régression

## ⚙️ Paramètres techniques
- **Learning rate** : 0.01 (ajustable)
- **Époques** : 2000 iterations
- **Normalisation** : Z-score complète
- **Algorithme** : Gradient Descent Batch

## 🧠 Compétences démontrées
- **Machine Learning** : Régression linéaire, descente de gradient
- **Mathématiques** : Statistiques, optimisation
- **Python** : Programmation orientée fonctions, gestion de fichiers
- **Data Science** : Normalisation, visualisation
- **Architecture logicielle** : Modularité, séparation des responsabilités

## 📚 Concepts abordés
- ✅ Algorithmes d'apprentissage supervisé
- ✅ Fonction de coût (Mean Squared Error)
- ✅ Optimisation par gradient descent
- ✅ Preprocessing des données
- ✅ Validation de modèles

## 🎓 Contexte académique
**École 42** - Projet ft_linear_regression  
**Section** : Intelligence Artificielle  
**Objectif** : Comprendre les fondements du Machine Learning

## 📝 Notes d'implémentation
- Pas d'utilisation de bibliothèques ML (scikit-learn, etc.)
- Code from scratch pour une compréhension approfondie
- Gestion robuste des erreurs et edge cases
- Interface utilisateur claire et informative

---
*Développé avec passion pour l'IA et l'apprentissage automatique* 🚀
