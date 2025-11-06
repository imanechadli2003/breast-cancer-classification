#  Breast cancer classification

##  Objectif du projet
Ce projet a pour but d’**implémenter manuellement la régression logistique** à partir de zéro (sans utiliser `sklearn.linear_model.LogisticRegression`) afin de :

- comprendre le fonctionnement mathématique du modèle ;
- maîtriser la **descente de gradient** et la **log-vraisemblance** ;
- construire les **métriques d’évaluation** et les **visualisations** associées ;
- comparer les résultats à une implémentation de référence (`sklearn`).

Le jeu de données utilisé est celui du **cancer du sein** fourni par Scikit-learn.

---

## 📦 Contenu du projet

### 1️⃣ Structure du code

Le script est organisé en **six grandes sections** :

| Section | Description |
|----------|--------------|
| **1. Fonctions de base** | Définition du modèle logistique : `sigmoid`, `log_likelihood`, `gradient`, `logistic_regression` |
| **2. Métriques d’évaluation** | Calcul manuel : accuracy, précision, rappel, F1-score, matrice de confusion |
| **3. Visualisations** | Tracés : log-vraisemblance, prédictions vs observations, matrice de confusion, courbe ROC |
| **4. PCA manuelle** | Réduction dimensionnelle pour visualiser la frontière de décision en 2D |
| **5. Pipeline principal (`main`)** | Préparation des données, apprentissage, évaluation et affichage des résultats |
| **6. Exécution** | Lancement automatique du pipeline complet |

---

## ⚙️ Exécution du projet

### 🧩 Prérequis

Installe les bibliothèques nécessaires :

```bash
pip install numpy matplotlib seaborn scikit-learn
