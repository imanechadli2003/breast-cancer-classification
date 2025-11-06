# ==============================================================
#  RÉGRESSION LOGISTIQUE - IMPLÉMENTATION MANUELLE COMPLÈTE
#  Jeu de données : Cancer du sein (Breast Cancer Dataset)
# ==============================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split


# ==============================================================
#   FONCTIONS DE BASE - SIGMOÏDE, LOG-VRAISEMBLANCE, GRADIENT
# ==============================================================

def sigmoid(z):
    """Fonction sigmoïde : transforme une valeur réelle en probabilité [0,1]."""
    return 1 / (1 + np.exp(-z))


def log_likelihood_logistic(w, X, y):
    """Calcule la log-vraisemblance d’un modèle logistique."""
    eta = np.clip(X @ w, -30, 30)
    p = sigmoid(eta)
    return np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))


def gradient_logistic(w, X, y):
    """Calcule le gradient de la log-vraisemblance."""
    eta = np.clip(X @ w, -30, 30)
    p = sigmoid(eta)
    return X.T @ (y - p)


def logistic_regression(X, y, lr=1e-4, max_iter=10000, tol=1e-6, X_val=None, y_val=None):
    """Implémentation manuelle de la descente de gradient pour la régression logistique."""
    w = np.zeros(X.shape[1])
    ll_history_train, ll_history_val = [], []

    for iteration in range(max_iter):
        grad = gradient_logistic(w, X, y)
        w = w + lr * grad

        ll_train = log_likelihood_logistic(w, X, y)
        ll_history_train.append(ll_train)

        if X_val is not None and y_val is not None:
            ll_val = log_likelihood_logistic(w, X_val, y_val)
            ll_history_val.append(ll_val)

        # Critère d’arrêt (convergence)
        if iteration > 0 and abs(ll_history_train[-1] - ll_history_train[-2]) < tol:
            print(f"Convergence atteinte en {iteration} itérations.")
            break

    return w, ll_history_train, ll_history_val


# ==============================================================
#  MÉTRIQUES D’ÉVALUATION MANUELLES
# ==============================================================

def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def compute_precision(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    return TP / (TP + FP + 1e-12)

def compute_recall(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP / (TP + FN + 1e-12)

def compute_f1(precision, recall):
    return 2 * (precision * recall) / (precision + recall + 1e-12)

def compute_confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[TN, FP], [FN, TP]])


# ==============================================================
#   OUTILS DE VISUALISATION
# ==============================================================

def plot_log_likelihood(history_train, history_val, y_train, y_val):
    plt.figure(figsize=(8, 5))
    plt.plot(np.array(history_train) / len(y_train), label="Train", color="green")
    plt.plot(np.array(history_val) / len(y_val), label="Validation", color="orange")
    plt.xlabel("Itérations")
    plt.ylabel("Log-vraisemblance normalisée")
    plt.title("Évolution de la log-vraisemblance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_predictions_vs_observations(p_pred, y_val):
    plt.figure(figsize=(8, 5))
    plt.scatter(p_pred, y_val, alpha=0.6, color='royalblue')
    plt.axvline(x=0.5, color='red', linestyle='--', linewidth=1.5, label='Seuil de décision (0.5)')
    plt.xlabel("Probabilité prédite")
    plt.ylabel("Classe observée")
    plt.title("Régression Logistique | Prédictions vs Observations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(conf_mat):
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Classe 0", "Classe 1"],
                yticklabels=["Classe 0", "Classe 1"])
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.title("Matrice de confusion")
    plt.tight_layout()
    plt.show()

def plot_roc_curve(fpr, tpr, auc):
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}", color='darkorange')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("Faux positifs (FPR)")
    plt.ylabel("Vrais positifs (TPR)")
    plt.title("Courbe ROC")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ==============================================================
#   PCA MANUELLE + FRONTIÈRE DE DÉCISION
# ==============================================================

def manual_pca(X, n_components=2):
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    sorted_idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, sorted_idx]
    components = eigvecs[:, :n_components]
    X_proj = X_centered @ components
    return X_proj, components

def plot_decision_boundary_pca(X, y, w, title="Frontière de décision (PCA manuelle)"):
    X_proj, components = manual_pca(X, n_components=2)
    x_min, x_max = X_proj[:, 0].min() - 1, X_proj[:, 0].max() + 1
    y_min, y_max = X_proj[:, 1].min() - 1, X_proj[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    X_grid_original = grid @ components.T + np.mean(X, axis=0)
    X_grid_augmented = np.column_stack((np.ones(X_grid_original.shape[0]), X_grid_original))
    probs = sigmoid(X_grid_augmented @ w).reshape(xx.shape)

    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, probs, levels=[0, 0.5, 1], alpha=0.2, colors=["blue", "orange"])
    plt.contour(xx, yy, probs, levels=[0.5], colors='k', linewidths=1)
    plt.scatter(X_proj[y == 0, 0], X_proj[y == 0, 1], c='blue', label="Classe 0", alpha=0.6)
    plt.scatter(X_proj[y == 1, 0], X_proj[y == 1, 1], c='orange', label="Classe 1", alpha=0.6)
    plt.xlabel("Composante principale 1")
    plt.ylabel("Composante principale 2")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ==============================================================
# 5️  PIPELINE PRINCIPAL
# ==============================================================

def main():
    # --- Chargement et séparation ---
    data = load_breast_cancer()
    X_raw, y = data.data, data.target
    feature_names = data.feature_names
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(X_raw, y, test_size=0.2, random_state=8302)

    # --- Standardisation ---
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train_raw)
    X_val_std = scaler.transform(X_val_raw)

    # --- Ajout de l’intercept ---
    X_train = np.column_stack((np.ones(X_train_std.shape[0]), X_train_std))
    X_val = np.column_stack((np.ones(X_val_std.shape[0]), X_val_std))

    # --- Entraînement du modèle ---
    beta_hat, history_train, history_val = logistic_regression(X_train, y_train, X_val=X_val, y_val=y_val)

    # --- Prédictions ---
    p_pred = sigmoid(X_val @ beta_hat)
    y_pred_class = (p_pred >= 0.5).astype(int)

    # --- Coefficients estimés ---
    print("\nCoefficients estimés :")
    for name, coef in zip(['Intercept'] + list(feature_names), beta_hat):
        print(f"{name} : {coef:.4f}")

    # --- Visualisations apprentissage ---
    plot_log_likelihood(history_train, history_val, y_train, y_val)
    plot_predictions_vs_observations(p_pred, y_val)

    # --- Évaluation ---
    accuracy = compute_accuracy(y_val, y_pred_class)
    precision = compute_precision(y_val, y_pred_class)
    recall = compute_recall(y_val, y_pred_class)
    f1 = compute_f1(precision, recall)
    conf_mat = compute_confusion_matrix(y_val, y_pred_class)
    fpr, tpr, _ = roc_curve(y_val, p_pred)
    auc = roc_auc_score(y_val, p_pred)

    print("\nMétriques d'évaluation :")
    print(f"Accuracy  : {accuracy:.3f}")
    print(f"Précision : {precision:.3f}")
    print(f"Rappel    : {recall:.3f}")
    print(f"F1-score  : {f1:.3f}")
    print(f"AUC       : {auc:.3f}")

    # --- Visualisations finales ---
    plot_confusion_matrix(conf_mat)
    plot_roc_curve(fpr, tpr, auc)
    plot_decision_boundary_pca(X_val[:, 1:], y_val, beta_hat)


# ==============================================================
#   EXÉCUTION
# ==============================================================

if __name__ == "__main__":
    main()
