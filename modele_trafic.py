import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── 1. Données ────────────────────────────────────────────────────────────────
df = pd.read_csv("dataset_trafic.csv")
categories = ["axe", "jour", "meteo", "etat_route", "travaux"]
numeriques = ["heure"]
niveaux = ["Fluide", "Dense", "Embouteillé"]

X = df[categories + numeriques]
y = df["niveau_trafic"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"📚 Entraînement : {len(X_train)} lignes | 🧪 Test : {len(X_test)} lignes\n")

# ── 2. Pipeline : encodage des mots + modèle ──────────────────────────────────
def construire(modele):
    prep = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), categories)],
        remainder="passthrough",
    )
    return Pipeline([("prep", prep), ("clf", modele)])

candidats = {
    "Arbre de décision": construire(DecisionTreeClassifier(max_depth=6, random_state=42)),
    "Forêt aléatoire": construire(RandomForestClassifier(n_estimators=200, random_state=42)),
}

# ── 3. Comparer les modèles (validation croisée sur l'entraînement) ───────────
print("── Comparaison des modèles (validation croisée, 5 plis) ──")
scores = {}
for nom, m in candidats.items():
    scores[nom] = cross_val_score(m, X_train, y_train, cv=5).mean()
    print(f"{nom:20s} : {scores[nom] * 100:.1f} %")

meilleur_nom = max(scores, key=scores.get)
modele = candidats[meilleur_nom]
modele.fit(X_train, y_train)
print(f"\n🏆 Modèle retenu : {meilleur_nom}")

# ── 4. Évaluation finale sur les données de test ──────────────────────────────
y_pred = modele.predict(X_test)
print(f"\n🎯 Précision sur le test : {accuracy_score(y_test, y_pred) * 100:.1f} %\n")
print(classification_report(y_test, y_pred, labels=niveaux))

cm = confusion_matrix(y_test, y_pred, labels=niveaux)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=niveaux, yticklabels=niveaux)
plt.title(f"Matrice de confusion - {meilleur_nom}")
plt.xlabel("Niveau prédit"); plt.ylabel("Niveau réel")
plt.tight_layout(); plt.savefig("modele_confusion.png", dpi=150); plt.close()

# ── 5. Variables les plus importantes ─────────────────────────────────────────
noms = modele.named_steps["prep"].get_feature_names_out()
noms = [n.replace("cat__", "").replace("remainder__", "") for n in noms]
imp = pd.Series(modele.named_steps["clf"].feature_importances_, index=noms)
imp = imp.sort_values().tail(10)
plt.figure(figsize=(7, 5))
imp.plot(kind="barh", color="#3B5BDB")
plt.title("Les 10 variables les plus importantes")
plt.tight_layout(); plt.savefig("modele_importance.png", dpi=150); plt.close()

# ── 6. Sauvegarde pour le dashboard ───────────────────────────────────────────
joblib.dump(modele, "modele_trafic.joblib")
print("✅ Modèle sauvegardé : modele_trafic.joblib")

# ── 7. Test rapide : samedi, 18h, sous la pluie, à l'Escale ──────────────────
exemple = pd.DataFrame([{
    "axe": "Escale", "jour": "Samedi", "meteo": "Pluie",
    "etat_route": "Moyen", "travaux": "Non", "heure": 18,
}])
print("\n🔮 Prédiction test (Escale, samedi 18h, pluie) :", modele.predict(exemple)[0])
