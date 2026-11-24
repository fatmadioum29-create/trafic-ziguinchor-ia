import pandas as pd
import numpy as np

np.random.seed(42)

# ── À ADAPTER : mettez ici les vrais axes / zones de Ziguinchor ──────────────
# Le "poids" représente l'importance du trafic de base (1.0 = moyen)
axes = {
    "Boucotte":  1.3,
    "Kandé":     1.2,
    "Néma":      1.0,
    "Escale":    1.4,   # zone commerçante / port
    "Lyndiane":  0.9,
    "Santhiaba": 0.8,
}

jours = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
etats_route = ["Bon", "Moyen", "Dégradé"]

# Un état de route fixe par axe (plus réaliste qu'un tirage aléatoire)
etat_par_axe = {
    "Boucotte": "Moyen", "Kandé": "Bon", "Néma": "Dégradé",
    "Escale": "Moyen", "Lyndiane": "Dégradé", "Santhiaba": "Bon",
}

def facteur_heure(h):
    """Heures de pointe : matin, midi, soir."""
    if 7 <= h <= 9:
        return 2.2
    if 12 <= h <= 14:
        return 1.5
    if 17 <= h <= 19:
        return 2.4
    if 22 <= h or h <= 5:
        return 0.2
    return 1.0

lignes = []
for _ in range(1500):
    axe = np.random.choice(list(axes))
    heure = np.random.randint(0, 24)
    jour = np.random.choice(jours)
    meteo = np.random.choice(["Soleil", "Pluie"], p=[0.8, 0.2])
    travaux = np.random.choice(["Non", "Oui"], p=[0.88, 0.12])
    etat = etat_par_axe[axe]

    # ── Calcul du nombre de véhicules par heure ──────────────────────────────
    base = 300 * axes[axe] * facteur_heure(heure)
    if jour == "Dimanche":
        base *= 0.6
    if jour == "Samedi" and axe == "Escale":
        base *= 1.3            # jour de marché
    if meteo == "Pluie":
        base *= 1.15           # plus de voitures, moins de motos/piétons
    base += np.random.normal(0, 40)   # bruit aléatoire réaliste
    nb_vehicules = max(int(base), 10)

    # ── Capacité de la route (ralentissements) ───────────────────────────────
    penalite = 1.0
    if etat == "Dégradé":
        penalite += 0.25
    if etat == "Moyen":
        penalite += 0.1
    if travaux == "Oui":
        penalite += 0.3
    if meteo == "Pluie":
        penalite += 0.15
    charge = nb_vehicules * penalite

    if charge < 400:
        niveau = "Fluide"
    elif charge < 800:
        niveau = "Dense"
    else:
        niveau = "Embouteillé"

    lignes.append({
        "axe": axe, "heure": heure, "jour": jour, "meteo": meteo,
        "etat_route": etat, "travaux": travaux,
        "nb_vehicules": nb_vehicules, "niveau_trafic": niveau,
    })

df = pd.DataFrame(lignes)
df.to_csv("dataset_trafic.csv", index=False)

print("✅ Dataset créé :", len(df), "lignes")
print(df.head())
print("\n── Répartition du niveau de trafic ──")
print(df["niveau_trafic"].value_counts())
