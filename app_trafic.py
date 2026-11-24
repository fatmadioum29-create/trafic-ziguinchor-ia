import os
from datetime import datetime

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Trafic Ziguinchor IA", page_icon="🚦", layout="wide")

NIVEAUX = ["Fluide", "Dense", "Embouteillé"]
COULEURS = {"Fluide": "#2E9E5B", "Dense": "#E8A33D", "Embouteillé": "#D64545"}
EMOJIS = {"Fluide": "🟢", "Dense": "🟠", "Embouteillé": "🔴"}
JOURS = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
TYPES = ["Accident", "Embouteillage", "Route dégradée", "Travaux", "Inondation"]
FICHIER_SIGNALEMENTS = "signalements.csv"

LOGO = ('<svg width="44" height="44" viewBox="0 0 44 44">'
        '<rect x="12" y="2" width="20" height="40" rx="8" fill="#0F1630"/>'
        '<circle cx="22" cy="12" r="5" fill="#D64545"/>'
        '<circle cx="22" cy="22" r="5" fill="#E8A33D"/>'
        '<circle cx="22" cy="32" r="5" fill="#2E9E5B"/></svg>')

st.markdown("""
<style>
.bandeau{display:flex;align-items:center;gap:16px;background:linear-gradient(90deg,#1E2761,#3B5BDB);
padding:18px 24px;border-radius:14px;margin-bottom:18px}
.bandeau h2{color:#fff;margin:0;padding:0;font-size:26px}
.bandeau p{color:#CADCFC;margin:2px 0 0 0;font-size:14px}
.resultat{padding:22px;border-radius:14px;color:#fff;text-align:center;font-size:32px;font-weight:700}
[data-testid="stMetric"]{background:#F4F7FF;border:1px solid #CADCFC;border-radius:12px;padding:12px 16px}
</style>
""", unsafe_allow_html=True)


def bandeau(titre, sous_titre):
    st.markdown(
        '<div class="bandeau">' + LOGO + '<div><h2>' + titre + '</h2><p>' + sous_titre + '</p></div></div>',
        unsafe_allow_html=True,
    )


# ── Chargement ────────────────────────────────────────────────────────────────
@st.cache_resource
def charger_modele():
    return joblib.load("modele_trafic.joblib")


@st.cache_data
def charger_donnees():
    return pd.read_csv("dataset_trafic.csv")


@st.cache_data
def charger_coordonnees():
    return pd.read_csv("axes_coordonnees.csv").set_index("axe")


try:
    modele = charger_modele()
    df = charger_donnees()
except FileNotFoundError:
    st.error("Fichiers manquants. Lancez d'abord creer_dataset_trafic.py puis modele_trafic.py.")
    st.stop()

try:
    coords = charger_coordonnees()
except FileNotFoundError:
    coords = None

axes = sorted(df["axe"].unique())
etat_par_axe = df.groupby("axe")["etat_route"].first().to_dict()


def construire_ligne(axe, jour, meteo, etat, travaux, heure):
    return {"axe": axe, "jour": jour, "meteo": meteo,
            "etat_route": etat, "travaux": travaux, "heure": heure}


def taux_embouteillage(sub):
    if len(sub) == 0:
        return 0.0
    return float((sub["niveau_trafic"] == "Embouteillé").mean() * 100)


def ajouter_cartes_coords(tableau):
    """Ajoute lat/lon à un tableau qui possède une colonne 'axe'."""
    if coords is None:
        return tableau.iloc[0:0]
    tableau = tableau.copy()
    tableau["lat"] = tableau["axe"].map(coords["lat"])
    tableau["lon"] = tableau["axe"].map(coords["lon"])
    return tableau.dropna(subset=["lat", "lon"])


# ── Signalements ──────────────────────────────────────────────────────────────
def lire_signalements():
    if os.path.exists(FICHIER_SIGNALEMENTS):
        return pd.read_csv(FICHIER_SIGNALEMENTS)
    return pd.DataFrame(columns=["date", "axe", "type", "description"])


def ajouter_signalement(axe, type_signalement, description):
    ligne = pd.DataFrame([{
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "axe": axe,
        "type": type_signalement,
        "description": description[:300],
    }])
    ligne.to_csv(FICHIER_SIGNALEMENTS, mode="a", index=False,
                 header=not os.path.exists(FICHIER_SIGNALEMENTS))


# ── Menu ──────────────────────────────────────────────────────────────────────
st.sidebar.markdown(LOGO + "  **Trafic Ziguinchor IA**", unsafe_allow_html=True)
page = st.sidebar.radio("Navigation", [
    "🔮 Prévision et carte", "📊 Statistiques", "🚨 Signalements",
    "🏛️ Alertes mairie", "ℹ️ À propos"])
st.sidebar.caption("Prototype IA - L3 Informatique UASZ\nDonnées simulées")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 : PRÉVISION ET CARTE
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔮 Prévision et carte":
    bandeau("Prévoir le trafic", "Choisissez un axe et un moment : le modèle estime le niveau de trafic.")

    c1, c2, c3 = st.columns(3)
    with c1:
        axe = st.selectbox("Axe / zone", axes)
        jour = st.selectbox("Jour", JOURS)
    with c2:
        heure = st.slider("Heure", 0, 23, 8)
        meteo = st.radio("Météo", ["Soleil", "Pluie"], horizontal=True)
    with c3:
        etats = ["Bon", "Moyen", "Dégradé"]
        etat = st.selectbox("État de la route", etats, index=etats.index(etat_par_axe[axe]))
        travaux = st.radio("Travaux en cours", ["Non", "Oui"], horizontal=True)

    X = pd.DataFrame([construire_ligne(axe, jour, meteo, etat, travaux, heure)])
    proba = dict(zip(modele.classes_, modele.predict_proba(X)[0]))
    niveau = max(proba, key=proba.get)

    st.divider()
    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Résultat")
        st.markdown(
            '<div class="resultat" style="background:' + COULEURS[niveau] + '">'
            + EMOJIS[niveau] + " " + niveau + "</div>",
            unsafe_allow_html=True,
        )
        st.caption(axe + " - " + jour + " à " + str(heure) + "h - " + meteo)
    with g2:
        st.subheader("Confiance du modèle")
        dp = pd.DataFrame({"Niveau": NIVEAUX,
                           "Probabilité (%)": [proba.get(n, 0) * 100 for n in NIVEAUX]})
        fig = px.bar(dp, x="Probabilité (%)", y="Niveau", orientation="h", color="Niveau",
                     color_discrete_map=COULEURS, range_x=[0, 100])
        fig.update_layout(showlegend=False, height=220, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig)

    # Carte : tous les axes au même moment
    st.divider()
    st.subheader("🗺️ Carte du trafic prévu : " + jour + " à " + str(heure) + "h")
    lignes = []
    for a in axes:
        if a == axe:
            lignes.append(construire_ligne(a, jour, meteo, etat, travaux, heure))
        else:
            lignes.append(construire_ligne(a, jour, meteo, etat_par_axe[a], "Non", heure))
    carte = pd.DataFrame({"axe": axes, "Niveau": modele.predict(pd.DataFrame(lignes))})
    carte = ajouter_cartes_coords(carte)
    if carte.empty:
        st.info("Ajoutez le fichier axes_coordonnees.csv pour afficher la carte.")
    else:
        fig_carte = px.scatter_map(
            carte, lat="lat", lon="lon", color="Niveau", hover_name="axe",
            color_discrete_map=COULEURS, category_orders={"Niveau": NIVEAUX},
            zoom=12, height=450, map_style="open-street-map",
        )
        fig_carte.update_traces(marker=dict(size=20))
        fig_carte.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_carte)

    # Journée complète
    st.divider()
    st.subheader("📅 Journée complète : " + axe + ", " + jour)
    jour_df = pd.DataFrame([construire_ligne(axe, jour, meteo, etat, travaux, h) for h in range(24)])
    jour_df["Niveau prévu"] = modele.predict(jour_df)
    jour_df["Valeur"] = 1
    fig24 = px.bar(jour_df, x="heure", y="Valeur", color="Niveau prévu",
                   color_discrete_map=COULEURS, category_orders={"Niveau prévu": NIVEAUX})
    fig24.update_layout(yaxis_visible=False, height=260, margin=dict(l=0, r=0, t=10, b=0),
                        xaxis=dict(dtick=1, title="Heure"))
    st.plotly_chart(fig24)
    fluides = jour_df.loc[jour_df["Niveau prévu"] == "Fluide", "heure"].tolist()
    if fluides:
        st.success("✅ Heures les plus fluides : " + ", ".join(str(h) + "h" for h in fluides))
    else:
        st.warning("⚠️ Aucune heure fluide prévue dans ces conditions.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 : STATISTIQUES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Statistiques":
    bandeau("Statistiques du trafic", "Analyse des données par axe, heure, jour et conditions.")

    st.sidebar.subheader("Filtres")
    axes_sel = st.sidebar.multiselect("Axes", axes, default=axes)
    jours_sel = st.sidebar.multiselect("Jours", JOURS, default=JOURS)
    meteo_sel = st.sidebar.multiselect("Météo", ["Soleil", "Pluie"], default=["Soleil", "Pluie"])

    d = df[df["axe"].isin(axes_sel) & df["jour"].isin(jours_sel) & df["meteo"].isin(meteo_sel)]
    if d.empty:
        st.warning("Aucune donnée pour ces filtres.")
        st.stop()

    heure_pic = int(d.groupby("heure")["nb_vehicules"].mean().idxmax())
    taux_axe = d["niveau_trafic"].eq("Embouteillé").groupby(d["axe"]).mean() * 100
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Observations", str(len(d)))
    k2.metric("Véhicules / heure (moy.)", str(round(d["nb_vehicules"].mean())))
    k3.metric("Part d'embouteillages", str(round(taux_embouteillage(d), 1)) + " %")
    k4.metric("Heure de pointe", str(heure_pic) + "h")
    st.caption("Axe le plus critique : **" + taux_axe.idxmax() + "** ("
               + str(round(taux_axe.max(), 1)) + " % d'embouteillages)")

    st.divider()
    a, b = st.columns(2)
    with a:
        st.subheader("Trafic moyen par heure")
        par_heure = d.groupby("heure")["nb_vehicules"].mean().reset_index()
        f1 = px.line(par_heure, x="heure", y="nb_vehicules", markers=True,
                     labels={"heure": "Heure", "nb_vehicules": "Véhicules / heure"})
        f1.update_layout(xaxis=dict(dtick=2), margin=dict(t=10))
        st.plotly_chart(f1)
    with b:
        st.subheader("Part d'embouteillages par axe")
        t = taux_axe.sort_values().reset_index()
        t.columns = ["Axe", "Embouteillage (%)"]
        f2 = px.bar(t, x="Embouteillage (%)", y="Axe", orientation="h",
                    color_discrete_sequence=["#D64545"])
        f2.update_layout(margin=dict(t=10))
        st.plotly_chart(f2)

    c, e = st.columns(2)
    with c:
        st.subheader("Répartition des niveaux")
        rep = d["niveau_trafic"].value_counts().reindex(NIVEAUX).fillna(0).reset_index()
        rep.columns = ["Niveau", "Nombre"]
        f3 = px.pie(rep, names="Niveau", values="Nombre", hole=0.5,
                    color="Niveau", color_discrete_map=COULEURS)
        f3.update_layout(margin=dict(t=10))
        st.plotly_chart(f3)
    with e:
        st.subheader("Effet de la pluie et des travaux")
        part = pd.DataFrame({
            "Condition": ["Soleil", "Pluie", "Sans travaux", "Avec travaux"],
            "Embouteillage (%)": [
                taux_embouteillage(d[d["meteo"] == "Soleil"]),
                taux_embouteillage(d[d["meteo"] == "Pluie"]),
                taux_embouteillage(d[d["travaux"] == "Non"]),
                taux_embouteillage(d[d["travaux"] == "Oui"]),
            ],
        })
        f4 = px.bar(part, x="Condition", y="Embouteillage (%)", color_discrete_sequence=["#3B5BDB"])
        f4.update_layout(margin=dict(t=10))
        st.plotly_chart(f4)

    st.subheader("Carte de chaleur : jour × heure")
    pivot = d.pivot_table(values="nb_vehicules", index="jour", columns="heure", aggfunc="mean")
    pivot = pivot.reindex([j for j in JOURS if j in pivot.index])
    f5 = px.imshow(pivot, color_continuous_scale="YlOrRd", aspect="auto",
                   labels=dict(x="Heure", y="", color="Véhicules"))
    f5.update_layout(margin=dict(t=10))
    st.plotly_chart(f5)

    with st.expander("Voir les données brutes"):
        st.dataframe(d.head(200))

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 : SIGNALEMENTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Signalements":
    bandeau("Signalements des usagers", "Signalez un accident, une route dégradée, des travaux ou une inondation.")

    with st.form("formulaire_signalement", clear_on_submit=True):
        f1c, f2c = st.columns(2)
        with f1c:
            axe_s = st.selectbox("Axe / zone concerné", axes)
        with f2c:
            type_s = st.selectbox("Type de signalement", TYPES)
        desc_s = st.text_area("Description (lieu précis, gravité...)", max_chars=300)
        st.caption("Ne saisissez aucune information personnelle (nom, numéro de téléphone, plaque).")
        envoye = st.form_submit_button("📤 Envoyer le signalement", type="primary")

    if envoye:
        ajouter_signalement(axe_s, type_s, desc_s.strip())
        st.success("✅ Signalement enregistré. Merci pour votre contribution.")

    sig = lire_signalements()
    st.divider()
    if sig.empty:
        st.info("Aucun signalement pour le moment.")
    else:
        m1, m2 = st.columns(2)
        m1.metric("Signalements reçus", str(len(sig)))
        m2.metric("Axe le plus signalé", str(sig["axe"].value_counts().idxmax()))

        par_axe = sig.groupby("axe").size().reset_index(name="Signalements")
        par_axe = ajouter_cartes_coords(par_axe)
        if not par_axe.empty:
            st.subheader("🗺️ Où sont les signalements ?")
            fs = px.scatter_map(par_axe, lat="lat", lon="lon", size="Signalements",
                                hover_name="axe", color_discrete_sequence=["#D64545"],
                                zoom=12, height=400, map_style="open-street-map")
            fs.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fs)

        st.subheader("Derniers signalements")
        st.dataframe(sig.tail(20).iloc[::-1], hide_index=True)

    st.caption("Note : sur un hébergement gratuit, ce fichier peut être réinitialisé lors d'un redémarrage. "
               "Une version finale utiliserait une vraie base de données.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 : ALERTES MAIRIE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏛️ Alertes mairie":
    bandeau("Alertes et recommandations", "Aide à la décision pour les autorités municipales.")

    seuil = st.slider("Seuil d'alerte : part d'embouteillages (%)", 5, 50, 15)

    taux_axe = (d_all := df)["niveau_trafic"].eq("Embouteillé").groupby(df["axe"]).mean() * 100
    taux_axe = taux_axe.sort_values(ascending=False)
    moy_heure = df.groupby(["axe", "heure"])["nb_vehicules"].mean().reset_index()
    pic_axe = moy_heure.sort_values("nb_vehicules").groupby("axe").tail(1).set_index("axe")["heure"]

    st.subheader("🚦 État des axes")
    for nom, t in taux_axe.items():
        h = int(pic_axe[nom])
        texte = (nom + " : " + str(round(t, 1)) + " % d'embouteillages, pic vers "
                 + str(h) + "h, route " + etat_par_axe[nom].lower() + ".")
        if t >= seuil:
            st.error("🔴 Alerte - " + texte)
        elif t >= seuil / 2:
            st.warning("🟠 Vigilance - " + texte)
        else:
            st.success("🟢 Normal - " + texte)

    st.divider()
    st.subheader("💡 Recommandations")
    pire = taux_axe.index[0]
    st.markdown("**1. Présence d'agents** : renforcer la régulation sur **" + pire + "** vers "
                + str(int(pic_axe[pire])) + "h, l'axe et le créneau les plus critiques.")

    pluie = taux_embouteillage(df[df["meteo"] == "Pluie"])
    soleil = taux_embouteillage(df[df["meteo"] == "Soleil"])
    st.markdown("**2. Météo** : la part d'embouteillages passe de " + str(round(soleil, 1))
                + " % (soleil) à " + str(round(pluie, 1)) + " % (pluie). "
                "Prévoir des messages d'alerte aux usagers les jours de pluie.")

    t_sans = taux_embouteillage(df[df["travaux"] == "Non"])
    t_avec = taux_embouteillage(df[df["travaux"] == "Oui"])
    creux = df.groupby("heure")["nb_vehicules"].mean().nsmallest(4).index.tolist()
    st.markdown("**3. Travaux** : " + str(round(t_sans, 1)) + " % d'embouteillages sans travaux contre "
                + str(round(t_avec, 1)) + " % avec. Programmer les chantiers aux heures les plus calmes ("
                + ", ".join(str(h) + "h" for h in sorted(creux)) + ").")

    degrades = [a for a in taux_axe.index if etat_par_axe[a] == "Dégradé"]
    if degrades:
        st.markdown("**4. Réfection des routes** : prioriser " + ", ".join("**" + a + "**" for a in degrades)
                    + " (route dégradée et ralentissements marqués).")

    sig = lire_signalements()
    if not sig.empty:
        top = sig["axe"].value_counts()
        st.markdown("**5. Signalements** : " + str(len(sig)) + " reçus, surtout sur **" + str(top.index[0])
                    + "** (" + str(int(top.iloc[0])) + "). À vérifier sur le terrain.")

    st.caption("Recommandations indicatives, calculées sur des données simulées.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 : À PROPOS
# ══════════════════════════════════════════════════════════════════════════════
else:
    bandeau("À propos du projet", "Module Innovations Technologiques en IA - L3 Informatique, UASZ")
    st.markdown("""
**Problème** : les embouteillages, l'état des routes et l'absence d'outils de suivi compliquent
les déplacements à Ziguinchor.

**Solution** : un prototype qui prévoit le niveau de trafic (Fluide, Dense, Embouteillé) selon l'axe,
l'heure, le jour, la météo, l'état de la route et les travaux. Il affiche une carte, collecte des
signalements d'usagers et propose des alertes à la mairie.

**Méthode** : jeu de données simulé, classification avec scikit-learn (modèle retenu par validation
croisée), dashboard Streamlit.
""")
    st.warning("""
**Limites**
- Les données sont **simulées** : les performances du modèle ne prouvent pas sa valeur sur le terrain.
- Pas de données en temps réel : la prévision repose sur des habitudes (heure, jour, météo).
- Les signalements ne sont pas vérifiés et ne sont pas stockés durablement.
- Les positions des axes sur la carte sont approximatives.
""")
    st.info("""
**Perspectives**
- Réentraîner le modèle sur de vraies mesures (comptages, capteurs, données de la mairie).
- Stocker les signalements dans une base de données et les modérer.
- Intégrer la météo réelle et les événements locaux (marché, matchs, fêtes).
""")
