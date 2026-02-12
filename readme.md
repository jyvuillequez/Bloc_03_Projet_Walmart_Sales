# Bloc 03 – Projet Walmart Sales

Analyse et prévision des ventes hebdomadaires (Weekly_Sales) afin de comprendre les effets **magasin**, **saisonnalité** et **jours fériés**, et d’améliorer la **prévision de la demande**.

Projet réalisé dans le cadre de la certification **RNCP Niveau 6 – Concepteur Développeur en Science des données (Jedha Bootcamp)**.

## 1. Contexte & enjeux

- **Problématique métier :** Walmart dispose de ventes hebdomadaires par magasin, ainsi que de variables externes (météo, carburant, CPI, chômage) et d’un indicateur de semaines fériées.
L’objectif est d’identifier les principaux leviers expliquant les variations de ventes et de construire un modèle de prédiction (généralisable).
- **Décideurs cibles :** marketing, supply, direction

## 2. Objectifs du projet

- Comprendrela distribution des ventes et la variabilité (pics, dispersion, outliers).
- Mesurer l’impact de la saisonnalité (mois / trimestre) et des semaines fériées.
- Construire un modèle de régression pour prédire `Weekly_Sales` et améliorer la généralisation via régularisation + validation croisée.
- Fournir une restitution claire (métriques + diagnostics + pistes d’optimisation).

## 3. Compétences mobilisées

- EDA (statistiques descriptives, distributions, saisonnalité, comparaisons)
- Préparation des données (valeurs manquantes, encodage, scaling)
- Modélisation (Baseline LinearRegression, Ridge, Lasso)
- Validation et sélection d’hyperparamètres (GridSearchCV, RMSE)
- Interprétation (coefficients / poids des features, diagnostic overfitting)
- Visualisation et restitution (Plotly, slides)

## 4. Données

**Périmètre :**
- Granularité : **semaine par magasin**
- Cible : `Weekly_Sales`
- Période : 2010 → 2012

**Variables clés :**
- `Store` : identifiant magasin
- `Date` : date de la semaine
- `Holiday_Flag` : indicateur semaine fériée (0/1)
- Variables externes : `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`
- Features dérivées (feature engineering) : `Year`, `Month`, `Quarter`, `Week`, `Is_Year_End`, `Holiday_prev1`, `Holiday_next1`

## 5. Méthodologie

1. **Préparation des données :**
- Parsing `Date` → extraction de variables calendaires (`Year`, `Month`, `Quarter`, `Week`, etc.)
- Création de “holiday window” :
  - `Holiday_prev1` = semaine précédente fériée (par magasin)
  - `Holiday_next1` = semaine suivante fériée (par magasin)
- Split **train/test** (baseline rapide) puis amélioration via **TimeSeriesSplit** (validation temporelle).

2. **Préprocessing (pipeline) :**
Pipeline scikit-learn reproductible :
- Numériques → `SimpleImputer(median)` + `StandardScaler`
- Catégorielles (`Store`, `Holiday_Flag`) → `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown="ignore")`
- Modèle final : **Ridge** (régularisation L2)

3. **Modélisation et évaluation :**
- Baseline : `LinearRegression`
- Modèles régularisés :
  - **Ridge** (L2) + GridSearchCV sur `alpha`
  - **Lasso** (L1) + GridSearchCV sur `alpha`

4. **Métriques :**
- **RMSE** : erreur moyenne en unités métier (ventes), pénalise fortement les grosses erreurs (pics).
- **R²** : part de variance expliquée (capacité à capter un signal au-delà d’une moyenne).

## 6. Résultats

- La baseline colle très fort au train mais “décroche” sur test (RMSE test élevé) ⇒ **overfitting**.
- Ridge améliore nettement la perfornance test (RMSE en diminution, R² en augmentation) : régularisation utile sur ce dataset.
- Lasso est proche de Ridge mais légèrement moins performant ici.

## 7. Installations des librairies Python
```text
python -m pip install -r requirements.txt
```

## 8. Organisation du projet

```text
.
├─ data/
│  ├─ raw/       # données brutes
│  └─ outputs/   # exports csv / features
├─ notebooks/    # notebooks Jupyter (EDA + modèles)
└─ présentation/ # slides, exports captures d'images
