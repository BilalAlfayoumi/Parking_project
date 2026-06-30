# 🅿️ Prédicteur d'occupation de stationnement

[![CI](https://github.com/BilalAlfayoumi/Parking_project/actions/workflows/ci.yml/badge.svg)](https://github.com/BilalAlfayoumi/Parking_project/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11+-blue)

> Prédire où trouver une place de stationnement, à une heure et un jour donnés —
> pipeline ML de bout en bout : données → modèle → API → dashboard → automatisation.

Le projet couvre **deux villes**, chacune avec sa source de données réelle :

| Volet | Question | Donnée | Modèle (R²) | vs baseline |
| --- | --- | --- | --- | --- |
| 🇦🇺 **Melbourne** | Cette **rue** sera-t-elle libre ? | capteurs en voirie (2019, 42 M lignes) | **0,46** | +17 % |
| 🇫🇷 **Bordeaux** | Ce **parking** sera-t-il plein ? | flux temps réel, historisé par le projet | **0,73** | +52 % |

## Ce que fait le projet

- **Melbourne** — un modèle entraîné sur l'historique massif des capteurs de
  stationnement en voirie prédit le taux d'occupation d'une rue par tranche de 30 min.
- **Bordeaux** — un robot collecte en continu la disponibilité des parkings
  (open data temps réel), constitue un historique, et un modèle prédit leur
  remplissage futur. Utile concrètement pour anticiper où se garer.

Les deux sont exposés dans un **dashboard Streamlit** (carte interactive) et via une
**API FastAPI**.

## Démarche (et ce que le projet démontre)

Au-delà du code, le projet applique une méthode rigoureuse :

- **Baseline obligatoire.** Tout modèle est comparé à une baseline naïve (moyenne
  historique). Un modèle qui ne la bat pas est un signal, pas un détail.
- **Validation temporelle.** Split train/test *dans le temps* (jamais aléatoire) :
  on prédit le futur, pas un trou dans le passé. Cross-validation `TimeSeriesSplit`,
  *learning curve* et *validation curve* pour diagnostiquer sur/sous-apprentissage.
- **Transférabilité testée — et limite assumée.** J'ai testé si un modèle entraîné
  sur une ville se généralise à une autre (Melbourne → Seattle, features de contexte
  via OpenStreetMap). **Ça échoue** (R² négatif) : le niveau d'occupation dépend de
  facteurs locaux (tarifs, demande) qu'un modèle ne transfère pas. C'est ce constat
  qui a motivé l'approche Bordeaux : collecter la *vraie* donnée locale plutôt que de
  transférer un modèle. Documenter cette limite fait partie du livrable.

## Stack

Python · pandas · scikit-learn (HistGradientBoosting) · FastAPI · Streamlit ·
pytest · Ruff · mypy · uv · GitHub Actions.

## Installation

Prérequis : [uv](https://docs.astral.sh/uv/) et `make`.

```bash
make install      # venv + dépendances + hooks pre-commit
```

## Commandes

| Commande | Effet |
| --- | --- |
| `make train` | Entraîne le modèle Melbourne (voirie) |
| `make train-bordeaux` | Entraîne le modèle de prévision des parkings Bordeaux |
| `make api` | Lance l'API FastAPI (`/predict`, `/streets`, `/health`) |
| `make dashboard` | Lance le dashboard Streamlit (onglets Melbourne + Bordeaux) |
| `make test` | Lance les tests (pytest) |
| `make lint` | Vérifie le code (Ruff lint + format) |

## Automatisation

Deux workflows GitHub Actions :

- **`ci.yml`** — à chaque push : lint + tests (garde le code sain).
- **`collect.yml`** — toutes les ~30 min : relève les parkings de Bordeaux et
  committe l'historique. C'est ce robot qui constitue le jeu de données
  d'entraînement du modèle Bordeaux, en continu.

## Architecture

```
src/parking/
  ingestion.py       Lecture en streaming des données brutes Melbourne (zip)
  features.py        Nettoyage capteurs + reconstruction de l'occupation
  baseline.py        Baseline naïve (référence à battre)
  model.py           Entraînement Melbourne + make train
  evaluation.py      Métriques (MAE/RMSE/R²) + comparaison baseline
  predict.py         Inférence Melbourne
  bordeaux.py        Ingestion du flux temps réel des parkings Bordeaux
  collect.py         Collecteur d'historique (lancé par GitHub Actions)
  bordeaux_model.py  Prévision du remplissage des parkings Bordeaux
  api/               API FastAPI
  dashboard/         Dashboard Streamlit (Melbourne + Bordeaux)
tests/               Tests pytest
```

## Limites assumées

- **Voirie à Bordeaux : non couverte.** Aucune donnée d'occupation en voirie n'existe
  en open data pour Bordeaux (ni capteurs, ni transactions). Le volet Bordeaux porte
  donc sur les **parkings en ouvrage**, seule donnée locale réelle disponible.
- **Historique Bordeaux jeune.** Le modèle s'améliore à mesure que le robot collecte ;
  les premiers résultats reposent sur quelques jours de données.
- **Pas de transfert entre villes** (voir *Démarche*) : chaque ville a son modèle.

## Données

- Melbourne : *On-street Car Parking Sensor Data 2019*, City of Melbourne Open Data.
- Bordeaux : *Parking hors voirie* (`st_park_p`), Bordeaux Métropole Open Data.
