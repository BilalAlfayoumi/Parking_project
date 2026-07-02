# Dashboard Streamlit (Melbourne + Bordeaux) — image de déploiement Fly.io
FROM python:3.11-slim

WORKDIR /app

# Dépendances Python (couche cachée tant que requirements ne change pas).
# requirements.txt contient "-e ." → besoin de pyproject + src présents.
COPY requirements.txt pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir -r requirements.txt

# Artefacts nécessaires au dashboard uniquement (pas le zip brut Melbourne).
COPY models/model.joblib ./models/model.joblib
COPY data/interim/melbourne_street_geo.parquet ./data/interim/melbourne_street_geo.parquet
COPY data/history/bordeaux_parkings.parquet ./data/history/bordeaux_parkings.parquet

EXPOSE 8501

CMD ["streamlit", "run", "src/parking/dashboard/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
