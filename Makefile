.PHONY: install lint format test train api dashboard freeze clean

install:  ## Installe les dépendances et les hooks pre-commit
	uv sync
	uv run pre-commit install

lint:  ## Vérifie le code (Ruff lint + format check)
	uv run ruff check .
	uv run ruff format --check .

format:  ## Formate le code
	uv run ruff format .
	uv run ruff check --fix .

test:  ## Lance les tests
	uv run pytest

train:  ## Entraîne et sauvegarde le modèle Melbourne (jalon semaine 4)
	uv run python -m parking.model

train-bordeaux:  ## Entraîne le modèle de prévision des parkings Bordeaux
	uv run python -m parking.bordeaux_model

api:  ## Démarre l'API FastAPI en local
	uv run uvicorn parking.api.main:app --reload

dashboard:  ## Lance le dashboard Streamlit
	uv run streamlit run src/parking/dashboard/app.py

freeze:  ## Exporte requirements.txt pour Streamlit Community Cloud
	uv export --no-hashes --no-dev --format requirements-txt -o requirements.txt

clean:  ## Supprime les caches
	rm -rf .ruff_cache .mypy_cache .pytest_cache htmlcov .coverage
