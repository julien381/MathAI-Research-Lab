# Utilisation d'une image Python légère officielle
FROM python:3.10-slim

# Définition du dossier de travail dans le conteneur
WORKDIR /app

# Copie d'abord le fichier des dépendances pour maximiser le cache Docker
COPY requirements.txt .

# Installation des dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copie du reste des dossiers nécessaires à l'API
COPY ./app ./app
COPY ./models ./models

# Exposition du port utilisé par FastAPI
EXPOSE 8000

# Commande de lancement de l'API via le gestionnaire 'py' ou directement python
CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
