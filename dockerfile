# Base image: minimal Python 3.11
FROM python:3.11-slim

# Everything below runs relative to /app inside the container
WORKDIR /app

# Copy dependency list first, install — cached separately from app code
# so code changes don't force a full reinstall of sklearn/xgboost/etc.
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Only what's needed to SERVE predictions — not ml/, notebooks/, data/
COPY app/ ./app/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]