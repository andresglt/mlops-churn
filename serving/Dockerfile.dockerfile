FROM python:3.11-slim

WORKDIR /app

# Copiar requirements y dependencias
COPY serving/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el repo (incluye serving, data, models, dvc.yaml, etc.)
COPY . /app

ENV PORT=8000
EXPOSE 8000

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]