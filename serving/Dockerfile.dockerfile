FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias
COPY serving/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiar toda la carpeta serving
COPY serving /app/serving

ENV PORT=8000
EXPOSE 8000

# Ejecutar FastAPI desde el paquete serving
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]