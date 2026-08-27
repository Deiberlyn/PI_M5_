FROM python:3.13-slim

WORKDIR /app

# Copia de requerimientos (librerias) y dependendias
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copia de todo el proyecto
COPY . .

# Puerto de FastAPI
EXPOSE 8000

# Comando para ejecutar la API
CMD ["uvicorn", "mlops_pipeline.src.model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]