FROM python:3.11-slim

# Instalar dependencias del sistema necesarias para TensorFlow
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libgomp1 \
#     && rm -rf /var/lib/apt/lists/*

# Crear usuario que ejecuta el dash
RUN adduser --disabled-password --gecos '' dash-user

WORKDIR /app

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/tablero.py .
COPY app/model_data.csv .
COPY app/colombia.json .
COPY app/run.sh .

# Copiar archivos del modelo
COPY app/model.keras .
COPY app/scaler.pkl .
COPY app/X_columns.pkl .

# Hacer el directorio de trabajo ejecutable
RUN chmod +x /app/run.sh
# Cambiar propiedad de la carpeta a dash-user
RUN chown -R dash-user:dash-user ./

USER dash-user

EXPOSE 8000

CMD ["bash", "./run.sh"]