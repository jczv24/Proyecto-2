FROM python:3.11-slim

# Crear usuario que ejecuta el dash
RUN adduser --disabled-password --gecos '' dash-user

WORKDIR /app

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/tablero.py .
COPY app/model_data.csv .
COPY app/colombia.json .
COPY app/run.sh .

# Hacer el directorio de trabajo ejecutable
RUN chmod +x /app/run.sh
# Cambiar propiedad de la carpeta a dash-user
RUN chown -R dash-user:dash-user ./

USER dash-user

EXPOSE 8000

CMD ["bash", "./run.sh"]