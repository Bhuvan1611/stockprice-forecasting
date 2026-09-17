FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_DEBUG=0
EXPOSE 5000

CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-10000} -w 2 --timeout 120 app:app"]
