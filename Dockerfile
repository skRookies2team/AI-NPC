FROM python:3.11-slim

WORKDIR /app

COPY AI/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY AI/ .

EXPOSE 8000

CMD ["uvicorn", "ai_server:app", "--host", "0.0.0.0", "--port", "8000"]

# test
