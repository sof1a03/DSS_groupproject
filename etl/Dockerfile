FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY run_all.py test_db.py ./
CMD ["python", "-u", "run_all.py"]
