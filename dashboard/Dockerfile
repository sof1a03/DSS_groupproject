# dashboard/Dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir flask
EXPOSE 5000
CMD ["python", "-c", "from flask import Flask; app=Flask(__name__); \
@app.get('/')\ndef hi(): return 'hello from flask'; \
app.run(host='0.0.0.0', port=5000)"]
