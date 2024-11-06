FROM python:3.11-slim

RUN apt update -y && apt install awscli -y
WORKDIR /app

COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "app.py"]