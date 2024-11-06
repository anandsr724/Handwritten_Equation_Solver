FROM python:3.10.6-slim

RUN apt-get update && apt-get install -y \
    git \
    awscli \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "app.py"]