FROM python:3.10.6-slim

# Add retry logic and multiple package mirrors
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/80-retries && \
    echo "deb http://archive.debian.org/debian/ bullseye main" > /etc/apt/sources.list && \
    echo "deb http://archive.debian.org/debian-security/ bullseye/updates main" >> /etc/apt/sources.list && \
    apt-get update --allow-insecure-repositories && \
    apt-get install -y --no-install-recommends \
        git \
        awscli \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "app.py"]